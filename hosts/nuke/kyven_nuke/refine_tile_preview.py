"""Debounced CPU tile previews; Nuke exports and Read updates stay on the main thread."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kyven_nuke import config
from kyven_nuke.color_management import set_data_io
from kyven_nuke.knob_tools import suspend_callbacks
from kyven_nuke.node import _inside, _nuke, _nuke_file_path, _refresh_node_output
from kyven_nuke.refine_tiling import preview_info
from kyven_nuke.runtime import ensure_server

_lock = threading.RLock()
_revisions: dict[str, int] = {}
_frames: dict[str, int] = {}
_running: set[str] = set()
_scheduled: set[str] = set()
_owned_outputs: set[Path] = set()
_input_versions: dict[str, int] = {}
_guidance_keys: dict[str, tuple] = {}
_input_users: dict[Path, int] = {}


@dataclass(frozen=True)
class InputSnapshot:
    frame: int
    version: int
    size: tuple[int, int]
    path: Path


_snapshots: dict[str, InputSnapshot] = {}
PREVIEW_KNOBS = {
    'output_mode', 'inputChange', 'profile', 'tile_mode', 'tile_size', 'tile_overlap',
    'skip_known_tiles', 'generate_trimap', 'foreground_radius', 'background_radius',
    'crop_mode', 'processing_roi', 'context_padding', 'roi_enabled',
    'tile_preview_labels',
}


def is_preview(node: Any) -> bool:
    return 'output_mode' in node.knobs() and str(node['output_mode'].value()) == 'Tile Preview'


def show_preview() -> None:
    node = _nuke().thisNode()
    set_display(node, True, refresh_input=True)


@suspend_callbacks
def sync_display_control(node: Any) -> None:
    """Keep the checkbox and the existing Output selector in agreement."""

    if not {'show_tiles', 'tile_preview_previous_output', 'output_mode'}.issubset(node.knobs()):
        return
    selected = str(node['output_mode'].value())
    node['show_tiles'].setValue(selected == 'Tile Preview')
    if selected != 'Tile Preview':
        node['tile_preview_previous_output'].setValue(selected)


@suspend_callbacks
def set_display(node: Any, enabled: bool | None = None, *, refresh_input: bool = False) -> None:
    """Toggle the preview and return to the artist's previous output without inference."""

    enabled = bool(node['show_tiles'].value()) if enabled is None else enabled
    selected = str(node['output_mode'].value())
    if enabled:
        if selected != 'Tile Preview':
            node['tile_preview_previous_output'].setValue(selected)
        node['show_tiles'].setValue(True)
        node['output_mode'].setValue('Tile Preview')
        request_preview(node, force=True, refresh_input=refresh_input)
    else:
        node['show_tiles'].setValue(False)
        if selected == 'Tile Preview':
            previous = str(node['tile_preview_previous_output'].value())
            choices = list(node['output_mode'].values())
            if previous not in choices or previous == 'Tile Preview':
                previous = 'Source + Refined Alpha'
            node['output_mode'].setValue(previous)
        with _lock:
            name = str(node.fullName())
            _revisions[name] = _revisions.get(name, 0) + 1
        if 'Trimap' in str(node['output_mode'].value()):
            from kyven_nuke.refine_node import request_trimap_preview
            request_trimap_preview(node)
        _refresh_node_output(node, clear_cache=False)


@suspend_callbacks
def request_preview(node: Any, *, force: bool = False, refresh_input: bool = False,
                    refresh_trimap: bool = False) -> None:
    """Follow the timeline or debounce settings changes without running a model."""

    if not is_preview(node):
        return
    name, frame = str(node.fullName()), int(_nuke().frame())
    with _lock:
        if not force and _frames.get(name) == frame:
            return
        frame_changed = _frames.get(name) != frame
        if refresh_input or frame_changed:
            _input_versions[name] = _input_versions.get(name, 0) + 1
        guidance = (_input_versions.get(name, 0), bool(node['generate_trimap'].value()),
                    int(node['foreground_radius'].value()), int(node['background_radius'].value()))
        bounds_invalid = refresh_trimap or _guidance_keys.get(name) != guidance
        _guidance_keys[name] = guidance
        _frames[name] = frame
        revision = _revisions.get(name, 0) + 1
        _revisions[name] = revision
    if bounds_invalid:
        node['tile_preview_bounds_valid'].setValue(False)
    switch = _inside(node, 'KyvenTilePreviewSwitch')
    if switch is not None:
        switch['which'].setValue(0)
    node['tile_preview_info'].setValue(preview_info('Updating colors...'))
    _schedule(name, revision, 0.03)


def _schedule(name: str, revision: int, delay: float) -> None:
    # Throttle slider events into one main-thread dispatch rather than starting a
    # new timer for every intermediate value. An in-flight worker starts the
    # latest revision when it finishes, without another debounce delay.
    del revision
    with _lock:
        if name in _scheduled or name in _running:
            return
        _scheduled.add(name)
    timer = threading.Timer(delay, lambda: _nuke().executeInMainThread(_dispatch, args=(name,)))
    timer.daemon = True
    timer.start()


def _dispatch(name: str) -> None:
    with _lock:
        _scheduled.discard(name)
        revision = _revisions.get(name)
    if revision is not None:
        _start(name, revision)


def _start(name: str, revision: int) -> None:
    with _lock:
        if _revisions.get(name) != revision or name in _running:
            return
    nuke = _nuke()
    node = nuke.toNode(name)
    if node is None or not is_preview(node):
        return
    if nuke.executing() or bool(node['kyven_busy'].value()):
        _schedule(name, revision, 0.2)
        return
    from kyven_nuke import refine_node as adapter
    source, mask = node.input(adapter.SOURCE_INPUT), node.input(adapter.MASK_INPUT)
    if source is None or mask is None:
        node['tile_preview_info'].setValue(preview_info('Connect Source and Mask.'))
        return
    frame = int(nuke.frame())
    root = config.runtime_dir() / 'refine_tile_previews' / str(node['kyven_uuid'].value())
    token = f'{frame:04d}_{revision}_{uuid.uuid4().hex[:8]}'
    mask_path, output = root / f'mask_{token}.tif', root / f'tiles_{token}.png'
    with _lock:
        _running.add(name)
        _owned_outputs.add(output.resolve())
    try:
        root.mkdir(parents=True, exist_ok=True)
        size = (int(source.width()), int(source.height()))
        with _lock:
            version = _input_versions.get(name, 0)
            cached = _snapshots.get(name)
        if (cached is not None and cached.frame == frame and cached.version == version
                and cached.size == size and cached.path.is_file()):
            mask_path = cached.path
        else:
            writer = _inside(node, 'KyvenRefineMaskWrite')
            adapter._configure_export_writers(None, writer)
            previous = str(writer['file'].value())
            try:
                writer['file'].setValue(_nuke_file_path(mask_path))
                nuke.execute(writer, frame, frame)
            finally:
                writer['file'].setValue(previous)
            with _lock:
                _snapshots[name] = InputSnapshot(frame, version, size, mask_path)
                old_in_use = cached is not None and bool(_input_users.get(cached.path, 0))
            if cached is not None and not old_in_use:
                _remove(cached.path)
        with _lock:
            _input_users[mask_path] = _input_users.get(mask_path, 0) + 1
        raw = adapter._payload(node, source, mask_path, mask_path, output, output, frame)
        keys = ('mask', 'output', 'profile', 'crop_mode', 'roi', 'context_padding',
                'generate_trimap', 'foreground_radius', 'background_radius',
                'tile_size', 'tile_overlap', 'skip_known_tiles')
        payload = {key: raw[key] for key in keys}
        payload.update(image_width=int(source.width()), image_height=int(source.height()))
        payload['preview_labels'] = bool(node['tile_preview_labels'].value())
    except Exception as exc:  # noqa: BLE001 - Nuke callback boundary.
        _finish(name, revision, frame, mask_path, output, None, str(exc))
        return
    threading.Thread(target=_worker, args=(name, revision, frame, mask_path, output, payload),
                     name='kyven-refine-tile-preview', daemon=True).start()


def _worker(name: str, revision: int, frame: int, mask: Path, output: Path, payload: dict) -> None:
    result, error = None, ''
    try:
        result = ensure_server().preview_refine_tiles(payload)
    except Exception as exc:  # noqa: BLE001 - HTTP/thread boundary.
        error = str(exc)
    _nuke().executeInMainThread(_finish, args=(name, revision, frame, mask, output, result, error))


def _remove(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass  # The Windows Viewer can briefly retain an old Read file.
    with _lock:
        _owned_outputs.discard(path.resolve())


def _release_input(name: str, mask: Path, *, retain: bool) -> None:
    with _lock:
        remaining = max(0, _input_users.get(mask, 0) - 1)
        if remaining:
            _input_users[mask] = remaining
        else:
            _input_users.pop(mask, None)
        current = _snapshots.get(name)
        if not retain:
            _snapshots.pop(name, None)
        keep = retain and current is not None and current.path == mask
    if not remaining and not keep:
        _remove(mask)


def _finish(name: str, revision: int, frame: int, mask: Path, output: Path,
            result: dict | None, error: str) -> None:
    node = _nuke().toNode(name)
    _release_input(name, mask, retain=node is not None)
    with _lock:
        _running.discard(name)
        current = _revisions.get(name)
    if node is None or not is_preview(node) or current != revision:
        _remove(output)
        if node is not None and is_preview(node) and current != revision:
            _start(name, current)
        return
    if error or result is None:
        node['tile_preview_info'].setValue(preview_info('Unavailable; see Status for details.'))
        node['kyven_status'].setValue(f'Tile preview failed: {error}')
        _remove(output)
        return
    _apply_result(node, frame, output, result)


@suspend_callbacks
def _apply_result(node: Any, frame: int, output: Path, result: dict) -> None:
    """Publish only the changed preview branch, preserving Nuke's upstream cache."""

    read = _inside(node, 'KyvenTilePreviewRead')
    previous = str(read['file'].value())
    read['file'].setValue(_nuke_file_path(output))
    for knob in ('first', 'last', 'origfirst', 'origlast'):
        read[knob].setValue(frame)
    # Each overlay has a unique filename. Changing that path already invalidates
    # this Read; reload/clearRAMCache would discard the entire comp's image cache.
    bounds = result['trimap_bounds']
    height = int(result['processing_roi']['source_height'])
    node['tile_preview_bounds'].setValue(
        [bounds['x0'], height-bounds['y1'], bounds['x1'], height-bounds['y0']])
    node['tile_preview_bounds_valid'].setValue(True)
    _inside(node, 'KyvenTilePreviewSwitch')['which'].setValue(1)
    roi = result['processing_roi']
    node['tile_preview_info'].setValue(preview_info(
        f"{result['tiles']}/{result['tiles_total']} run | "
        f"{result['tiles_skipped']} skip | {result['unknown_pixels']:,} unknown px | "
        f"Crop {roi['width']}x{roi['height']}"))
    if not bool(node['kyven_busy'].value()):
        node['kyven_status'].setValue(
            f"Tile preview only - {result['tiles']}/{result['tiles_total']} tiles run; "
            f"{result['tiles_skipped']} skipped. No model inference.")
    _refresh_node_output(node, clear_cache=False)
    if previous and previous != _nuke_file_path(output):
        old = Path(previous).resolve()
        with _lock:
            owned = old in _owned_outputs
        if owned:
            _remove(old)


def build_graph(nuke: Any, source: Any) -> Any:
    """Overlay only; cached matte and trimap branches are independent."""

    read = nuke.toNode('KyvenTilePreviewRead')
    if read is None:
        read = nuke.nodes.Read(name='KyvenTilePreviewRead')
        set_data_io(read)
    premult = nuke.toNode('KyvenTilePreviewPremult')
    if premult is None:
        premult = nuke.nodes.Premult(name='KyvenTilePreviewPremult')
    premult.setInput(0, read)
    merge = nuke.toNode('KyvenTilePreviewOverlay')
    if merge is None:
        merge = nuke.nodes.Merge2(name='KyvenTilePreviewOverlay', operation='over')
    merge.setInput(0, source)
    merge.setInput(1, premult)
    diagnostic = nuke.toNode('KyvenTilePreviewAlpha')
    if diagnostic is None:
        diagnostic = nuke.nodes.Copy(name='KyvenTilePreviewAlpha')
    diagnostic.setInput(0, merge)
    diagnostic.setInput(1, read)
    diagnostic['from0'].setValue('rgba.alpha')
    diagnostic['to0'].setValue('rgba.alpha')
    switch = nuke.toNode('KyvenTilePreviewSwitch')
    if switch is None:
        switch = nuke.nodes.Switch(name='KyvenTilePreviewSwitch')
    fallback = nuke.toNode('KyvenTilePreviewSource')
    if fallback is None:
        fallback = nuke.nodes.Expression(name='KyvenTilePreviewSource')
    fallback.setInput(0, source)
    fallback['expr3'].setValue('0')
    switch.setInput(0, fallback)
    switch.setInput(1, diagnostic)
    from kyven_nuke.refine_tile_grid import build_graph as build_live_grid
    return build_live_grid(nuke, source, switch)
