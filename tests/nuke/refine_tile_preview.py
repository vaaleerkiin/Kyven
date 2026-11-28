"""Real Nuke export -> CPU HTTP planner -> Viewer overlay, with inference forbidden."""
import json
import os
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'hosts/nuke'), str(ROOT/'src')]
import kyven_nuke  # noqa: F401 - serialized callbacks use this name.
import nuke
from kyven_nuke import live, refine_node
from kyven_nuke import refine_tile_preview as preview
from kyven_nuke.client import NukeKyvenClient

report = {'checks': []}


def check(name, condition):
    assert condition, name
    report['checks'].append(name)


def forbid(*args, **kwargs):
    raise AssertionError('Tile Preview must not request model inference')


class InlineWorker:
    def __init__(self, target, args, **kwargs):
        self.target, self.args = target, args

    def start(self):
        self.target(*self.args)


def run():
    root = Path(os.environ['KYVEN_NUKE_TILE_PREVIEW_REPORT']).parent
    # This headless harness invokes the worker inline, already on Nuke's main thread.
    nuke.executeInMainThread = lambda call, args=(), kwargs=None: call(*args, **(kwargs or {}))
    preview._schedule = lambda *args: None  # Dispatch explicitly for deterministic headless checks.
    preview.threading = SimpleNamespace(Thread=InlineWorker)
    preview.config = SimpleNamespace(runtime_dir=lambda: root)
    client = NukeKyvenClient(os.environ['KYVEN_NUKE_TILE_PREVIEW_URL'], 'x'*32)
    plans = []

    class PreviewClient:
        def preview_refine_tiles(self, payload):
            result = client.preview_refine_tiles(payload)
            plans.append(result)
            return result

    preview.ensure_server = lambda: PreviewClient()
    fmt = nuke.addFormat('384 384 0 0 384 384 1 TilePreviewQA')
    source = nuke.nodes.Constant(format=fmt.name())
    source['color'].setValue([0.1, 0.2, 0.3, 1])
    mask = nuke.nodes.Expression(inputs=[source], expr3='x<180 ? 0 : x<184 ? 0.5 : 1')
    node = refine_node.create_refine_node()
    node.setInput(refine_node.SOURCE_INPUT, source)
    node.setInput(refine_node.MASK_INPUT, mask)
    node['crop_mode'].setValue('manual')
    node['processing_roi'].setValue([50, 40, 350, 344])
    node['generate_trimap'].setValue(False)
    node['tile_mode'].setValue('Tiles')
    node['tile_size'].setValue(128)
    check('128 px tiles automatically reduce the default 64 px overlap to 32',
          node['tile_overlap'].value() == 32)
    identity = node['kyven_uuid'].value()
    source_file = node.node('KyvenRefineSourceWrite')['file'].value()
    mask_file = node.node('KyvenRefineMaskWrite')['file'].value()
    refine_node.process_current_frame = forbid
    live.request_live_update = forbid
    nuke.clearRAMCache = forbid
    node['live_mode'].setValue(True)
    node['output_mode'].setValue('Tile Preview')
    check('selecting Tile Preview synchronizes Show Tiles', node['show_tiles'].value())
    exports = []
    original_execute = nuke.execute

    def counted_execute(writer, *args, **kwargs):
        if writer.name() == 'KyvenRefineMaskWrite':
            exports.append(writer['file'].value())
        return original_execute(writer, *args, **kwargs)

    nuke.execute = counted_execute
    check('pending preview already shows white square outlines without a server call',
          node.sample('rgba.alpha', 177, 100) == 1
          and node.sample('rgba.alpha', 160, 100) == 0 and not plans)
    pending_info = node['tile_preview_info'].value()
    check('loading keeps the color legend and all three interface rows',
          pending_info.count('<br>') == 2 and 'Green: RUN' in pending_info
          and 'Yellow: unknown' in pending_info)
    check('yellow unknown pixels are visible before any server result',
          node.sample('rgba.red', 181, 100) > 0.7
          and abs(node.sample('rgba.alpha', 181, 100)-200/255) < 1e-6)
    node['generate_trimap'].setValue(True)
    node['foreground_radius'].setValue(1)
    node['background_radius'].setValue(1)
    before_grow = node.sample('rgba.alpha', 173, 100)
    node['background_radius'].setValue(12)
    check('trimap sliders update yellow immediately without export or server work',
          before_grow == 0 and abs(node.sample('rgba.alpha', 173, 100)-200/255) < 1e-6
          and not exports and not plans)
    node['generate_trimap'].setValue(False)

    def refresh():
        preview.request_preview(node, force=True)
        preview._start(node.fullName(), preview._revisions[node.fullName()])

    refresh()
    check('initial preview exports the mask once', len(exports) == 1)
    check('preview is the appended eighth output', int(node['output_mode'].getValue()) == 7)
    check('CPU plan counts active and skipped tiles', plans[-1]['tiles'] == 3
          and plans[-1]['tiles_total'] == 9 and plans[-1]['tiles_skipped'] == 6)
    check('preview preserves the exact Manual crop', plans[-1]['processing_roi']['x0'] == 50
          and plans[-1]['processing_roi']['width'] == 300)
    check('Viewer composites the yellow unknown band over Source', node.sample('rgba.red', 181, 100) > 0.5)
    check('asynchronous colors match the premultiplied native yellow overlay',
          abs(node.sample('rgba.red', 181, 100)-(200/255 + 0.1*(1-200/255))) < 1e-6)
    check('Alpha Viewer also shows the diagnostic plan', 0.7 < node.sample('rgba.alpha', 181, 100) < 0.9)
    check('preview keeps full Source canvas', node.width() == 384 and node.height() == 384)
    check('preview restores both export-writer paths', source_file == node.node('KyvenRefineSourceWrite')['file'].value()
          and mask_file == node.node('KyvenRefineMaskWrite')['file'].value())
    check('preview does not install a cached matte', node.node('KyvenMatteRead') is None)
    check('node exposes the actual preview counts', '3/9 run' in node['tile_preview_info'].value())
    check('ready and loading states reserve the same interface rows',
          node['tile_preview_info'].value().count('<br>') == pending_info.count('<br>'))

    node['tile_overlap'].setValue(64)
    refresh()
    check('failed preview keeps the same rows and the color legend',
          node['tile_preview_info'].value().count('<br>') == 2
          and 'Green: RUN' in node['tile_preview_info'].value()
          and 'Tile preview failed' in node['kyven_status'].value())
    node['tile_overlap'].setValue(32)
    refresh()

    def check_native_squares(index, tile, overlap, crop, roi=None, padding=12):
        node['tile_mode'].setValue('Tiles' if tile else 'Full Crop')
        if tile:
            node['tile_size'].setValue(tile)
        node['tile_overlap'].setValue(overlap)
        node['crop_mode'].setValue(crop)
        if roi is not None:
            node['processing_roi'].setValue(roi)
        node['context_padding'].setValue(padding)
        snapshot = preview._snapshots[node.fullName()].path
        raw = refine_node._payload(node, source, snapshot, snapshot,
            root/f'grid-{index}.png', root/'unused.png', int(nuke.frame()))
        expected = client.preview_refine_tiles(raw)
        node.node('KyvenTilePreviewSwitch')['which'].setValue(0)
        points_x, points_y = {0, 383}, {0, 383}
        for box in expected['boxes']:
            x0, y0, x1, y1 = (box[key] for key in ('x0', 'y0', 'x1', 'y1'))
            points_x.update((x0-1, x0, x0+2, x0+3, (x0+x1)//2, x1-4, x1-3, x1-1, x1))
            points_y.update((y0-1, y0, y0+2, y0+3, (y0+y1)//2, y1-4, y1-3, y1-1, y1))
        grid = node.node('KyvenTilePreviewGrid')
        for y in points_y:
            for x in points_x:
                if not (0 <= x < 384 and 0 <= y < 384):
                    continue
                edge = any(b['x0'] <= x < b['x1'] and b['y0'] <= y < b['y1'] and
                           min(x-b['x0'], b['x1']-1-x, y-b['y0'], b['y1']-1-y) < 3
                           for b in expected['boxes'])
                actual = grid.sample('rgba.alpha', x, 383-y)
                assert abs(actual-int(edge)) < 1e-6, (index, x, y, actual, edge)
        check(f'live square grid {index} matches every planned tile edge', True)

    check_native_squares(1, 128, 32, 'manual', [50.2, 40.7, 350.8, 344.1])
    check_native_squares(2, 192, 31, 'manual', [50, 40, 350, 344])
    check_native_squares(3, 512, 64, 'manual', [50, 40, 350, 344])
    check_native_squares(4, 0, 0, 'manual', [50, 40, 350, 344])
    check_native_squares(5, 128, 0, 'full')
    check_native_squares(6, 128, 32, 'auto', padding=12.9)
    check('moving geometry sliders never exports or reruns a model', len(exports) == 1)
    node['crop_mode'].setValue('manual')
    node['processing_roi'].setValue([50, 40, 350, 344])
    node['tile_size'].setValue(128)
    node['tile_overlap'].setValue(32)
    refresh()
    check('same-frame slider preview reuses prepared trimap', plans[-1]['trimap_cache_hit'])
    old_overlay = Path(plans[-1]['output'])
    node['skip_known_tiles'].setValue(False)
    refresh()
    check('disabling skips updates the displayed plan', plans[-1]['tiles'] == 9)
    check('known-tile toggle reuses the original snapshot', len(exports) == 1)
    node['tile_preview_labels'].setValue(True)
    refresh()
    check('optional labels are independent of the model and mask export',
          plans[-1]['preview_labels'] and len(exports) == 1)
    check('replaced owned overlays are cleaned up', not old_overlay.exists())
    foreign = root/'user-owned-overlay.png'
    foreign.write_bytes(Path(plans[-1]['output']).read_bytes())
    node.node('KyvenTilePreviewRead')['file'].setValue(str(foreign).replace('\\', '/'))
    refresh()
    check('preview cleanup preserves files it did not create', foreign.exists())
    node['tile_mode'].setValue('Full Crop')
    refresh()
    check('Full Crop preview shows a single pass', plans[-1]['tiles'] == plans[-1]['tiles_total'] == 1)
    live._request_revisions[node.fullName()] = 42
    node['kyven_live_frame'].setValue(-2147483647)
    live._run_requested_update(node.fullName(), 42)
    live._last_scan = 0
    live.update_live_nodes()
    check('queued and periodic Live callbacks do not infer in Tile Preview', True)
    nuke.frame(2)
    live._last_scan = 0
    live.update_live_nodes()
    check('timeline schedules a new CPU preview', preview._frames[node.fullName()] == 2)
    preview._start(node.fullName(), preview._revisions[node.fullName()])
    check('preview Read follows the frame', node.node('KyvenTilePreviewRead')['first'].value() == 2)
    check('new frame exports one new mask and discards the old snapshot', len(exports) == 2
          and not Path(exports[0]).exists())
    preview.request_preview(node, force=True, refresh_input=True)
    preview._start(node.fullName(), preview._revisions[node.fullName()])
    check('explicit refresh captures upstream edits', len(exports) == 3)
    snapshot = preview._snapshots[node.fullName()].path
    before_toggle = refine_node._payload(node, source, snapshot, snapshot,
        root/'toggle-result.png', root/'toggle-trimap.png', int(nuke.frame()))
    for output in ('Refined Matte', 'Source + Refined Alpha', 'Refined Cutout', 'Trimap',
                   'Source + Trimap Alpha', 'Trimap Cutout', 'Source (Bypass)'):
        node['output_mode'].setValue(output)
        check(f'{output} synchronizes display off', not node['show_tiles'].value())
        baseline = [node.sample(channel, 181, 100) for channel in ('rgba.red', 'rgba.alpha')]
        node['show_tiles'].setValue(True)
        check(f'Show Tiles enables preview over {output}', preview.is_preview(node))
        node['show_tiles'].setValue(False)
        check(f'hiding tiles restores {output}', node['output_mode'].value() == output
              and [node.sample(channel, 181, 100) for channel in ('rgba.red', 'rgba.alpha')] == baseline)
    check('display toggles preserve inference payload', before_toggle == refine_node._payload(
        node, source, snapshot, snapshot, root/'toggle-result.png', root/'toggle-trimap.png',
        int(nuke.frame())))
    check('display toggles do not re-export the mask', len(exports) == 3)
    node['show_tiles'].setValue(True)
    stale_revision = preview._revisions[node.fullName()]
    stale_output = root/'stale-preview.png'
    stale_output.write_bytes(Path(plans[-1]['output']).read_bytes())
    node['show_tiles'].setValue(False)
    preview._finish(node.fullName(), stale_revision, int(nuke.frame()), snapshot,
                    stale_output, plans[-1], '')
    check('late preview completion cannot turn the display back on',
          not node['show_tiles'].value() and node['output_mode'].value() == 'Source (Bypass)'
          and not stale_output.exists() and snapshot.exists())
    node['show_tiles'].setValue(True)
    refresh()
    script = root/'tile-preview.nk'
    name = node.name()
    nuke.scriptSaveAs(str(script), overwrite=1)
    nuke.scriptClear()
    nuke.scriptOpen(str(script))
    node = nuke.toNode(name)
    check('tile preview survives reopen', node['output_mode'].value() == 'Tile Preview'
          and node['show_tiles'].value()
          and node['kyven_uuid'].value() == identity and node.sample('rgba.red', 181, 100) > 0.5)
    node['show_tiles'].setValue(False)
    check('previous output also survives reopen', node['output_mode'].value() == 'Source (Bypass)')
    report['last_plan'] = plans[-1]


try:
    run()
    report['ok'] = True
except Exception:  # noqa: BLE001 - report failures to the external pytest runner.
    report['error'] = traceback.format_exc()
Path(os.environ['KYVEN_NUKE_TILE_PREVIEW_REPORT']).write_text(json.dumps(report, indent=2))
