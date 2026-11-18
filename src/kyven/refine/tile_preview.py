"""CPU-only drawing of the exact refinement tile plan as a Viewer overlay."""

from __future__ import annotations

import tempfile
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from kyven.atomic_io import replace_with_retry
from kyven.cancellation import CancellationToken
from kyven.hashing import cached_file_sha256
from kyven.refine.models import RefineRequest
from kyven.refine.regions import refine_region, with_context
from kyven.refine.tiling import plan_tiles
from kyven.refine.trimap import generate_trimap, normalize_trimap
from kyven.segment.roi import ResolvedRegion


@dataclass(frozen=True, slots=True)
class PreparedTrimap:
    pixels: np.ndarray
    bounds: ResolvedRegion


class TrimapPreviewCache:
    """Bounded, thread-safe CPU guidance cache for repeated slider requests."""

    def __init__(self, max_bytes: int = 64 * 1024 * 1024) -> None:
        self.max_bytes = max_bytes
        self._items: OrderedDict[tuple, PreparedTrimap] = OrderedDict()
        self._bytes = 0
        self._lock = threading.RLock()

    def prepare(self, request: RefineRequest) -> tuple[PreparedTrimap, bool]:
        checksum = cached_file_sha256(request.mask)
        key = (checksum, request.generate_trimap,
               request.foreground_radius if request.generate_trimap else 0,
               request.background_radius if request.generate_trimap else 0)
        with self._lock:
            cached = self._items.get(key)
            if cached is not None:
                self._items.move_to_end(key)
                return cached, True
        with Image.open(request.mask) as image:
            mask = np.asarray(image.convert('L'))
        trimap = (generate_trimap(mask, request.foreground_radius, request.background_radius)
                  if request.generate_trimap else normalize_trimap(mask))
        if cached_file_sha256(request.mask) != checksum:
            raise OSError('Mask changed while preparing the tile preview.')
        trimap.setflags(write=False)
        prepared = PreparedTrimap(trimap, refine_region(trimap, 'auto', None, 0))
        if trimap.nbytes <= self.max_bytes:
            with self._lock:
                previous = self._items.pop(key, None)
                if previous is not None:
                    self._bytes -= previous.pixels.nbytes
                self._items[key] = prepared
                self._bytes += trimap.nbytes
                while self._bytes > self.max_bytes:
                    self._bytes -= self._items.popitem(last=False)[1].pixels.nbytes
        return prepared, False


_trimap_cache = TrimapPreviewCache()


def draw_tile_preview(request: RefineRequest, *, labels: bool = False) -> dict[str, Any]:
    """Write a transparent overlay; no source RGB or model weights are required."""

    request.validate()
    prepared, cache_hit = _trimap_cache.prepare(request)
    trimap = prepared.pixels
    region = (with_context(prepared.bounds, request.context_padding) if request.crop_mode == 'auto'
              else refine_region(trimap, request.crop_mode, request.roi, request.context_padding))
    crop = trimap[region.y0:region.y1, region.x0:region.x1]
    plan = plan_tiles(Image.fromarray(crop), request.tile_size, request.tile_overlap,
                      request.skip_known_tiles, CancellationToken())
    overlay = Image.new('RGBA', (region.source_width, region.source_height), (30, 30, 30, 100))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle((region.x0, region.y0, region.x1-1, region.y1-1), fill=(0, 0, 0, 0))
    if labels:
        try:
            font = ImageFont.load_default(size=max(12, min(24, min(overlay.size)//90)))
        except TypeError:  # Pillow 10.0 has no size argument.
            font = ImageFont.load_default()
    active = set(plan.boxes)
    boxes = []
    label_data = []
    outlines = []
    for index, box in enumerate(plan.grid_boxes, 1):
        x, y, x1, y1 = box
        full_box = (x+region.x0, y+region.y0, x1+region.x0-1, y1+region.y0-1)
        run = box in active
        color = (55, 235, 105, 255) if run else (140, 140, 140, 180)
        if run:
            draw.rectangle(full_box, fill=(40, 190, 80, 28))
        outlines.append((full_box, color, run))
        if labels:
            label = f"{'RUN' if run else 'SKIP'} {index}  {x1-x}x{y1-y}"
            label_data.append((full_box, label, color))
        boxes.append({'index': index, 'run': run, 'x0': x+region.x0, 'y0': y+region.y0,
                      'x1': x1+region.x0, 'y1': y1+region.y0})
    # Yellow is the actual unknown band; green boxes also include known context.
    pixels = np.asarray(overlay).copy()
    view = pixels[region.y0:region.y1, region.x0:region.x1]
    view[(crop > 1) & (crop < 254)] = (255, 195, 35, 200)
    overlay = Image.fromarray(pixels)
    draw = ImageDraw.Draw(overlay)
    # Keep complete square contours on top of both overlapping fills and the
    # unknown band, including a completely gray trimap.
    for box, color, run in sorted(outlines, key=lambda item: item[2]):
        draw.rectangle(box, outline=color, width=3 if run else 2)
    draw.rectangle((region.x0, region.y0, region.x1-1, region.y1-1),
                   outline=(55, 200, 255, 255), width=2)
    occupied = []
    for (x0, y0, x1, y1), label, color in label_data:
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font, stroke_width=1)
        width, height = right-left, bottom-top
        candidates = [(x0+5, y0+5), (x0+5, y1-height-5),
                      (x1-width-5, y0+5), (x1-width-5, y1-height-5)]
        # Edge-aligned full-size tiles can overlap almost completely. Move labels
        # rather than hiding the actual geometry or drawing unreadable overprints.
        for x, y in candidates:
            box = draw.textbbox((x, y), label, font=font, stroke_width=1)
            if box[0] < region.x0 or box[2] >= region.x1 or box[1] < region.y0:
                continue
            if any(box[0] < old[2]+3 and box[2]+3 > old[0] and
                   box[1] < old[3]+3 and box[3]+3 > old[1] for old in occupied):
                continue
            draw.text((x, y), label, font=font, fill=color,
                      stroke_width=1, stroke_fill=(0, 0, 0, 220))
            occupied.append(box)
            break
    request.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=request.output.parent, suffix='.png', delete=False) as f:
            temporary = Path(f.name)
        overlay.save(temporary, format='PNG', compress_level=1)
        replace_with_retry(temporary, request.output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {'output': str(request.output), **plan.metadata, 'tile_size': request.tile_size,
            'tile_overlap': request.tile_overlap, 'skip_known_tiles': request.skip_known_tiles,
            'processing_roi': region.metadata(), 'boxes': boxes,
            'trimap_bounds': prepared.bounds.metadata(), 'preview_labels': labels,
            'trimap_cache_hit': cache_hit}
