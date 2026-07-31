# Kyven Refine

Refine converts a coarse mask into a soft alpha matte using ViTMatte while keeping model inference
outside the compositing host.

Kyven offers the official ViTMatte Small and Base backbone sizes. There is no official Medium
variant. Small Composition-1k is the recommended 4–8 GB default; Base Distinctions-646 has 96.7 M
parameters and is intended for 8 GB+ systems with tiling when required. Install either checkpoint
from **Kyven > Model Manager...** without rerunning the portable installer.

## Nuke graph

```text
Source --------------------> Kyven Refine (input 0)
Source -> Kyven Segment ---> Kyven Refine (input 1)
```

Input 0 is the original RGB image. Input 1 is normally any mask with useful alpha: Segment, Roto,
Keyer, Paint, or a corrected combination of those nodes.
`Mask Input Channel` defaults to Alpha; choose Red for a grayscale RGB mask or artist trimap.

## Trimap option

`Generate Trimap from Mask` is enabled by default and is part of Refine, not a separate node.
Foreground erosion produces definite white foreground, background dilation produces the gray
unknown band, and pixels outside the band remain definite black background. Disable the option only
when input 1 is already an artist-created black/gray/white trimap.

The `Output` selector includes the exact trimap written for ViTMatte:

- `Trimap` displays the black / gray / white guidance as RGBA;
- `Source + Trimap Alpha` keeps Source RGB and places the trimap in alpha;
- `Trimap Cutout` premultiplies Source RGB by the trimap.

`Source + Refined Alpha` is the default output. When Processing ROI is enabled, areas outside the
actual model crop are black in the trimap preview because they were not sent to ViTMatte.

## Processing

- `Live Current Frame` follows the timeline and regenerates ViTMatte after model/ROI changes.
- `Process Current Frame` updates one cached matte.
- `Process Frame Range` exports and refines every selected frame sequentially.
- `Create Matte Read` creates a standard Nuke Read for the cached frame or sequence.
- `Cancel` stops queued work cooperatively.

Manual current-frame processing and frame-range processing open a native Nuke progress window with
export stage, frame/tile progress, percentage, ETA, and Cancel. Live refreshes remain unobtrusive and
report through the node Status field instead of opening a new window after every Viewer edit.

Range input export uses two batched Nuke renders instead of two render calls per frame. Source is
stored as fast lossless TIFF and the mask as PNG; ViTMatte jobs remain sequential after export.
Inverted ROI corners are normalized, ROI is clamped to the input, and an empty/outside ROI safely
falls back to the full input format.

The cached files for a processed frame are `refine_source`, `refine_mask`, `refined_matte`, and
`trimap`. Range processing writes matching `%04d` sequences and connects internal Reads to both the
refined matte and trimap. `Create Matte Read` creates a Read for the refined matte; use
the node's trimap output modes to inspect the cached trimap in context.

Trimap preview is independent of ViTMatte. Connecting or changing Input 1, moving to another frame,
or adjusting `Foreground Erosion`, `Background Dilation`, trimap mode, or ROI immediately rebuilds
the black/gray/white preview on CPU. The two radius controls are full-width sliders. This preview
does not load or run ViTMatte; press Process only when the guidance looks correct.

Processing ROI crops both image and mask before inference, then pastes the refined alpha into the
full-size coarse mask. Low Memory uses 512 px tiles, Balanced uses 1024 px tiles, and Quality uses a
full-frame pass unless a custom tile size is set. Overlap blending reduces tile seams.

## CLI

```text
kyven refine --input source.png --mask coarse.png --output alpha.png \
  --model vitmatte-small-composition-1k --profile low_memory
```

Add `--manual-trimap` when the mask path is already a trimap.

## Updating an existing node

After updating the repository and restarting Nuke, select an older Refine Group and choose
`Kyven > Upgrade Selected Refine Node`. The command adds the current seven output modes without
discarding the existing refined matte. The trimap outputs become available immediately from the
CPU-only preview; process a frame or range only when you want a new ViTMatte result.
