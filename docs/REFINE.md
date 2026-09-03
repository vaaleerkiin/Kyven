# Kyven Refine

[Documentation home](README.md) · [Nuke workflow](NUKE.md) · [Segment](SEGMENT.md) ·
[Models](MODELS.md) · [Troubleshooting](TROUBLESHOOTING.md)

Kyven Refine supports ViTMatte and the memory-efficient MEMatte ViT-S AIM-500 backend. Both consume
the same source RGB and binary mask/trimap inputs and produce the same alpha/output branches.

Refine converts a coarse mask into a soft alpha matte using ViTMatte while keeping model inference
outside the compositing host.

## On this page

- [Nuke graph](#nuke-graph)
- [Trimap option](#trimap-option)
- [Processing and performance](#processing)
- [Cache](#cache)
- [CLI](#cli)
- [Updating an existing node](#updating-an-existing-node)

Kyven offers the official ViTMatte Small and Base backbone sizes. There is no official Medium
variant. Small Composition-1k is the recommended 4–8 GB default; Base Distinctions-646 has 96.7 M
parameters and is intended for 8 GB+ systems with tiling when required. Install either checkpoint
from **Kyven > Model Manager...** without rerunning the portable installer.

## Nuke graph

```text
Source --------------------> Kyven Refine (input 1, left)
Source -> Kyven Segment ---> Kyven Refine (input 0, right)
```

Input 1 is the original RGB image. Input 0 is normally any mask with useful alpha: Segment, Keyer,
Paint, or a corrected combination of those nodes. The connector labels are the safest guide: Source
is on the left and Mask is on the right.
`Mask Input Channel` defaults to Alpha; choose Red for a grayscale RGB mask or artist trimap.

## Trimap option

`Generate Trimap from Mask` is enabled by default and is part of Refine, not a separate node.
Foreground erosion produces definite white foreground, background dilation produces the gray
unknown band, and pixels outside the band remain definite black background. Disable the option only
when the Mask input is already an artist-created black/gray/white trimap.

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
The selected provider loads once and remains resident across the range. For Full Frame, Kyven also
avoids re-encoding the already exported Source into another temporary PNG; Auto/Manual crops still
write a temporary cropped input as required.
Inverted ROI corners are normalized, ROI is clamped to the input, and an empty/outside ROI safely
falls back to the full input format.

The artist-facing output is `<Refine Name>_<Version>.%04d.png`. Source, mask, trimap, preview,
manifest, and other implementation files are kept under `_kyven_work/`. Range processing connects
internal Reads to both the public refined sequence and private trimap sequence. **Create Matte Read**
creates a normal Read for the public output; use the node's trimap modes to inspect guidance.

Trimap preview is independent of ViTMatte. Connecting or changing the Mask input, moving to another
frame, or adjusting `Foreground Erosion`, `Background Dilation`, trimap mode, or ROI immediately
rebuilds the black/gray/white preview on CPU. The two radius controls are full-width sliders. This
preview does not load or run ViTMatte; press Process only when the guidance looks correct.

Processing ROI crops both image and mask before inference, then pastes the refined alpha into the
full-size coarse mask. Low Memory uses 512 px tiles, Balanced uses 1024 px tiles, and Quality uses a
full-frame pass unless a custom tile size is set. Overlap blending reduces tile seams.

## Cache

New Refine nodes use three artist fields: **Cache Folder**, **Refine Name**, and **Version**.
The version root contains only the public sequence
`<Refine Name>_<Version>.%04d.png` and `_kyven_work/`. Source exports are stored under
`_kyven_work/source/`; masks, trimaps, previews, manifests, and other implementation files remain
inside `_kyven_work/`. Current Frame and Frame Range share the same public sequence. Changing any
of the three fields reconnects an existing matching sequence.

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
