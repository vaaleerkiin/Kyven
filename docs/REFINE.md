# Kyven Refine

[Documentation home](README.md) · [Nuke workflow](NUKE.md) · [Segment](SEGMENT.md) ·
[Models](MODELS.md) · [Troubleshooting](TROUBLESHOOTING.md)

Kyven Refine supports ViTMatte and the memory-efficient MEMatte ViT-S AIM-500 backend. Both consume
the same source RGB and binary mask/trimap inputs and produce the same alpha/output branches.

Refine converts a coarse mask into a soft alpha matte while keeping model inference outside the compositing host.

## On this page

- [Quick start](#quick-start)
- [Nuke graph](#nuke-graph)
- [Trimap option](#trimap-option)
- [Processing and performance](#processing)
- [Optional alpha integration](#optional-alpha-integration)
- [Cache](#cache)
- [CLI](#cli)
- [Updating an existing node](#updating-an-existing-node)

Kyven offers the official ViTMatte Small and Base backbone sizes. There is no official Medium
variant. Small Composition-1k is the recommended 4–8 GB default; Base Distinctions-646 has 96.7 M
parameters and is intended for 8 GB+ systems with tiling when required. Install either checkpoint
from **Kyven > Model Manager...** without rerunning the portable installer.

## Quick start

1. Connect Source and a coarse Mask with the matte in alpha, as shown below.
2. Choose an installed model and set Cache Folder, Refine Name and Version.
3. Inspect **Trimap** output and adjust Foreground Erosion and Background Dilation; disable trimap generation if your input is already a trimap.
4. Press **Process Current Frame** and inspect **Source + Refined Alpha**. Set the frame range and process it when the result is ready.

Hover over controls for English help. See [shared processing and cache](NUKE.md#input-preparation-and-processing) for refreshing inputs after upstream edits.

## Nuke graph

```text
Source --------------------> Kyven Refine (input 1, left)
Source -> Kyven Segment ---> Kyven Refine (input 0, right)
```

Input 1 is the original RGB image. Input 0 is normally any mask with useful alpha: Segment, Keyer,
Paint, or a corrected combination of those nodes. The connector labels are the safest guide: Source
is on the left and Mask is on the right.
The Mask input always reads **alpha**. There is no Alpha/Red selector. For a grayscale RGB
image, copy its matte into alpha upstream with a Shuffle or Copy node. Existing nodes also use
alpha after upgrade; their old channel selector is hidden and disconnected.

## Trimap option

`Generate Trimap from Mask` is enabled by default and is part of Refine, not a separate node.
Generation uses the simple Cattery ViTMatte recipe: erode and dilate the original Mask alpha,
average those two masks equally, then map values below 0.01 to black, above 0.99 to white,
and everything else to gray. The mask is not binarized before morphology. Foreground Erosion and
Background Dilation control the two radii. Disable the option when the Mask input is already
an artist-created black/gray/white trimap; the same cutoffs apply without erosion/dilation.
With generation disabled, alpha is quantized to three guidance states; the gray unknown region
is deliberately preserved. In Segment this input is called **Edge Guide**.

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

Range processing prepares reusable lossless TIFF snapshots for Source and Mask, then runs the selected provider sequentially. Use **Refresh Frames** after upstream changes; see [input preparation](NUKE.md#input-preparation-and-processing). The selected provider remains resident across the range.

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

## Optional alpha integration

For local repairs to an existing alpha, enable **Blend ROI into Input Alpha** in
**Alpha Integration / Optional**. This is off by default and applies in **Manual** crop mode.
The entire integration section is hidden in Auto and Full; returning to Manual restores its settings.
It combines the already cached refined matte with the live **Mask** input alpha on the node's
output. It does not change model inputs, crop size, trimap generation, or cached files.

**ROI Feather (px)** controls the smooth transition inward from the Manual ROI boundary.
Start with 64 px and increase it if the boundary is still visible. The center keeps the refined
matte; outside the ROI, the original incoming float alpha is preserved. The blend changes the
contribution of each matte without blurring hairs. A value of 0 gives a hard paste. Borders that
touch the source image edge are not faded, and the feather is limited for very narrow ROIs.

Toggling integration or adjusting its feather updates immediately, including with the server
stopped; it does not start model processing, even with Live enabled. The three refined output modes
use the integrated alpha. Trimap modes are independent. Use a normal Nuke **Write** connected to
the Group to save the integrated result; **Create Matte Read** opens the cached model result.

After updating, restart Nuke or use **Kyven > Upgrade Selected Refine Node** to add the optional
controls to an existing node. Its cached matte and current processing settings are preserved.

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

After restoring Cattery-style trimap preparation, processing automatically refreshes old input
snapshots and invalidates results from the previous positive-alpha recipe. Reprocess affected frames.
