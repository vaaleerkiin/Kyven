# Kyven Segment for Nuke

The initial adapter is under `hosts/nuke`. It creates a `KyvenSegment` Group node with:

- Source input with selectable Matte, Source + Alpha, Cutout, and Source (Bypass) outputs;
- selectable SAM 2.1 Tiny, Small, Base+, or Large model;
- Low Memory, Balanced, and Quality execution profiles;
- dynamically addable/removable positive and negative point controls;
- independent show/use toggles that hide disabled Viewer handles;
- an optional input-sized Processing ROI and input-centered initial points;
- source-inherited output format and canvas, including before the first matte is generated;
- Process Current Frame, Process Frame Range, and Cancel actions;
- in-process source export followed by asynchronous inference;
- asynchronous server submission and result polling;
- model installation and VRAM warnings from the server catalog.

The adapter imports no PyTorch or SAM code. It starts Kyven Server as a hidden external process
when required.

## Development installation

Add the following to the user's existing `.nuke/init.py`, using the actual repository path:

```python
import nuke
nuke.pluginAddPath("D:/Kyven/hosts/nuke")
```

Restart Nuke and choose `Kyven > Segment` from the Nodes menu. Set `KYVEN_ROOT` if the checkout
is not at `D:/Kyven`.

## Frame ranges

`Process Frame Range` exports the selected range and generates a lossless PNG matte sequence.
The same static points and Processing ROI are applied to every frame. The
resulting sequence is connected inside the Group as `matte.%04d.png`.

## Processing ROI

`Enable Processing ROI` treats the Viewer rectangle as a crop, not as a SAM box prompt. Kyven
crops the source before inference, translates point coordinates into the crop, and expands the
returned mask back onto a full-resolution black canvas. For video tracking, every temporary JPEG
is cropped consistently and every final PNG is restored to the original frame dimensions.

A positive point must be inside the ROI. Negative points outside it are ignored because those
pixels never reach the model. SAM 2 internally resizes inputs to its fixed encoder resolution, so
ROI mainly isolates the search area and gives the selected region more effective detail; it does
not guarantee a proportional reduction in GPU time or VRAM.

## SAM 2 video tracking

The `SAM 2 VIDEO TRACKING` section uses temporal memory instead of segmenting every frame
independently:

1. Move to the frame where the object is easiest to identify.
2. Place points, optionally limit the Processing ROI, and click `Set Key Frame to Current`.
3. Set `Range First` and `Range Last`.
4. Choose `Propagate Forward`, `Propagate Backward`, or `Propagate Both Directions`.

Nuke exports a temporary high-quality JPEG sequence for the video predictor. Final masks remain
lossless PNG files. Frames and inference state are offloaded to system RAM so SAM 2.1 Small is
practical on an 8 GB GPU. This version starts a fresh tracking session for each run; correction
on additional key frames is follow-up work.

## Output modes

- `Matte` puts the mask in RGB and alpha;
- `Source + Alpha` preserves the original RGB and replaces alpha with the mask;
- `Cutout` premultiplies the source by the generated alpha;
- `Source (Bypass)` returns the unchanged input.

These modes are native Nuke operations and do not rerun SAM. For an existing Kyven Segment node,
select it and choose `Kyven > Upgrade Selected Segment Node`; the cached matte is preserved.

## Cache controls

Each Segment node displays its own cache folder. `Create Read from Current Matte` creates a regular
Nuke Read node pointing at the current single-frame or sequence matte. `Delete This Node Cache`
asks for confirmation, disconnects the internal cached matte, and removes only that Segment node's
cache folder. `Delete All Kyven Cache` clears every Segment cache under `.runtime/nuke_cache` but
keeps downloaded models, the virtual environment, and server configuration.
