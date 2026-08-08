# Kyven Segment

`Kyven Segment` creates a binary base matte from positive points, negative points, and an optional
model prompt box. Host adapters may also supply a separate Processing ROI that crops inference and
is never sent to the model as a prompt.

## Current pipeline

```text
Nuke / CLI adapter
          |
          v
SegmentRequest
          |
          v
SegmentService
          |
          v
SegmentationProvider (SAM 2.1 initially)
          |
          v
Atomic grayscale PNG matte
```

The provider registry stores factories and constructs a provider only when selected. Merely
starting a host or importing Kyven does not import PyTorch, load SAM 2, or consume VRAM.

## Development

Create a Python 3.10 or newer virtual environment and install the core package:

```text
python -m pip install -e ".[dev]"
python -m unittest discover -s tests -v
```

SAM 2 is an optional provider runtime. On a compatible Windows NVIDIA system, the tested CUDA
12.8 runtime can be installed with `requirements/runtime-cu128.txt`. Kyven downloads model
weights separately from its trusted catalog and verifies both size and SHA-256.

## CLI

Example with one positive point, one negative point, and the SAM 2.1 Small checkpoint:

```text
kyven segment \
  --input frame.png \
  --output matte.png \
  --model sam2.1-small \
  --models-dir models \
  --point 640,360,positive \
  --point 100,100,negative \
  --profile balanced
```

Box prompts use `--box X0,Y0,X1,Y1`. Device selection is `auto`, `cuda`, or `cpu`.

The local server API additionally accepts `roi: {x0, y0, x1, y1}`. `SegmentService` translates
prompts into ROI coordinates, runs the provider on a temporary crop, and expands the mask back to
the source dimensions. ROI participates in deterministic cache identity.

After inference, optional enclosed-hole filling labels black connected components and fills only
those that do not touch the frame border and fit under `max_hole_area`. This changes neither the
outer silhouette nor model inference. The Nuke adapter stores an additional raw SAM matte; changing
the post-process checkbox or full-width area slider rebuilds the displayed matte immediately on CPU
without running SAM again.

On success the CLI prints structured JSON with the output path, selected-mask score,
deterministic cache key, device, and provider metadata. On failure it prints a structured Kyven
error suitable for display by a host adapter.

## Implemented host workflow

The Nuke adapter supports multiple Viewer points, static or animated Processing ROI, independent
ranges, SAM 2 video tracking with progress/ETA/cancellation, four output modes, native Read creation,
and cache cleanup. Video points are sampled on every saved key/correction frame while the animated
ROI is sampled for each frame and reconstructed into full-resolution coordinates. All corrections
condition one SAM 2 tracking state before propagation. Remaining Segment work includes resumable
per-frame metadata, explicit out-of-memory retry
profiles, and broader host testing.

New Segment nodes default to `Source + Alpha`. Updating an older Group through
`Kyven > Upgrade Selected Segment Node` preserves its prompts, UUID, input, and cached matte.
