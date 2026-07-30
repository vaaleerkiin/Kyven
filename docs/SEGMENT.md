# Kyven Segment

`Kyven Segment` creates a binary base matte from positive points, negative points, and an
optional bounding box. It is the first host-independent Kyven vertical slice.

## Current pipeline

```text
Nuke / Fusion / CLI adapter
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

On success the CLI prints structured JSON with the output path, selected-mask score,
deterministic cache key, device, and provider metadata. On failure it prints a structured Kyven
error suitable for display by a host adapter.

## Next integration slice

The asynchronous server and initial Nuke adapter now exist. The next Segment work is production
host testing, multi-point viewer interaction, frame-range processing, resumable cache metadata,
and explicit out-of-memory retry profiles.
