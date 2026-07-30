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

SAM 2 is an optional provider runtime. Install PyTorch for the required CUDA or CPU platform,
then install the official SAM 2 package according to its upstream instructions. Kyven does not
download or redistribute the checkpoint yet.

## CLI

Example with one positive point, one negative point, and the SAM 2.1 Small checkpoint:

```text
kyven segment \
  --input frame.png \
  --output matte.png \
  --checkpoint models/sam2.1_hiera_small.pt \
  --point 640,360,positive \
  --point 100,100,negative \
  --profile balanced
```

Box prompts use `--box X0,Y0,X1,Y1`. Device selection is `auto`, `cuda`, or `cpu`.

On success the CLI prints structured JSON with the output path, selected-mask score,
deterministic cache key, device, and provider metadata. On failure it prints a structured Kyven
error suitable for display by a host adapter.

## Next integration slice

The next step is an asynchronous job API around `SegmentService`, followed by a thin Nuke node
that collects prompts, submits a job, remains responsive, and reads the cached matte result.

