# Kyven

Local, modular AI masking for node-based compositing.

Kyven currently provides a working `Kyven Segment` node for Foundry Nuke. It runs SAM 2 in a
separate authenticated local process, keeps PyTorch and CUDA outside Nuke, and writes reusable
matte files to a per-node cache. Fusion and DaVinci Resolve adapters are planned around the same
host-independent server.

## Current features

- SAM 2.1 Tiny, Small, Base+, and Large model selection;
- positive and negative Viewer points;
- optional Processing ROI that crops inference and restores a full-frame matte;
- current-frame and independent frame-range segmentation;
- SAM 2 video propagation forward, backward, or in both directions;
- Matte, Source + Alpha, Cutout, and Source (Bypass) outputs;
- per-node cache paths, native Read creation, and cache cleanup;
- asynchronous jobs, cancellation, and automatic local-server startup;
- API version checks, bearer-token authentication, and loopback-only networking.

## Architecture

```text
Nuke Group node
      |
      | authenticated HTTP on 127.0.0.1
      v
Kyven Server -> job queue -> provider registry -> SAM 2
      |
      v
Atomic grayscale PNG matte cache
```

The Nuke process never imports PyTorch or SAM. Only one selected segmentation model is kept
resident, and older local server revisions are asked to unload before a new revision starts.

## Development setup on Windows

Python 3.10 or newer is required. The tested NVIDIA runtime uses CUDA 12.8 wheels:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements\runtime-cu128.txt
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Install a model from the trusted catalog:

```powershell
.\.venv\Scripts\kyven.exe models download sam2.1-small --models-dir models
```

Add the repository host folder to the existing Nuke `init.py`:

```python
import nuke

nuke.pluginAddPath("D:/Kyven/hosts/nuke")
```

Restart Nuke and choose `Kyven > Segment`. Existing nodes can be migrated with
`Kyven > Upgrade Selected Segment Node`.

## Typical Nuke workflow

1. Connect a Source to `KyvenSegment`.
2. Place a positive point on the object and optional negative points on unwanted areas.
3. Optionally enable and draw a Processing ROI around the useful area.
4. Run `Process Current Frame`, an independent range, or SAM 2 video tracking.
5. Choose the required output mode or create a native Read from the cached matte.

See [Nuke workflow](docs/NUKE.md) for every control and [Troubleshooting](docs/TROUBLESHOOTING.md)
for server, cache, CUDA, and logging help.

## Hardware guidance

The project target begins at 4 GB VRAM, with SAM 2.1 Tiny intended for the lowest tier. SAM 2.1
Small is the current default for an 8 GB GPU. Model selection remains manual: Kyven reports VRAM
guidance but does not prevent an artist from choosing a larger installed model.

Processing ROI isolates the search area and gives it more effective encoder detail. SAM 2 still
resizes inputs to a fixed internal resolution, so ROI does not guarantee a proportional reduction
in GPU time or VRAM.

## Privacy, security, and licensing

Inference is local by default. The server binds only to `127.0.0.1` and requires a random token.
Kyven does not commit or redistribute model weights. Project code is Apache-2.0; provider and
dependency licenses are recorded in [Third-party notices](THIRD_PARTY_NOTICES.md).

## Documentation

- [Nuke workflow and UI](docs/NUKE.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
- [Segment engine and CLI](docs/SEGMENT.md)
- [Server API](docs/SERVER.md)
- [Model catalog](docs/MODELS.md)
- [Development benchmarks](docs/BENCHMARKS.md)

## Project status

Active pre-alpha implementation. The Nuke Segment vertical slice works; Refine/ViTMatte, Fusion,
and DaVinci Resolve integrations are not implemented yet.
