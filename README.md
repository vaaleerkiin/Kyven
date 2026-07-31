# Kyven Tools

Local, modular AI tools for node-based compositing.

Kyven Tools is an expandable compositing toolkit, not a single masking plug-in. The first working
tools are `Kyven Segment`, `Kyven Refine`, and `Kyven Inpaint` for Foundry Nuke. SAM 2, ViTMatte,
and LaMa run in a separate
authenticated local process, keeping PyTorch and CUDA outside Nuke and writing reusable results to
per-node caches. Depth, inpainting, paint-oriented utilities, Fusion, and DaVinci Resolve support
are planned around the same host-independent server.

The product name is **Kyven Tools**. The repository, Python package, CLI command, server, and node
prefix keep the short technical name `kyven` / `Kyven` for compatibility.

## Available today

- SAM 2.1 Tiny, Small, Base+, and Large model selection;
- ViTMatte Small refinement from any connected mask or artist trimap;
- LaMa object removal from Source + Mask, including CPU operation and automatic mask-bounds ROI;
- automatic trimap generation with foreground erosion and background dilation;
- exact cached trimap outputs: matte, Source + Trimap Alpha, and trimap cutout;
- Live current-frame processing in Segment and Refine;
- positive and negative Viewer points;
- optional Processing ROI that crops inference and restores a full-frame matte;
- dependency-free enclosed-hole filling with a configurable maximum area;
- current-frame and independent frame-range segmentation;
- SAM 2 video propagation forward, backward, or in both directions, with animated ROI and progress;
- Matte, Source + Alpha, Cutout, and Source (Bypass) outputs;
- Source + Alpha as the default output for new Segment and Refine nodes;
- per-node cache paths, native Read creation, and cache cleanup;
- asynchronous jobs, cancellation, and automatic local-server startup;
- API version checks, bearer-token authentication, and loopback-only networking.

## Architecture

```text
Nuke Group node
      |
      | authenticated HTTP on 127.0.0.1
      v
Kyven Server -> job queue -> provider registry -> task-specific local models
      |
      v
Atomic per-node result cache
```

The Nuke process never imports PyTorch or model implementations. Providers are loaded on demand,
inactive models can be unloaded, and older local server revisions are asked to unload before a new
revision starts.

## Planned tools

- **Kyven Depth:** interactive single-frame depth, temporally consistent video depth, independent
  frame fallback, scene-cut handling, temporal stabilization, multiple output views, and a
  low-memory/CPU path.
- **Kyven Inpaint quality mode:** PowerPaint is the planned optional guided-fill provider after its
  large multi-file checkpoint and runtime receive a secure installer path.
- **Kyven Utils:** future focused utilities such as paint/cleanup assistance and other compositing
  image operations. Each major operation remains an independent node.

The detailed, non-binding implementation plan and model-license constraints are recorded in
[Vision and roadmap](docs/ROADMAP.md).

## Portable installation on Windows

Clone or extract the repository into its final folder, then double-click `install.cmd`. A console
window opens and asks which model or models to install. The equivalent PowerShell command is:

```powershell
powershell -ExecutionPolicy Bypass -File .\install.ps1
```

The script installs everything beside the repository and does not modify Nuke or system settings:

- `.venv` contains Python packages, PyTorch, and SAM 2;
- `models` contains verified checkpoints;
- `.runtime` contains the pip cache, server files, and generated Nuke cache.

The console asks which model or models to install. Enter one or several numbers such as `1,2,5`.
Press Enter without typing anything to install SAM 2.1 Small and ViTMatte Small, the recommended
complete setup for an 8 GB GPU.
For unattended installation, models can also be selected explicitly:

```powershell
.\install.ps1 -Model sam2.1-tiny
.\install.ps1 -Model sam2.1-tiny,sam2.1-small
.\install.ps1 -Model sam2.1-base-plus
.\install.ps1 -Model sam2.1-large
.\install.ps1 -Model vitmatte-small-composition-1k
.\install.ps1 -Model lama-2025jan-onnx
.\install.ps1 -Model none
```

The installer is safe to run again after `git pull`. It reuses `.venv` and existing verified model
files. It does not require administrator access and does not add anything to the system `PATH`.
Choose the repository's final location before installation. If it is moved later, run
`install.ps1` again in the new location so the private Python environment is rebuilt correctly.
Restart Nuke after an update so it reloads the adapter and starts the required API revision.

After installation, manually add the path printed by the script to the existing Nuke `init.py`:

```python
import nuke

nuke.pluginAddPath("D:/Kyven/hosts/nuke")
```

The script intentionally never edits `init.py`. Restart Nuke and choose `Kyven > Segment`,
`Kyven > Refine`, or `Kyven > Inpaint`. Existing nodes can be migrated with `Upgrade Selected Segment Node` or
`Upgrade Selected Refine Node` in the same menu.

## Typical Nuke workflow

1. Connect a Source to `KyvenSegment`.
2. Place a positive point on the object and optional negative points on unwanted areas.
3. Optionally enable and draw a Processing ROI around the useful area.
4. Run `Process Current Frame`, an independent range, or SAM 2 video tracking.
5. Choose the required output mode or create a native Read from the cached matte.

For refinement, connect the original image to Refine input 0 and a coarse mask (for example the
Segment output) to input 1. Keep `Generate Trimap from Mask` enabled, then use Live, process one
frame, or render a range. Disable the option only when input 1 is already a black/gray/white trimap.
`Source + Refined Alpha` is the default; choose a trimap output to inspect the exact 0/128/255 image
used by ViTMatte.

For cleanup, connect Source to Inpaint input 0 and a matte to input 1. `Auto` crops to the mask
bounds plus Context Padding, runs LaMa only on that crop, then reconstructs the original format.
Pixels outside the grown/feathered merge mask remain identical to Source.

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
Kyven Tools does not commit or redistribute model weights. Project code is Apache-2.0; provider and
dependency licenses are recorded in [Third-party notices](THIRD_PARTY_NOTICES.md).

## Documentation

- [Nuke workflow and UI](docs/NUKE.md)
- [Portable installation and updates](docs/INSTALLATION.md)
- [Refine and trimap workflow](docs/REFINE.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
- [Segment engine and CLI](docs/SEGMENT.md)
- [Server API](docs/SERVER.md)
- [Model catalog](docs/MODELS.md)
- [Development benchmarks](docs/BENCHMARKS.md)
- [Kyven Tools vision and roadmap](docs/ROADMAP.md)

## Project status

Active pre-alpha implementation. Kyven Tools currently ships working Segment and Refine/ViTMatte
vertical slices in Nuke. Depth and Inpaint are planned, not implemented. Fusion and DaVinci Resolve
integrations are also not implemented yet.
