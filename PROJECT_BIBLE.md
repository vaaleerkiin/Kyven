# Kyven project scope

Kyven is a local, Nuke-focused toolkit with three core product areas:

1. Segment — promptable image segmentation and SAM 2 video mask propagation.
2. Refine — trimap construction and ViTMatte alpha refinement.
3. Inpaint — source-and-mask object removal with LaMa.

An owner-requested optional DLSS reconstruction node is authorized by
[ADR 0002](docs/adr/0002-optional-dlss-worker.md) and its installation update
[ADR 0003](docs/adr/0003-managed-dlss-runtime.md). It uses a Kyven-managed Windows/RTX worker, external
motion and alpha/control guides. It cannot replace or weaken the core CPU/4 GB baseline.

## Architecture

The owner-requested standalone WAFT Cattery plugin is authorized by
[ADR 0008](docs/adr/0008-standalone-waft.md). It keeps its embedded models and
native Nuke Inference graph under `plugins/Cattery/WAFT/`, independent of the
Kyven server. Its complete folder remains portable back to Cattery.

Nuke is a thin host adapter. It exports frames, submits authenticated jobs to the local server,
polls progress, and reads durable cached results. The host-independent backend owns model loading,
validation, inference, cancellation, and atomic output writes. Backend modules must not import Nuke.

Model Manager can read the trusted local catalog before the backend is installed.
It launches the existing portable base installer and optional component installers
as external processes; installation and verification must never import Torch into Nuke.
If no compatible interpreter is available, the base installer downloads a pinned,
SHA-256-verified standalone Python into `.runtime/python`, without system registration.

Models are selected through a trusted catalog with pinned sources, expected sizes, SHA-256 hashes,
license metadata, and hardware guidance. Optional runtimes load lazily, and only the active model
should occupy accelerator memory.

## Product rules

- Keep Segment, Refine, and Inpaint independently usable.
- Preserve node UUIDs and cached outputs during node upgrades.
- Keep source media local and bind the server only to localhost with bearer authentication.
- Report progress and support cooperative cancellation for long operations.
- Keep output files usable after the server or model is stopped.
- Do not add unrelated tools or speculative product documentation without explicit approval.

Detailed behavior lives in `docs/SEGMENT.md`, `docs/REFINE.md`, `docs/INPAINT.md`,
`docs/DLSS.md`, `docs/NUKE.md`, and `docs/SERVER.md`.

Owner-requested optional Generative Fill is authorized by
[ADR 0009](docs/adr/0009-generative-fill.md). Its native Nuke node uses a separate
local FLUX.2 Klein 4B Q4/LanPaint worker with shared Qwen3-4B prompt improvement.
It retains the existing lower-memory tools and has no ComfyUI dependency. See
[Generative Fill](docs/GENERATIVE_FILL.md).

The owner also authorized an experimental MOSS SFT INT8 + Edit backend on
2026-11-25. It has a separate pinned runtime and quantized model bundle, keeps
Klein available, and constrains known latents throughout LanPaint sampling.
Its Edit adapter's commercial licensing remains unconfirmed. See [MOSS](docs/MOSS_FILL.md).
