# Kyven project scope

Kyven is a local, Nuke-focused toolkit with exactly three product areas:

1. Segment — promptable image segmentation and SAM 2 video mask propagation.
2. Refine — trimap construction and ViTMatte alpha refinement.
3. Inpaint — source-and-mask object removal with LaMa.

## Architecture

Nuke is a thin host adapter. It exports frames, submits authenticated jobs to the local server,
polls progress, and reads durable cached results. The host-independent backend owns model loading,
validation, inference, cancellation, and atomic output writes. Backend modules must not import Nuke.

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
`docs/NUKE.md`, and `docs/SERVER.md`.
