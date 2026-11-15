# Kyven documentation

[Project overview](../README.md) · [Install](INSTALLATION.md) · [Nuke](NUKE.md) ·
[Troubleshooting](TROUBLESHOOTING.md)

This is the main index for Kyven documentation. Choose the path that matches what you are trying to
do; you do not need to read every guide in order.

## I am an artist using Nuke

1. [Install Kyven](INSTALLATION.md).
2. Read [the shared Nuke workflow](NUKE.md) for connections, processing modes, output modes, cache,
   progress, cancellation, and updating old nodes.
3. Open the guide for the task:
   - [Segment](SEGMENT.md) for point prompts, roto, correction frames, and SAM 2 tracking;
   - [Refine](REFINE.md) for trimaps, soft alpha, hair, and edge refinement;
   - [Inpaint](INPAINT.md) for object removal and clean-up;
   - [DLSS](DLSS.md) for optional RTX neural reconstruction and temporal setup.
4. Use [Troubleshooting](TROUBLESHOOTING.md) when the server, CUDA, a model, ROI, cache, or Read
   behaves unexpectedly.

## I am choosing hardware or models

- [Models and safety](MODELS.md) lists every supported model, expected hardware, download source,
  checksum policy, and licensing notes.
- [Development benchmarks](BENCHMARKS.md) records measured cold/warm behavior and ROI validation.

Start with SAM 2.1 Small for Segment, ViTMatte Small for Refine, and LaMa ONNX Fast for Inpaint.
Move to heavier models only when the shot demonstrates a quality limitation.

## I am integrating or developing Kyven

1. [Project scope](../PROJECT_BIBLE.md) defines the product and architecture boundaries.
2. [Server API](SERVER.md) documents authentication, jobs, payloads, and progress.
3. [Segment internals](SEGMENT.md) explains the provider pipeline, logits, confidence trimaps, ROI,
   and CLI.
4. [Instructions for coding agents](../CODEX.md) contains repository constraints and the definition
   of done.
5. [ADR 0001](adr/0001-selectable-model-catalog.md) records the selectable model-catalog decision.

`OWNER_NOTES_RU.md` is an owner-maintained product note, not an end-user manual.

## Guide map

| Guide | Audience | Read it for |
| --- | --- | --- |
| [Installation](INSTALLATION.md) | Artist / admin | First install, update, move, model installation |
| [Nuke workflow](NUKE.md) | Artist | Shared controls, ranges, outputs, cache, server behavior |
| [Segment](SEGMENT.md) | Roto artist / developer | Prompts, tracking, Seamless Key Frame, logits, CLI |
| [Refine](REFINE.md) | Roto artist | Trimaps, ViTMatte/MEMatte, tiling, outputs |
| [Inpaint](INPAINT.md) | Paint artist | Models, ROI, masks, blending, quality modes |
| [DLSS (optional RTX)](DLSS.md) | Artist / TD | Local runtime, external vectors, alpha guides, live and temporal behavior |
| [DLSS temporal research](DLSS_TEMPORAL.md) | Artist / TD | Native vector tools, finite history buffer, rejection and measured limits |
| [Models](MODELS.md) | Artist / TD | VRAM, model trade-offs, downloads, licenses |
| [Troubleshooting](TROUBLESHOOTING.md) | Everyone | Symptoms, checks, logs, recovery |
| [Server API](SERVER.md) | TD / developer | Endpoints, payload fields, queue and cache behavior |
| [Benchmarks](BENCHMARKS.md) | TD / developer | Measured performance and validation scope |

## Quick decision guide

| Goal | Start with | Then |
| --- | --- | --- |
| Create roto on one frame | Segment `Process Current Frame` | Add negative points or Refine |
| Propagate roto through a shot | Segment video tracking | Add correction frames; enable Seamless Key Frame if the primary key jumps |
| Improve a coarse edge | Refine + generated trimap | Inspect Trimap, then adjust erosion/dilation |
| Use SAM uncertainty in Refine | Segment `Trimap` output | Disable Refine `Generate Trimap from Mask` |
| Remove an object quickly | Inpaint + Auto ROI + LaMa ONNX | Switch to Big-LaMa Native when texture detail is insufficient |
| Reconstruct a shot on a supported RTX system | DLSS with Source | Add external Motion vectors for temporal stability |
| Hand cache to another artist | Open `<Cache Folder>/<Name>/<Version>/` | Read `<Name>_<Version>.%04d.png` for Segment, Refine or Inpaint; DLSS uses `.exr` |

## Documentation conventions

- **Source** means the original RGB input.
- **Mask** means the artist or model mask supplied to a node.
- **Processing ROI** is an inference optimization crop; it does not crop the final image format.
- **Public sequence** means files intentionally placed in the selected version root.
- **Work files** means private implementation data under `_kyven_work/`.
- Nuke labels are written in **bold**; paths and API values use `code`.

Every guide links back here near its title. If behavior and documentation disagree, treat the current
node Status and cache contents as evidence and report the mismatch with the Kyven commit hash.
