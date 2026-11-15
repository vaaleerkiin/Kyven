# ADR 0004: DLSS HDR conditioning and qualified RGBA control guides

- Status: Accepted; qualified on the development RTX 4070 Laptop
- Date: 2026-11-15

## Context

The owner requested internal HDR rolloff with restoration to scene-linear RGB and working alpha
guides. Values above 1 were previously passed directly into NR. Initial mask experiments used a
scalar R32F GPU texture and an uninitialized backbuffer, producing invalid output.

## Decision

Condition RGB in the backend, preserving the original float32 Source as the output base. Use a
fixed per-pixel peak shoulder above a configurable knee, uniform RGB gain and a temporary lift for
negative RGB. Quantize the working image once to actual half-float transport precision. Decode the
neural residual with saved inverse gains and restore it onto Source. Do not derive exposure from
frame statistics or add implicit Premult/Unpremult nodes. Transform code uses bounded row strips;
only one inverse-gain plane is retained at processing resolution. HDR rolloff defaults on.

Kyven worker 2.0 expands the scalar artist guide into equal RGBA32F controls. White enables NR;
black preserves the initialized current-frame backbuffer. Set its explicit subrect, initialize it
before each guided evaluation and keep deterministic DXGI RGB channel order. Require the worker's
`control-mask=rgba-v1` version capability before sending guides. Model Manager reports stale owned
workers as updates and installs the bundled replacement through its existing operation queue.

Expose Full Frame, Source Alpha and Mask Input without changing existing full-frame selections.
Always preserve output alpha. Optional final protection zeros corrections at black full-resolution
mask pixels; soft guide weights are not applied a second time. Guidance remains active when final
protection is disabled. No optical-flow generator is added; temporal motion remains external.

API 35 accepts rolloff settings; cache algorithm revision 4 and request settings invalidate older
outputs/history. Existing nodes upgrade in place with their UUID, inputs and cache paths intact.
The owner-requested Processing Size menu defaults to 1; reduced scales remain available for memory.

## Evidence and limits

`scripts/qualify_dlss_guides.py` performs opt-in GPU A/B qualification and writes a JSON report.
Development tests passed black passthrough, white/full equivalence, split/soft neural guide response,
HDR up to 1e6, negative RGB, exact alpha preservation and three external-vector temporal frames,
with final RGB protection disabled. Ordinary tests cover identity HDR round trips, half-float
precision anchoring, cache/history invalidation, legacy worker rejection and host Live scheduling.

This establishes transport/image behavior on one runtime/GPU/driver, not quality for every shot or
4 GB VRAM compatibility. RGBA guide allocation uses 16 bytes per processing pixel. DLSS remains an
optional Windows/RTX provider; core CPU/4 GB policies remain unchanged. Nonfinite restored outputs
fail before publication, and failures/cancellation unload native state.
