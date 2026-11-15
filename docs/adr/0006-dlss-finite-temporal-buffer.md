# ADR 0006: Finite external-vector temporal residual buffer

- Status: Accepted for first implementation; real-motion shot qualification remains pending
- Date: 2026-11-15

## Decision

The owner requested work on vectors and a frame buffer to reduce flicker, including research into
native tools. Preserve the earlier constraint that vector generation is external. Document native
Nuke options without embedding them or adding undocumented NR parameters.

Replace recursive single-output residual history with 1–4 raw historical corrections, default 3.
Compose backward displacement fields to align samples. Use optional previous-forward consistency,
photometric/mask/depth rejection, residual clamping and decreasing age weights. Store original
source data for validation, never blend source beauty frames and never store the accumulated result
as another raw history sample. Keep one causal shot/session buffer with no future-frame lookahead.

Bound retained CPU history by both count and bytes (512 MiB default). Reduce effective count when
needed; fail before native inference if a single sample cannot fit. Clear CPU/GPU history on the
existing discontinuities, cancellation/errors and explicit unload. Reprojection is cancelable between
row strips. No new heavy dependency or GPU allocation is introduced by the postprocessor; core
CPU/4 GB policies remain unchanged and optional DLSS has no CPU inference guarantee.

Expose buffer controls and explicit Nuke channel mapping in the existing interface. New settings
are artist settings and may trigger Live; upstream updates never do. ADR 0005's strict frame/settings
scheduling remains authoritative. Use API 36 and cache algorithm 5 to invalidate older request/output
assumptions. Synthetic tests and opt-in GPU qualification must distinguish controlled fixtures from
real shot quality. See [research, limits and setup](../DLSS_TEMPORAL.md).
