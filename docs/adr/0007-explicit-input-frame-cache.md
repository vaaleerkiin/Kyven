# ADR 0007: Explicit input frame preparation

- Status: Accepted (owner request)
- Date: 2026-11-15

## Current contract

Current Frame, Frame Range, tracking and Live prepare missing or invalid snapshots before inference.
Ready snapshots are reused. Cache Frames remains a separate preparation action; Refresh Frames
explicitly replaces inputs after upstream edits. The updates below record how this contract evolved.

## Decision

Separate input preparation from inference in all four Nuke adapters. Add Cache Frames and Refresh
Frames beside range processing, with Current Frame / Frame Range scope. Cache Frames fills missing
or invalid snapshots; Refresh Frames replaces snapshots after upstream changes. Neither starts
inference. Process, tracking and Live only consume verified snapshots and report missing frames.
Keep the existing backend output-cache behavior, model interfaces and asynchronous job queue.

## Consequences

Input export no longer repeats for every Process click. Inpaint model-mask settings invalidate
prepared inputs; other model settings and prompts do not invalidate source frames. Upstream graph
edits require explicit Refresh Frames, consistent with the owner-requested snapshot workflow.
Nuke Write evaluation stays on the main thread with cancellable progress. Cancellation retains
completed input snapshots. Existing nodes receive buttons through their existing upgrade action.
DLSS Live scheduling remains governed by ADR 0005; preparation does not broaden its triggers.

## Owner-requested update: fast single-frame processing and one input format

Process Current Frame may prepare its own missing inputs before inference; Frame Range and
tracking remain explicit. Ready inputs are always reused. Segment shares its SAM2-compatible
JPEG pool with single-frame segmentation, eliminating the PNG/JPEG double export. JPEG is lossy.
Refine and Inpaint use uncompressed TIFF for all input roles; DLSS retains float EXR. Output image
formats are unchanged. This update supersedes the original requirement that Current Frame and
Live must reject missing snapshots.

## Owner-requested update: automatic preparation for range processing

Process Frame Range and tracking now invoke the same preparation function as Cache Frames.
They scan the selected range, export only missing/invalid inputs, and submit model work only
after the full range is ready. Cache Frames remains available as a separate artist action.
Cancellation or export errors prevent model submission; completed inputs survive for retry.
This supersedes the earlier requirement to reject missing range snapshots.
