# ADR 0002: Optional local DLSS reconstruction with external motion

- Status: Accepted for implementation; GPU/runtime qualification pending
- Date: 2026-11-15

## Context

The project owner explicitly requested a DLSS node based on
[DLSS5-for-Nuke](https://github.com/KJzzzKJ/DLSS5-for-Nuke), with live updates, temporal stability,
external motion vectors and alpha used as an inference guide. This authorizes one optional product
area in addition to the three core tools.

## Decision

Use `dlss-external` behind the authenticated server and existing single GPU queue, with the D5V5
pipe contract at upstream commit `d61307ef4543c8991668730a4cab8019b3cb0096`. The Python Nuke Group
exports float EXRs; the backend owns decoding, worker lifecycle, inference, history and atomic output.
No Nuke native DLL ABI is involved.

The user supplies a compatible local worker/runtime. A helper builds the pinned upstream worker
and project-authored caller shim without downloading proprietary binaries. This is an explicit
exception to ADR 0001's model catalog: there is no approved downloadable DLSS checkpoint/runtime.
Worker and adjacent DLL SHA-256 identities participate in cache/history keys.

Expose native reconstruction only. Reduced-resolution inference restores a neural residual onto
full-resolution Source; this must not be advertised as DLSS Super Resolution. Do not tile private
temporal state. Require external current-to-previous vectors for temporal mode; never generate flow.

Reset history on nonconsecutive frames, cuts, settings/session changes, provider replacement,
failure and cancellation. Stabilize the neural residual with reprojection and photometric/mask
rejection. Replay temporal cached frames to reconstruct private runtime state instead of skipping them.

Send alpha/control masks as float `DLSSNR.ControlMask` guides, independently of Source alpha.
Preserve alpha, avoid implicit premultiplication, and optionally protect RGB outside the mask after
inference. The undocumented runtime's interpretation of ControlMask needs real-runtime A/B
qualification; passing a texture does not prove its semantic effect.

Live uses debouncing, cancellation, upstream graph signatures and a final signature check before
publication. Stale results stay private. Live means asynchronous current-frame refresh, without a
real-time playback frame-rate guarantee.

## Hardware and validation

The core 4 GB VRAM / 16 GB RAM and CPU policies remain in force for Segment, Refine and Inpaint.
Optional DLSS is Windows/RTX only and gives a structured unsupported-device error on CPU.
Default processing is limited to 1024 pixels on the long side, with one reduced-frame history and
strip-based CPU reprojection. Failure kills/unloads the worker and suggests a lower processing
limit; no automatic retry silently changes render quality.

Synthetic tests cover mask delivery, HDR/alpha, vector conversion, working-size limits, rejection,
history/cache chains, real subprocess pipe framing, timeouts, cancellation, authenticated jobs and
live supersession. Real GPU memory, quality, runtime compatibility and Nuke interaction must be
qualified on the target workstation.
