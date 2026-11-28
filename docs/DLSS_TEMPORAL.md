# DLSS vectors and finite frame history

[DLSS workflow](DLSS.md) · [Architecture decision](adr/0006-dlss-finite-temporal-buffer.md)

This first temporal-buffer implementation adds a causal stabilizer around the native NR pass.
It uses external vectors and earlier neural corrections. It does not generate optical flow,
blend original beauty frames, or request future frames. Live scheduling remains frame/settings only.

## Native tools researched

| Tool | Useful role | Limit |
| --- | --- | --- |
| SmartVector → VectorToMotion | Generate vectors outside Kyven, then convert to conventional motion/forward/backward channels | Raw SmartVector layers are not ordinary XY displacement; convert first |
| SmartVector Flicker Compensation | Reduce luminance variation affecting vector estimation | Improves estimation input; it does not stabilize the DLSS result itself |
| F_DeFlicker2 | Separate NukeX/FurnaceCore deflicker pass, with motion-compensated analysis | May blur motion, reads neighboring frames and can be expensive; not inserted by Kyven |
| FrameBlend | Deliberate frame averaging effect | Foundry explicitly notes ghosting on moving features; not the DLSS history implementation |

Primary references: [SmartVector generation and Flicker Compensation](https://learn.foundry.com/nuke/content/comp_environment/smartpaint/generating_vectors.html),
[VectorToMotion](https://learn.foundry.com/nuke/current/content/reference_guide/time_nodes/vectortomotion.html),
[F_DeFlicker2](https://learn.foundry.com/nuke/content/reference_guide/furnacecore_nodes/f_deflicker2.html),
[FurnaceCore availability](https://learn.foundry.com/nuke/12.0/content/reference_guide/furnacecore_nodes/fc_nukex.html),
[FrameBlend](https://learn.foundry.com/nuke/content/reference_guide/time_nodes/frameblend.html).

The maintained worker already supplies native NR with external motion and reset flags, retaining
its private state on consecutive forward frames. It has no qualified public buffer-length control.
[NVIDIA's integration checklist](https://developer.nvidia.com/blog/?p=95492) emphasizes vector,
jitter, exposure and reset correctness. That reference concerns documented DLSS SR/RR integration;
it is not evidence that their flags or model presets can be copied into the undocumented NR feature.
Kyven does not invent jitter for compositing frames or add unverified native parameters.

## Connecting vectors

1. Calculate/cache vectors upstream. For SmartVector, add VectorToMotion before connecting Motion.
2. In DLSS Advanced, **Use Nuke Backward Channels** explicitly sets `backward.u/backward.v`,
   `forward.u/forward.v`, Pixels, Y Up and unit scales. It creates no nodes and computes no vectors.
   Custom R/G or UV inputs remain supported; existing selections are preserved until the button is used.
3. Backward at frame t must map current pixels to t-1. Forward at frame t maps current pixels to t+1.
   The consistency test samples the **previous frame's** forward vectors at the backward landing point.
   A feature moving 5 pixels right needs backward X=-5 and forward X=+5 in pixel space.
4. Enable **Check Forward / Backward** only when the named forward channels exist in Motion.
   Missing channels produce an explicit error; direction is never guessed by negating forward vectors.
5. Enable Temporal and process Frame Range forward from a deliberate shot boundary. Isolated stills,
   jumps and reverse playback start with less history. The buffer does not silently render pre-roll.

## Buffer and rejection

**History Buffer** holds 1–4 previous raw correction frames, default 3. The current frame is separate.
Backward fields are composed through previous frames to align each sample once. Original scene-linear
Source RGB remains the base. The blended output is never stored as another history sample, so old
corrections have a finite lifetime instead of recursively contaminating later frames.

**Temporal Stability** sets the total allowed history contribution. Accepted historical samples
are averaged with decreasing age weight; the number of samples does not multiply the total strength.
Photometric disagreement, black/changed alpha guides, off-screen motion and broken intermediate
correspondences reject reuse. **Check Forward / Backward** additionally rejects inconsistent vector
pairs. **Vector Error (Source px)** defaults to 1.5 source pixels and scales with Processing Size.

When Depth is connected, **Depth Rejection** (default 0.1 relative difference) rejects inconsistent
surfaces. It assumes corresponding depth values are comparable across frames; tune it for strong
depth changes. It does not provide a full geometric visibility solution. **History Clamp** defaults
to 2: old residuals are limited around the current residual by `clamp * (0.01 + abs(current_residual))`
in scene-linear units. Zero disables clamping. A disappearing large correction therefore does not
leave an unbounded trail, but excessive clamping can reduce temporal smoothing.

Cuts, gaps, repeats, reset, settings/session/resolution changes, provider replacement, cancellation
and errors clear the buffer. Provider unload/shutdown also releases the retained CPU arrays.
There is one active shot/session buffer; interleaving different nodes does not retain separate
private GPU histories. Still caches remain deterministic; temporal cached frames replay inference.

## Memory and diagnostics

**History Budget (MiB)** defaults to 512, configurable 64–1024. This bounds retained CPU history,
not peak process memory or native GPU allocations. Each float32 sample uses 36 bytes per processing
pixel, plus 8 for forward vectors and 4 for depth when present. At 1920×1080, three samples with
forward checks retain about 261 MiB; at larger sizes the budget reduces the effective sample count.
If even one sample exceeds the budget, processing fails before GPU inference with an actionable
message to lower Processing Size or increase the CPU budget. It never silently changes render size.

Reprojection uses 64-row strips with cooperative cancellation. The CPU buffer needs no CUDA/PyTorch
dependency; DLSS inference remains optional Windows/RTX with no CPU fallback. This does not establish
4 GB VRAM compatibility. OOM/timeout recovery continues to unload the worker and invalidate history.

Cache metadata reports `history_capacity`, `history_frames_buffered`, `history_cpu_bytes`,
`temporal_samples_used`, `temporal_reused_fraction` and `temporal_mean_weight`. Low reuse suggests
checking vector direction, units, channel names, disocclusions, masks and rejection settings before
increasing stability strength. API 36 / cache algorithm 5 introduced this behavior
as distinct from the earlier EMA; the current Nuke/server contract is API 48.
The node's History readout displays previous frames actually used, capacity, reused pixel fraction
and mean temporal blend for the last processed result. It does not claim to inspect the opaque native
DLSS history. These diagnostic updates never trigger Live processing.

Source RGB is not denoised. The buffer stabilizes only the correction generated by DLSS. NVIDIA's
[Ray Reconstruction](https://developer.nvidia.com/blog/generate-groundbreaking-ray-traced-images-with-next-generation-nvidia-dlss/)
replaces ray-tracing denoisers, but it is a separate integration from this NR feature. No exposed,
qualified native NR denoise control is currently available in this worker. Do not interpret NR as
a generic denoiser for camera footage. Intensity and Local Structure can reduce generated detail;
they are reconstruction controls, not Source denoise controls.

## Qualification and remaining work

`scripts/qualify_dlss_temporal.py` compares native NR history alone with native NR plus this buffer.
It uses a known one-pixel translation and analytically supplied vectors, not a new estimator.
The development RTX 4070 Laptop run on eight translated frames at 512×288 measured aligned residual
RMS 0.0017812 → 0.0015263 (about 14.3% lower), at strength 0.75 and three historical samples.
The buffer retained 19,464,192 bytes, all later frames stayed continuous and Source alpha was exact.
This is one controlled motion test; it is not a result for arbitrary real motion, smoke or fire.

```powershell
.venv\Scripts\python.exe scripts\qualify_dlss_temporal.py --worker .runtime\dlss\bin\KyvenDLSSWorker.exe --output .runtime\dlss-temporal-qa
```

Synthetic tests cover stronger alternating flicker, finite expiry, composed motion, occlusions,
FB mismatch, mask/depth rejection, budgets and cancellation. Remaining qualification should use
real externally estimated vectors on fast motion, transparency, thin structures and fire/smoke.
Poor or incomplete vectors can still ghost or reject nearly all history. Shot-specific tuning
and visual review remain necessary; future work can add a dedicated confidence input and diagnostics views.
