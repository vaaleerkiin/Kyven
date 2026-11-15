# Kyven DLSS (optional RTX)

[Documentation home](README.md) · [Nuke workflow](NUKE.md) · [HDR and guide decision](adr/0004-dlss-hdr-and-control-guides.md)

Kyven DLSS applies local neural reconstruction at Source resolution through the worker protocol from
[DLSS5-for-Nuke](https://github.com/KJzzzKJ/DLSS5-for-Nuke/tree/d61307ef4543c8991668730a4cab8019b3cb0096).
It requires Windows, an NVIDIA RTX GPU, Kyven's bundled worker and a separately licensed NVIDIA
Neural Rendering component. The runtime uses
undocumented interfaces. Transport tests alone do not establish runtime compatibility or the
artistic effect of its experimental guide inputs.

## Setup

Open **Nodes > Kyven > Model Manager**, select **Kyven DLSS Runtime**, then **Install Selected**.
The manager installs the bundled, checksum-verified `KyvenDLSSWorker.exe`, shim and pinned OpenEXR
dependency in Kyven's private environment. End users do not need Git, a compiler, NDK or pip commands.
Kyven does not inspect or launch a worker from another plugin or `.nuke/DLSS5Live`.

The preview package now includes `nvngx_dlssnr.dll` alongside Kyven's worker. **Install Selected**
copies it locally after checking the pinned size and SHA-256; no runtime download or other plugin
is needed. OpenEXR is installed into the private Python environment when missing, which can still
require internet access. The component is included in both the installable ZIP and Python wheel.

Distributions built without that DLL can download it during **Install Selected** from the pinned
[DLSS5-for-Nuke v1.0.0 community release](https://github.com/KJzzzKJ/DLSS5-for-Nuke/releases/tag/v1.0.0).
The manager presents the source and terms before starting. This is not presented as an official NVIDIA
SDK or a verified redistribution grant. The public NVIDIA SDK inspected on 2026-11-15 does not carry
NR; the community ZIP does, despite the current upstream README's narrower runtime policy.

Kyven verifies the exact archive size/SHA-256 and the extracted NR DLL size/SHA-256 against
`resources/dlss/vendor-source.json`. It extracts that one member, never the foreign worker, ReShade
or RenoDX. All executable processing uses Kyven's own worker inside its managed folder. Cancelled
or mismatched downloads do not publish a DLL. **Import NVIDIA Component** remains available for a
separately obtained licensed DLL. An incomplete install is never marked Ready.

For development builds, `scripts/build_dlss_worker.ps1` compiles sources from `native/dlss` and
packages the portable worker with static MSVC CRT and its SHA-256 manifest into distribution
resources. The built wheel includes those files. See [ADR 0003](adr/0003-managed-dlss-runtime.md).

Restart Nuke after updating Kyven. Existing nodes can be upgraded in place using **Upgrade Selected
DLSS Node**; UUID, inputs and settings are retained. The layout uses the same native sections as
Segment, Refine and Inpaint. Expert vector controls live under **Advanced**.

Create **Nodes > Kyven > DLSS (Optional RTX)** and connect Source. Live is enabled by default,
but Cache Folder must be selected before caching or processing. Set Cache Folder, Name and
Version to choose the output location. A blank folder reports an error and writes nothing. Inputs:

| Input | Label | Meaning |
| --- | --- | --- |
| 0 | Source | Full-frame scene-linear RGB; Source alpha is preserved |
| 1 | Motion | External **current-to-previous** displacement; required for Temporal |
| 2 | Depth | Optional guide; default `depth.Z`, runtime assumes inverted depth |
| 3 | Mask | Artist alpha guide; used when Mask Alpha Input is Mask Input |

## Quick start

1. Install the runtime in Model Manager and create **DLSS (Optional RTX)**. Connect Source.
2. Set Cache Folder, Name and Version. Choose a mask guide if you want to restrict changes.
3. Press **Process Current Frame** and compare Result with Source. Reduce Processing Size if memory is insufficient.
4. For a stable range, connect correctly directed external Motion vectors, configure their channels and units, enable Temporal and process forward from the shot start.

Hover over controls for English help. For input snapshots and upstream changes, see [input preparation](NUKE.md#input-preparation-and-processing).

## Alpha as an inference guide

**Mask Alpha Input** offers **Full Frame** (default), **Source Alpha** and **Mask Input**.
White permits neural changes, black preserves Source and grey provides a soft neural guide.
The connected mask always uses its alpha channel `A`; there is no channel selector. Full Frame ignores the connected
Mask input. Source alpha is always preserved on output, including HDR/out-of-range alpha values.

Kyven worker 2.0 sends the clamped scalar artist mask directly as four equal channels in an
`RGBA32F DLSSNR.ControlMask` texture. It initializes the native backbuffer with the current image
before evaluating NR. Earlier experiments used an R32F texture and an uninitialized backbuffer;
that invalid combination caused black/incorrect results and the earlier disabled controls.
The working RGBA convention is **white allows changes**, with no inversion.

**Protect RGB Outside Mask** additionally removes neural corrections at exactly black pixels
after residual resizing and temporal processing. Soft values are not multiplied a second time.
Disable it to examine native guide response alone; the mask still enters inference. No RGB Premult
is used to implement the guide. The output copies generated RGB onto Source and retains its other channels.

After updating Kyven, reinstall **Kyven DLSS Runtime** in Model Manager to update the small worker.
An older worker is reported as Update Required and is rejected for masked inference. The bundled
NVIDIA DLL is reused; no other plugin is involved.

Supply straight RGB when Source has already been premultiplied; use an explicit upstream Unpremult
where appropriate. The node adds no implicit Premult/Unpremult operations.

## Internal HDR rolloff

**HDR Rolloff** defaults on. It conditions scene-linear RGB before inference, then restores the
neural change to the original scene-linear range. The original full-resolution Source remains the
base: unchanged inference preserves its HDR values, negative RGB and alpha exactly, including the
precision lost in half-float transport. This is an internal working transform, not an output look.

The **Rolloff Knee** in Advanced defaults to 0.5. Positive values below the knee pass unchanged;
above it, a smooth shoulder compresses the pixel peak toward 1 and applies one gain to RGB,
preserving positive color ratios. Negative minima are temporarily lifted. The original lift cancels
when restoring the neural residual using the saved inverse gain. There are no frame statistics or
automatic exposure changes, so another highlight entering the shot cannot change every pixel's
working exposure. Actual neural changes to highlights remain changes; this does not freeze bright pixels.

Model input is quantized once to the actual half-float values used on the wire. Residuals are
measured against that same input, avoiding round-trip drift. CPU temporaries use 128-row strips
and one inverse-gain plane at processing resolution. Finite values above 65504 can be processed
with rolloff enabled; disabling it restores the half-float input limit. Nonfinite restored float32
results fail before publication. Changes to rolloff/knee invalidate caches and temporal history.

The optional GPU qualification can be repeated with synthetic input or a local EXR:

```powershell
.venv\Scripts\python.exe scripts\qualify_dlss_guides.py --worker .runtime\dlss\bin\KyvenDLSSWorker.exe --output .runtime\dlss-guide-qa
```

On the development RTX 4070 Laptop, the native black-mask difference was zero, white matched full
NR exactly, split masks protected the black region, and soft masks changed inference with final
protection disabled. Service qualification also covered HDR/negative values, alpha precision and
three forward temporal frames with supplied zero vectors. This does not establish quality for every
shot, runtime, driver or GPU, or a 4 GB VRAM guarantee. RGBA guide storage adds 16 bytes per processing pixel.

## Temporal setup

See [vectors, native tools and the finite frame buffer](DLSS_TEMPORAL.md) for the new controls,
memory limits, qualification results and remaining real-motion work.

1. Generate vectors externally and connect Motion.
2. Select exact EXR channels in **Motion X/Y Channel**, for example `R/G` or
   `backward.u/backward.v`. No channel or direction is guessed.
   **Use Nuke Backward Channels** applies the conventional Nuke mapping explicitly.
3. At frame `t`, vectors must point from each current pixel to its position at `t-1`.
   A feature moving 5 px right needs X = -5. Negating forward flow does not generally produce
   correct backward flow; obtain the direction from your vector generator.
4. Choose **Pixels** or **UV**. UV displacement is multiplied by width/height. With **Motion Y
   Points Up (Nuke)** enabled, Y is converted to top-down worker coordinates. A feature moving
   up by 3 px needs current-to-previous Y = -3 in Nuke's convention.
5. Enable **Temporal** and process **Frame Range** forward from a shot boundary.

**Temporal Stability** blends motion-aligned changes introduced by DLSS, without blending original
RGB frames. Start at 0.35. Lower **History Rejection Threshold** rejects history more aggressively
where Source disagrees with reprojected previous Source. Changed mask edges, off-screen vectors
and photometric disocclusions reject history. **Scene Cut Threshold** resets on large photometric
discontinuities. These checks cannot detect every occlusion; repeated textures or bad vectors can
still cause artifacts.

**History Buffer** defaults to three previous raw corrections, bounded by the CPU history budget.
Optional forward/backward checks, depth rejection and a history clamp provide additional safeguards.
The accumulated output is not recycled into the buffer, so history has a finite horizon.

Scrubbing, reverse playback, gaps, repeated frames, settings/session changes, provider unload,
cancellation and errors reset history. **Reset History** starts another history generation. A range
always resets on its first frame. Render final sequences forward from a deliberate shot start:
an isolated live frame has less history and may differ.

An explicit reset recreates the worker's private native state. Real-runtime A/B testing showed that
the undocumented Reset flag alone could retain an old control-mask effect on repeated stills.
Continuous forward temporal frames keep the resident worker; independent frames and discontinuities
pay initialization cost for consistent masking and history isolation.

## Live mode

**Live Current Frame** starts enabled. After its initial preview, only two changes schedule a render:
the current frame changes, or an artist processing setting inside this DLSS node changes.
**Live Debounce** waits for edits/scrubbing to settle. One request runs per node; newer frame/settings
requests cancel obsolete work. A 100 ms timer finishes debounce even without another UI event.
It reads only the frame and the node's own settings; it does not validate or hash upstream nodes.

Upstream image edits, rewires, file/runtime changes, Read reloads, cache publication, model refresh,
status changes and redraws do not schedule or restart Live. After an upstream edit, select **Current Frame** under Input Frames and use **Refresh Frames**, then press **Process Current Frame** to regenerate the result. Changing only output display or debounce timing also does
not request inference. Rapid own-setting changes coalesce. After cancellation for a newer actual
intent finishes, that pending intent starts automatically; internal cancellation does not retry an
unchanged intent. Disabling Live or deleting the node stops its wake-ups.
The latest signature is checked before moving results into the public cache. Failed requests do
not retry endlessly: edit a setting, toggle Live or press Current Frame to retry. **Cancel** also
switches Live off when cancelling a live request.

Nuke owns graph export on its main thread. Inference, decoding and temporal processing run outside
Nuke. Heavy upstream graphs can make export slow. Live means asynchronous preview refresh;
it does not promise 24 fps playback.
Internal Write/Read updates do not count as artist edits. Live state is scanned from Root even
when a nested Group is open. Source hashes and connection identity are absent from Live intent.
Nuke can leave `execute(Write)` at `last_frame + 1` until its main-thread callback returns. That
private render context is not observed as a timeline edit: validation waits for the UI context to
return, and the final publication check still rejects an actual scrub. Live scans and per-node
polls reject reentrant callbacks. Publishing reloads only the DLSS Read and redraws the Viewer;
it does not clear the global RAM cache or force upstream graphs to revalidate from publication.

## Resolution, memory and cache

**Processing Size** is a scale menu: **1**, **0.5**, **0.25**, **0.125**, defaulting to **1**.
It scales Source width and height for neural processing: 1920×1080 at 0.5 processes at 960×540.
Output retains the full-resolution Source base with the restored neural correction. Processing
is bounded to 64–8192 pixels on the longest side by the backend request contract; smaller Source
images are not upscaled. Reduce the scale if GPU memory is insufficient.
Existing nodes receive the menu on restart/upgrade, preserving their UUID and cached outputs.
This version does not expose DLSS Super Resolution.
It never creates a vector generator or selects upstream Auto DIS mode.

Full-frame mode processes RGB regardless of Source alpha; alpha is preserved on output.

The runtime has no verified VRAM requirement here. Reducing Processing Size reduces memory but does not prove
4 GB compatibility. There is no CPU DLSS inference; core Kyven tools retain CPU support. A runtime,
OOM or timeout failure unloads the worker and suggests a lower limit. Provider switching unloads
the previous provider, releasing its child process.

Public outputs are ordinary float EXRs:

```text
<Cache Folder>/<Name>/<Version>/<Name>_<Version>.1001.exr
```

Cache Folder is required. DLSS does not fall back to a hidden local cache.

Exports, staging outputs and manifests stay in `_kyven_work/`. Completed staging files are removed
after publication/cancellation. **Load Cache** reconnects an existing sequence. Missing frames read
black: process a complete range before final rendering. Results remain usable without the server.

Still jobs can reuse validated cache files. Temporal jobs replay inference to rebuild private DLSS
history even when output EXRs exist. Keys include input/guide content, settings, worker/runtime
SHA-256 identities and preceding history. A different shot start can produce a different result.

## CLI / API

```powershell
.venv\Scripts\kyven.exe dlss --request D:\shot\dlss-request.json
```

Example JSON; all paths must be absolute:

```json
{
  "source": "D:/shot/source.1001.exr",
  "output": "D:/shot/dlss.1001.exr",
  "worker": "D:/Kyven/.runtime/dlss/bin/KyvenDLSSWorker.exe",
  "motion": "D:/shot/vectors.1001.exr",
  "mask_source": "none",
  "temporal": true,
  "session": "shot010",
  "frame": 1001,
  "motion_x": "R",
  "motion_y": "G",
  "motion_units": "pixels",
  "motion_y_up": true,
  "max_dimension": 1024,
  "hdr_rolloff": true,
  "rolloff_knee": 0.5
}
```

Send the same contract to authenticated `POST /v1/jobs/dlss`, poll `/v1/jobs/{id}`, and cancel via
`/v1/jobs/{id}/cancel`. Sequential submissions in one server preserve temporal state. Separate
CLI invocations start independent workers and cannot share temporal history.
