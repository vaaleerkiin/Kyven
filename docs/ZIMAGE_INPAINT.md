# Native Z-Image Inpaint for Nuke

The existing Generative Fill Group offers **Z-Image Base Q4 + Native ControlNet
Inpaint (Preview)** alongside FLUX.2 Klein. MOSS is retired from the node selector; its
catalog entry remains for managing old installations and cached work. Model files
are not silently deleted. Refreshing an old MOSS Group upgrades its selection to
native Z-Image with the new settings, without deleting its cached renders.

## Installation and controls

In Model Manager select **Z-Image Base Q4 + Native ControlNet Inpaint (Preview)** and
**Install Selected**. The existing bootstrap installs private Python/core when
needed, then `scripts/setup_zimage_fill.py` creates `.runtime/zimage-fill/venv`.
Nuke never imports Torch. The private runtime pins Diffusers 0.38.0, Transformers
4.57.6 and GGUF 0.19.0; Klein's and core ViTMatte's pins are unchanged.

The pinned bundle is approximately **14.10 GB decimal**, including Qwen3-4B.
Identical, SHA-verified Qwen/VAE files in another installed bundle can be reused
through hard links on the same volume. This saves downloads and physical disk
space; an unsupported filesystem falls back to normal verified downloads.
Replacing a model file is atomic and never writes through a shared hard link.

Choose the native backend in the Group. Defaults are working side **768**,
**28** steps, guidance **3**, context strength **0.9**, inference overlap **64**,
and automatic crop with context. The final output always uses the original
artist alpha. Native guidance, context strength and an optional negative prompt
are in Advanced. LanPaint solver knobs and manual VAE tile size are disabled for this backend;
VAE tiling is automatic. Automatic seamless blending applies to all native edit
operations, including additions and replacements; Off remains available.

Qwen automatically turns removal/addition/replacement instructions into an English
photographic description of the desired content and a negative prompt. It is a
text-only planner, not an image captioner; unspecified hidden geometry is inferred
by the image model. The visible English
editing instruction is kept separate from the model's target caption so automatic
translation retains the artist's requested action. The same Qwen weights support
Improve Prompt and its editable system instruction. Plans are cached in the live
worker; up to four text-condition pairs are retained in RAM. Switching models or
restarting the worker can evict these caches.

CLI requests set `model_id` to `z-image-base-q4-control-inpaint`. Relevant `fill`
settings are `zimage_guidance`, `zimage_control_scale` and
`zimage_negative_prompt`, in addition to common prompt/seed/size/color settings.
`prompt-enhance --model z-image-base-q4-control-inpaint` uses this worker too.

## Actual masked generation

This backend uses the trained **Z-Image Fun ControlNet Union 2.1 Lite** and native
Diffusers Z-Image/ControlNet modules. Its conditioning is the official 33-channel
tensor: 16 zero control latents, one known-region mask channel, and 16 encoded
masked-image channels. The source is cleared inside the inference mask before
this encoding. This learned condition participates in every conditional and
unconditional denoiser pass.

The native loop does not add an extra source-latent projection over the learned
ControlNet. That experimental constraint was removed after qualification.
The inference mask extends beyond the final artist alpha to move the model/VAE
transition away from the output seam. Binary mask expansion uses linear-time CPU
box counts; a 64-pixel expansion of the local 2048x1080 mask took 0.09 seconds.
Raw model RGB is decoded and recorded before color calibration or compositing.
The separate final Nuke composite preserves original float RGB/channels/alpha
outside the artist mask. VAE reconstruction alone does not provide bit-exact
original pixels.

The denoising loop uses sequential CFG, the official Z-Image velocity sign/time
convention, and FlowMatch Euler with the base model's pinned scheduler. It follows
the mask-conditioning contract without installing ComfyUI or the VideoX-Fun
application. The worker is separately GPL-3.0 like the other fill workers.

## Memory, errors and cancellation

This is an optional **8 GB GPU target**, with 32 GB system RAM recommended. Qwen
runs on CPU and unloads before image generation. VAE encoding, denoising and VAE
decoding are separate stages. Shared refiners and main transformer/ControlNet
blocks use bounded GPU residency; remaining blocks stream from CPU. CFG branches
run separately to bound activation memory. A two-GiB allocation reserve is kept
when placing resident weights. The worker retries once at a smaller canvas after
CUDA OOM; a second failure returns a recoverable error. Timeout/cancellation stop
and unload the worker through the existing provider protocol.

CPU inference is supported and smoke-tested with small native components; full
CPU frames are slow. The lower-memory LaMa/CPU baseline is unchanged. Independent
frame processing does not imply temporal consistency. Logs are under
`.runtime/zimage-fill/worker.log`.

## Validation

`tests/test_zimage_fill.py` checks model provenance, cache relevance, provider/IPC
isolation, English instruction/target-caption separation and verified weight
deduplication, including rejecting a same-size corrupt peer.

Run `tests/zimage_sampling_smoke.py` with the private interpreter to exercise the
actual native model/ControlNet graph on CPU and forced CUDA offload. Its
`--full-model` option uses the installed real Q4/Lite checkpoints: changing only
the conditioning mask changed the raw denoiser by a maximum 4.3203125 in the local
test, before any RGB composite. This checks learned conditioning before output compositing.

`tests/nuke/zimage_fill.py` checks actual Nuke controls, old-group migration,
float/HDR RGB and alpha preservation, and RGB sources without an alpha channel.
`tests/qualify_zimage_fill.py` records full-frame output, raw pre-composite RGB,
source/mask crops and measurements under `.runtime/zimage-research`. Artist QA
footage and model weights are not included in the repository.

Initial removal qualification on the same 2048x1080 street frame, RTX 4070 Laptop
8188 MiB, seed 42, 28 steps, working canvas 464x512: **140.42 seconds**, peak Torch
VRAM **3.76 GiB**, no OOM retry, known latent error **0**, and exact unchanged
pixels outside the artist mask. This initial run included the extra latent
projection that was subsequently removed. The foreground person was removed in raw model
RGB as well as the final image. Generated hidden geometry can still differ from the original scene.

The revised native profile (no extra latent projection, overlap 64, guidance 3,
752x768 canvas) removed the foreground man on the same frame. With the revised
automatic photographic caption it took **216.09 seconds** end to end and peaked
at **4.11 GiB** Torch VRAM, with zero outside-mask error and no OOM. Texture was
cleaner than the failed silhouette runs, but invented background details and
structural transitions remain. A manually supplied concrete background caption
also produced a cleaner removal, demonstrating sensitivity to text conditioning.

**Visual qualification has not passed.** Repeated removal runs can leave dirty
textures, a person-shaped ghost or unrelated objects. An addition test placed a
red vase on an unwanted rectangular backdrop even after improving the caption.
Increasing the canvas to 688x768 and using 40 steps/guidance 3/context 1 took
238.88 seconds and 4.07 GiB peak Torch memory, but made removal worse. A paired
same-seed test disabling the extra source-latent projection did not resolve the
artifacts either. The 40-step/high-control profile is not promoted to defaults. Pixel preservation
and learned mask conditioning passing tests do not establish visual quality.
Klein remains the default; this backend is an opt-in preview, without a claim of
Photoshop-equivalent color, edge matching, or reliable universal editing.

## Sources and licenses

- [Z-Image Base](https://huggingface.co/Tongyi-MAI/Z-Image), Apache-2.0.
- [Base Q4 GGUF](https://huggingface.co/unsloth/Z-Image-GGUF), Apache-2.0.
- [ControlNet Union 2.1 Lite](https://huggingface.co/alibaba-pai/Z-Image-Fun-Controlnet-Union-2.1), Apache-2.0.
- The small VAE is the Z-Image bundle's **16-channel AutoencoderKL**, not the
  incompatible 32-channel FLUX.2 VAE. Its configuration and weights have separate
  pinned hashes. Runtime checks reject an incompatible VAE.

Every downloaded artifact has a repository revision, expected size and SHA-256
in `resources/generative_fill/zimage_models.json`. Inference stays offline.
