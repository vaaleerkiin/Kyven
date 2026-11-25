# ADR 0009: optional Nuke Generative Fill with a standalone worker

Status: accepted by explicit owner request, 2026-11-24.

The owner requested Z-Image Turbo Q4 + LanPaint for Nuke, an 8 GB memory target,
mask autocrop with context, unchanged pixels outside the mask, Qwen prompt
improvement and an editable system instruction in Advanced. ComfyUI was excluded.

The Group reuses Kyven's frame export, color transforms, cache, async HTTP queue
and cancellation. A lazy provider communicates through JSON lines with a
resident private process. Only the worker imports Torch, Diffusers, Transformers
and LanPaint. Its GPL-3.0 source ships beside its license; the Apache-2.0 Nuke
adapter and transport do not import GPL modules.

A trusted manifest pins every model file by revision, size and SHA-256. Inference
is offline. The private environment pins the two LanPaint repositories and
Diffusers/Transformers without changing core ViTMatte's dependencies, and reuses
the installed core Torch wheel. Qwen3ForCausalLM loads one checkpoint in RAM;
its base model provides embeddings and its language head improves prompts.
GPU stages are VAE encoding, Q4 denoising and tiled VAE decoding.

The optional profile starts at 768 pixels, caps Low Memory at 512 and offers one
smaller OOM retry. CPU execution remains available but slow. This does not replace
the CPU/4 GB baseline. Range generation does not promise temporal consistency.

The output permission mask is independent of inference growth. Inward feathering
and compositing over the original Source enforce this in the backend and native
Nuke graph. Test fixtures must include a provider that changes every crop pixel,
soft masks, inference growth, cached patches and float/HDR Nuke Source pixels.

## Model revision: FLUX.2 Klein 4B, 2026-11-24

The owner requested universal instruction editing and reconsidered the original
model after person removal produced a replacement person. Z-Image Turbo failed
removal qualification even after prompt rewriting. Replace that backend with
Apache-2.0 FLUX.2 Klein 4B Q4_K_M, keeping LanPaint, the same staged worker and
Qwen3-4B weights. Klein consumes the full causal LM's hidden layers 9/18/27.
Default diffusion steps become four. Removal, addition and replacement were
checked on the real source; pixels outside the supplied masks remain exact.
Old preview model IDs are migrated at request parsing without deleting caches.

## Additional backend: MOSS SFT INT8 + Edit, 2026-11-25

The owner requested an experimental MOSS backend in the same Nuke node, retaining
Klein and installing through Model Manager. The MOSS environment is separate to
protect Klein's pins. Published INT8 SFT and W4A8 Qwen checkpoints avoid downloading
or constructing complete BF16 base weights. The Edit adapter supplies reference
attention; LanPaint constrains known source latents during denoising, and the
existing composite separately enforces the final artist permission mask.
The adapter's weights license is unspecified, so commercial permission is not
claimed. This additive experiment is documented in [MOSS_FILL](../MOSS_FILL.md).

## Klein Base preview, 2026-11-25

The optional Base model uses a pinned Apache-2.0 Q4 bundle and 20-step, CFG 4
initial profile. Mask-constrained LanPaint and strict final alpha stay in place.
Base additions and replacements use automatic boundary blending. The condensed
Klein path omits its mathematically redundant unconditional pass at guidance 1;
Base retains both CFG passes. Shared files are hard-linked only after SHA-256
verification and fall back to verified download if linking is unavailable.
