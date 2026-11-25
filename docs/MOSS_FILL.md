# Experimental MOSS Generative Fill

> Retired from normal node selection; this document describes the legacy experiment.

MOSS is an additional backend in the existing native Nuke Generative Fill Group.
FLUX.2 Klein remains available and remains the default for existing and new nodes.
The MOSS backend uses a standalone worker; it requires no ComfyUI installation or
server. Heavy libraries never load in Nuke.

## Install and select

1. Update Kyven, restart Nuke and create a Generative Fill node, or refresh its
   models through Model Manager to upgrade the selector in an existing Group.
2. In Model Manager, choose **MOSS SFT INT8 + Edit + LanPaint (Experimental)** and
   **Install Selected**. Setup installs base Python when missing, then the isolated
   `.runtime/moss-fill/venv` and the verified model bundle.
3. Choose MOSS in the node's **Model** selector. Start with 20 diffusion steps,
   MOSS Guidance 4, two thinking steps and working side 768. Low Memory caps it at 512.
   For a faster preview, use working side **512**, **12** steps and **1** thinking
   step. This reduces detail and must still be inspected for alignment/seams.
4. Connect Source and the alpha mask, enter the edit instruction, set Cache Folder,
   Name and Version, then process a frame.

The complete download is about **13.32 GB decimal**: ready INT8 ConvRot MOSS SFT,
W4A8 GPTQ Qwen3.5-9B, the BF16 Edit adapter, small VAE, tokenizer and pinned code.
It does not download BF16 copies of the two large base models and does not
quantize them on first run. This replaces the initially explored 27.52 GB package.
Files have revision, size and SHA-256 identities in `moss_models.json`.

**Install Required Components** installs dependencies without model weights.
The node's **Install Runtime** button targets the selected backend. The optional
console route is `.venv/Scripts/python.exe scripts/setup_moss_fill.py --download-models`.
CLI fill requests select it with `"model_id": "moss-sft-int8-edit-lanpaint"` in
the request JSON. CLI prompt enhancement accepts
`prompt-enhance --model moss-sft-int8-edit-lanpaint --prompt "Remove the person"`.

## Editing and masks

The Edit adapter adds a source-reference prefix at timestep zero to MOSS attention.
Both CFG branches retain that reference during denoising. LanPaint also receives
the source latents and the conservatively downsampled mask at every diffusion
step, including its inner conditional sampling iterations. This changes the
generated latents themselves, before the VAE decodes RGB.
For REMOVE and REPLACE, selected packed latents are hidden from the Edit reference
so attention cannot directly copy the old selected subject. The original source
latents remain the separate LanPaint constraint. ADD and EDIT retain the full
reference, including the selected subject/background.

The final Nuke composite separately enforces unchanged original float Source
outside the artist's alpha. VAE reconstruction alone cannot provide bit-exact
source pixels. Model Mask Grow and Generation Overlap expand inference only;
they do not expand the final output permission. The backend also retains the
existing color calibration and optional gradient-domain boundary reconstruction.
These mechanisms do not guarantee a successful object removal or a perfect seam.

Qwen3.5-9B provides text embeddings, automatic English translation, operation
classification and **Improve Prompt**, using the same compact weights on CPU.
The editable Qwen system prompt continues to control prompt improvement.
Translations and operation labels are cached in the live worker; four prompt
condition bundles can also be reused across frames, masks and seeds. These are
bounded RAM caches and are lost when the worker restarts or another backend
causes it to unload. Improve Prompt itself continues to run the language model.

## Memory and recovery

This is an optional 8 GB target with 32 GB system RAM recommended. The text model
stays in RAM and is released before loading the image transformer. The VAE runs
in separate stages. A bounded subset of transformer layers stays on the GPU;
other projection/norm operations stream from CPU when VRAM is limited. This is
slower than a fully resident model but accommodates VRAM occupied by Nuke.
Reference KV caching works with streamed leaves as well. Smaller working canvases
reserve room for it before choosing resident layers. A free-VRAM guard can skip
the cache, and a cache allocation OOM falls back to uncached sampling.
CPU inference is supported but slow. One OOM retry reduces the working size;
a second failure is recoverable. Cancellation and timeout terminate the worker.

Klein's dependency pins and model files are retained. MOSS uses its own Diffusers,
Transformers and tensor-kernel packages. Logs are in `.runtime/moss-fill/worker.log`.
Range generation remains independent per frame and provides no temporal guarantee.

## Verification

`tests/test_moss_fill.py` covers catalog provenance, routing, cache identity,
license presentation and UTF-8 worker transport. The isolated-runtime scripts
`tests/moss_sampling_smoke.py` and `tests/moss_language_smoke.py` exercise real
MOSS Edit attention and LanPaint, quantized checkpoint loading, CPU generation
and forced CUDA streaming with small synthetic models. `tests/nuke/moss_fill.py`
checks the actual Nuke Group, saved-node upgrades and fractional guidance.

`tests/qualify_moss_fill.py --await-models` compares both full backends on local
QA inputs under `.runtime/generative-fill/removal-{source,mask}.png`. These
artist images and model weights are not distributed as test fixtures. Results
under `.runtime/moss-research/qualification` include the model's raw RGB before
color matching/compositing, the final image, masks and measured metadata.
Exact equality outside the mask verifies preservation; it does not establish
correct hidden geometry or perfect color/texture matching inside the mask.

### Same-frame measurements, 2026-11-25

RTX 4070 Laptop, 8188 MiB total VRAM; the local 2048x1080 street frame, seed 42,
auto crop/context 256, overlap 32 and exact final alpha. These are cold end-to-end
measurements including model verification and CPU text preparation; the newer
text/reference caches were added afterwards.

| Backend/profile | Working canvas | Steps/thinking | Time | Peak Torch VRAM | Visual result |
| --- | --- | --- | --- | --- | --- |
| Klein Q4 | 720x768 | 4/2 | 118.19 s | 3.31 GiB | Foreground person removed; some invented background geometry |
| MOSS, initial full reference | 720x768 | 20/2 | 1165.31 s | 5.64 GiB | Failed removal: retained the person |
| MOSS, hidden selected reference | 480x512 | 12/1 | 485.27 s | 5.19 GiB | Person removed in raw generation; geometric boundary mismatch remains |

Both MOSS runs completed without an OOM retry. Output outside the supplied mask
was exactly unchanged. **MOSS is not qualified as a production removal/seam
improvement over Klein on this frame.** The revised raw image removes the person,
but the regenerated entrance and pavement do not align perfectly with the
unchanged surroundings; boundary blending cannot guarantee a correction for that
structural drift. Keep Klein as the default. This is an optional experiment,
not a claim of Photoshop-equivalent fill quality.

A subsequent full-checkpoint CUDA microbenchmark at 480x512, with identical
inputs and 34 resident/8 streamed layers, measured one uncached forward at
3.298 s and a cached forward at 1.205 s (2.74x); maximum output difference was
zero. The guarded reference cache was enabled and peak Torch VRAM was 5.31 GiB.
This is a forward-pass comparison, not a claim of 2.74x end-to-end speedup:
first-use text encoding, model loading, cache prefill and decoding still cost
time. Repeated prompts can additionally skip CPU encoding through the bounded
condition cache. The MOSS operation-classification instruction is shorter than
Klein's; its semantic decision still comes from Qwen rather than keyword routing.

## Licensing and provenance

The [OpenMOSS SFT base](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT) is
Apache-2.0. The [Edit adapter](https://huggingface.co/chinoll/Moss-Image-Edit) has
no explicit weights license in its model card; its author's editing code is GPLv3.
Commercial permission for the Edit adapter is **not confirmed**. Model Manager
marks the entry experimental, does not advertise commercial use or redistribution,
and presents its terms before installation. The Apache base license alone does
not establish permission for the separate adapter.

Quantized checkpoints are pinned from
[Comfy-Org/MOSS-Image](https://huggingface.co/Comfy-Org/MOSS-Image).
Only its standalone Apache-2.0 tensor-kernel library is used to read/run those
formats; the ComfyUI application and its graph runtime are not dependencies.
The worker and mask sampler integration are distributed separately under GPL-3.0.
