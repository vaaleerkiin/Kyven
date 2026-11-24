# Generative Fill

**Nodes > Kyven > Generative Fill** is a native Nuke Group with Source and Mask
inputs, local FLUX.2 Klein 4B Q4_K_M generation and LanPaint. It has no ComfyUI
dependency and uses no remote inference service.

## Model choice and edit instructions

The backend uses the instruction-trained FLUX.2 Klein 4B model. Describe the edit
in natural language: remove, add, replace or modify. The original crop is supplied
as the reference image throughout LanPaint sampling. No manual operation switch
or negation-to-description rewrite is required.

The initial Z-Image Turbo implementation was rejected after a real removal request
produced another person. Rewriting the prompt and clearing the source hole were
not reliable fixes. Klein was qualified on removal, addition and replacement
before becoming the default. These examples do not guarantee every edit will
succeed; inspect the generated patch before accepting it.

## First use

1. Open **Model Manager**, select **FLUX.2 Klein 4B Q4 + Qwen3-4B + VAE**, then
   **Install Selected**. This installs/verifies base components, the isolated
   runtime and the complete 10.83 GB model bundle. Files are revision-pinned and
   size/SHA-256 verified. The manager also opens before the server is installed.
2. Connect Source and an alpha mask. White alpha selects edits; black preserves
   Source. Mask-input RGB is ignored.
3. Enter Prompt in English or Russian. Non-Latin prompts are automatically
   translated to English. Improve Prompt and Restore Original are optional.
4. Set Cache Folder, Fill Name and Version, then **Process Current Frame**.

**Install Required Components** in the manager prepares dependencies without
weights. The node's **Install Runtime** button uses this same manager installation
path. Python 3.10–3.13 is required; Python 3.12 is recommended.

The equivalent optional command-line setup is
`.venv/Scripts/python.exe scripts/setup_generative_fill.py --download-models`.

## Mask, context and output

Auto crop finds the mask/model-mask bounds, adds **Context Padding** (256 source
pixels by default), and clips to the image. The crop is reduced to **Working
Size**, preserving aspect ratio, and edge-padded to a 16-pixel grid. The generated
patch returns to source crop resolution. Manual ROI and Full crop also work.

**Model Mask Grow** affects inference only. The original interpreted mask remains
the output permission mask. **Result Edge Softness** feathers inward. Both the
backend and Nuke graph restrict edits to that permission. Black-mask RGB comes
from the live original float Nuke Source, bypassing the model round trip. The
default **Result** also keeps Source alpha; other output modes intentionally
show mask alpha, generated patch or difference. An empty mask loads no model.

RGB-only Source inputs receive opaque alpha instead of attempting to copy a
missing channel. Existing Source alpha is preserved, including after upstream
channel changes. The graph repair path also upgrades saved preview groups.

Generation uses sRGB pixels and the standard Kyven input-colorspace selector.
Generated RGB is cached as 8-bit sRGB; HDR reconstruction inside the selected
area is not supported. Unselected Source pixels retain their original Nuke
precision. Refresh input snapshots after upstream edits using **Refresh Frames**.
Range processing generates each frame independently, without temporal consistency.

## Advanced

The **Advanced** tab exposes working size, diffusion/thinking steps, early stop,
Langevin step size, friction, constraint strength, noise beta, VAE tile size,
OOM retry, timeout, memory profile and mask/edge controls. Distilled Klein guidance stays
fixed at 1; negative-prompt/CFG controls are not presented. Defaults: working
side 768, 4 scheduler steps, 2 thinking iterations. Low Memory caps the side at 512.

The Qwen section contains an editable **System Prompt**, output-token limit,
temperature, top-p, timeout, reset button and last improved prompt. The system
instruction affects prompt improvement only. Qwen is text-only and cannot see
the source image. The same Qwen3-4B checkpoint supplies language generation and
image prompt embeddings; no second LLM download is needed. Late responses do
not overwrite a newer artist edit. Both actions use the shared cancelable queue.
The interface and generated prompts are English. Improve Prompt also translates
Russian input to English. Output token filtering prevents Qwen from switching
to Chinese/Cyrillic mid-answer. Automatic translation preserves the brief instead
of expanding it; the editable system instruction controls Improve Prompt.

## Memory and recovery

This optional 8 GB profile leaves the existing CPU/4 GB LaMa tools intact. Qwen
stays in system RAM. VAE encoding, Q4 denoising and tiled VAE decoding use the GPU
in separate stages. Expect approximately 32 GB of system RAM. CPU text generation
can be slow; explicit CPU image inference is also supported through `--device cpu`.

On CUDA OOM, the default retries once at a smaller size and records the actual
working size in result metadata. A second OOM returns a recoverable error. Cancel,
timeout or provider unload terminates the private worker and releases its models.
Successful workers stay resident between requests. Diagnostics are written to
`.runtime/generative-fill/worker.log`. The 8 GB profile is a target; performance
depends on the image and working size and must be measured on the target GPU.

Qualification on an RTX 4070 Laptop 8 GB / 32 GB RAM (2026-11-24), using a
2048 x 1080 street image, masks and 256-pixel context:

| Check | Measured result |
| --- | --- |
| Person removal, Klein + LanPaint | Foreground person removed; background reconstructed |
| Add a red vase, Klein + LanPaint | Vase added in its separate sidewalk mask; person retained |
| Replace person, Klein + LanPaint | Requested woman in a blue coat generated |
| Diffusion / thinking steps | 4 / 2 |
| Actual working canvases | 688 x 768 for removal/replacement; 736 x 544 for addition |
| GPU image generation, precomputed English embeddings | 30.2-37.2 s |
| Peak PyTorch GPU allocation | 3.17-3.29 GiB; excludes other processes and driver memory |
| Full cold request with automatic Russian translation | 67.4 s, 3.29 GiB peak |
| OOM retry | Not needed |
| Pixels outside each input mask | Exactly unchanged in all three runs |

These GPU timings exclude model loading, CPU text encoding and translation. Full
request time is longer. A separate native-edit reference without LanPaint removed
the person in 9.3 s at 3.83 GiB, but is not the node's sampling path.

The native Nuke 16 graph was also checked with cached patches, HDR/negative Source
values, Source alpha, mask changes and save/reopen. The lightweight CPU smoke test
uses tiny random models to verify the real LanPaint adapter; it is not a full
CPU performance or image-quality benchmark. Other image sizes and masks may differ.

## CLI and API

`kyven generative-fill --request request.json --models-dir models` reads absolute
source, mask and output paths, crop/mask settings and a `fill` settings object:

```json
{
  "source": "D:/shot/source.png",
  "mask": "D:/shot/mask.png",
  "output": "D:/shot/fill.png",
  "context_padding": 256,
  "mask_grow": 8,
  "edge_softness": 4,
  "fill": {"prompt": "a red ceramic vase", "seed": 42, "working_size": 768}
}
```

`kyven prompt-enhance --prompt "a red vase" --models-dir models` improves text;
`--system-prompt instructions.txt` uses a custom system instruction. HTTP routes
are `POST /v1/jobs/generative-fill` and `POST /v1/jobs/prompt-enhance`; use standard
job polling/cancellation. Image cache identity includes source, masks, prompt,
seed, crop, all solver settings and pinned model identity.

## Licensing

Klein, Qwen and VAE weights are Apache-2.0. LanPaint and its Diffusers pipeline
are GPL-3.0. The standalone worker source is separately GPL-3.0 and ships with its
license in `src/kyven/resources/generative_fill/`; the Nuke adapter and IPC client
remain Apache-2.0 and do not import GPL modules. Distributions must preserve the
worker/dependency licenses and corresponding source. See
[third-party notices](../THIRD_PARTY_NOTICES.md) and
[ADR 0009](adr/0009-generative-fill.md).

## Updating the early Z-Image preview

Saved preview requests with the old model ID are migrated to Klein. The old
nine-step default becomes four; seed and mask settings are retained. Existing
rendered caches are not deleted. The new model identity invalidates the old
inference cache. Reload Kyven or restart Nuke after saving the script to refresh
an already loaded preview node's Python module and model label.

## Automatic color matching

Advanced > Color Matching defaults to Auto with strength 1. Paired clean context
pixels calibrate gain and offset; pixels of the removed source object are excluded.
For removal, Qwen selects conservative flat-background continuity matching. Short
isolated scanlines and inconsistent surfaces are excluded to avoid color bands.
Additions/replacements retain their new subject colors using context calibration.
Context, Background and Off modes are available for manual control. No extra
model is downloaded. Matching runs before the strict permission-mask composite,
so pixels outside the artist mask remain unchanged. Older preview nodes inherit
Auto on the server, and the changed inference identity invalidates old results.

## Mask boundary protection

Generation Overlap defaults to 32 source pixels. This expands only the model's
inference region; it never expands the output permission mask. Partially selected
latent cells are treated as editable instead of using nearest-neighbor mask
sampling, preventing pieces of the old silhouette from being locked in place.
The extra generated band keeps the model/VAE transition away from the final edge.
A tight artist mask can still leave unselected blur or shadow in Source; those
pixels remain protected. Inspect the patch at full resolution before accepting it.

## Seamless Blend

Seamless Blend defaults to Auto. For removals, a CPU gradient-domain solve matches
boundary colors to the reconstructed surrounding background; it does not use the
removed subject as its boundary reference. All permitted boundary pixels participate in the sparse Poisson solve;
image-edge neighbors use a zero normal derivative. The patch keeps generated interior gradients and is copied only through
the original permission mask. Inward feathering is bypassed for a successful solve
so it cannot mix the old subject back into the edge. Soft artist alpha is retained.

Auto keeps additions/replacements on context color calibration to preserve new
subject colors. On forces the solve for those edits too; Off restores standard
masked compositing. A generated structural mismatch or an unmasked shadow can
still be visible; the solver does not make arbitrary images identical to Photoshop.
OpenCV builds the clean boundary reference; SciPy solves the sparse system. Both
run only inside the private optional worker and use no additional VRAM.
