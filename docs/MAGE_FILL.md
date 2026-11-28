# Mage Generative Fill

[Documentation home](README.md) · [Generative Fill guide](GENERATIVE_FILL.md) ·
[Visual walkthrough](NODES.md#generative-fill) · [Nuke workflow](NUKE.md)

New Generative Fill nodes default to **Mage-Flow Edit Turbo Q6**.
Existing FLUX and Base nodes keep their selected model. Use the Model menu to
switch. Model Manager's Set as Default affects new nodes. LanPaint and Base-only
controls are hidden for Mage.

In Model Manager select Mage and **Install Selected**. **Install Required
Components** prepares its private Python environment without downloading weights.
The node's **Install / Repair Model** button installs or verifies the selected
model and all required components in one operation. Model Manager opens with
that model selected. The Windows
installer also offers Mage, or run `install.cmd -Model mage-flow-edit-turbo-q6`.
The verified download is approximately 7.69 GB, plus extracted runtime files.
Model files, CUDA DLL archives and CPU prompt helper are SHA-256 verified. Mage's
environment does not change FLUX's dependencies. Runtime executables run hidden.
Installing only Mage uses a lightweight Kyven base and a separate NumPy/Pillow/
OpenCV/SciPy worker. It does not install PyTorch, SAM, FLUX weights or Git.
Verified model files are reused when repairing an installation. Keep roughly
12 GB free for the model bundle and extracted executables, plus Python packages.

## Responsive Lab Color Match

After one generation, these controls update cached pixels directly in Nuke:

| Control | Default | Use |
| --- | --- | --- |
| **Color Matching** | Lab | Lab matches clean Source context; Off shows raw generated colors |
| **Color Match Strength** | 1 | Master amount of the fitted Lab correction |
| **Match Luminance** | 1 | Match brightness/contrast independently of color |
| **Match Chroma** | 1 | Match the Lab a/b channels independently of brightness |
| **Grow / Erode (px)** | 0 | After generation: positive expands the blend mask, negative shrinks it |
| **Mask Blur (px)** | 0 | Ordinary Gaussian Blur after grow/erode, in Source pixels |
| **Preview Blend Mask** | Off | Inspect the final alpha after grow/erode and blur |

For example, set Match Luminance to 0 and tune Match Chroma to correct a color cast
while retaining generated brightness. Set strength to 0 or choose Off to compare
the raw patch. Preview Model Mask shows the exact binary alpha exported to Mage.

The fit uses standard Reinhard mean/std transfer in D65 Lab, inspired by the
[DLSS5 Color Matcher in this ComfyUI pack](https://github.com/Strix-3D/ComfyUI-DLSS5-Enhancer-v15/blob/main/nodes.py).
Statistics use paired clean context outside the inference mask and its fringe;
removed Source subjects are excluded. Insufficient context leaves the patch
unchanged. Bounded NumPy samples need no new model or heavy dependency.

Generation stores raw and fully matched straight-RGBA patches, with generated
crop coverage in alpha. Native Nuke operators blend luminance/chroma in Lab and
process the result alpha. The Group preserves Source alpha and auxiliary channels;
RGB outside the final blend alpha comes from live float Source. Color Match cannot
correct geometry or remove an unmasked shadow.

**Create Base Result Read** opens the base cache before live adjustments. Use an
ordinary Nuke Write connected to the Group to save Color Match, result mask,
Source alpha and auxiliary channels. Changes to live controls leave generation's
request identity and cached bytes unchanged. Old caches remain readable; generate
once to prepare paired patches. Mage forces experimental boundary passes off.
FLUX and legacy CLI requests retain Auto/Context/Background worker matching.

## Input mask and result mask

Mage receives the nonzero input alpha as a binary mask, with optional Invert Mask.
The same graph feeds Preview Model Mask and the exported model-mask file. Hidden
legacy Model Mask Grow, Threshold and Generation Overlap do not affect Mage.
The worker crops and resamples this mask to Processing Resolution; no result mask
control adds dilation, erosion or blur on the model path.

The separate **RESULT MASK / AFTER GENERATION** block contains **Grow / Erode**,
**Mask Blur** and **Preview Blend Mask**. Its pipeline is input alpha, signed
FilterErode, ordinary Gaussian Blur, then generated crop coverage. Blur does not
automatically erode or multiply the original alpha back in. Both controls default
to 0. Try Grow / Erode **+8** and Mask Blur **4** to extend and soften a composite,
or Grow / Erode **-2** and Blur **0** to tighten its edge. Values are Source pixels
and update the cached result immediately without generating again.

Grow and blur can expose generated context beyond the original alpha, up to the
actual generated crop boundary. Increase Context Padding and generate again if
you need more generated coverage. Grow cannot reconstruct content that the model
left unchanged outside its edit mask.

For removal, supply an input mask covering the object, shadow and edge
contamination. Blurring a tight subject matte can mix the original subject back
into the result. Post mask controls only change compositing; change the input
mask and generate again when the model must remove more content.
The old compound Smooth Removal Edges preset is hidden. Result mask adjustments
never change crop, resolution, context, model mask or color settings.

The default processing side is 512. This limits the crop, not the output frame.
Processing Resolution is a dropdown: Mage offers 256 (preview), 512, 768, 1024,
1280, 1536, 1792 and 2048; FLUX stops at 1024. Existing custom sizes are retained.
Existing nodes acquire the range when their controls refresh after loading the
updated Python module. Restart Nuke to load an update in an existing session.
Above 1024, the editing reference is reduced to a maximum side of 1024 to fit
Qwen3-VL's token context. The initial image, inference mask and diffusion canvas
retain the selected processing resolution; this is not a 1024 render upscaled
after generation. The backend also uses that reduced reference for conditioning.
Qwen text encoding runs on CPU; the quantized generator and VAE run in staged
CUDA execution. CPU inference is selectable. A GPU allocation failure can retry
at a smaller processing size. Cancel or timeout terminates the worker tree.

Prompt improvement uses the same Qwen3-VL GGUF text weights on CPU. Its system
instruction, token count, temperature, top-p and seed remain configurable. Russian
image prompts are translated locally. No model calls a hosted image service.

## Memory failures and recovery

The worker recognizes CUDA allocation failures and managed **RAM graph/weight
capacity exceeded** failures. **Retry Smaller After Memory Error** is enabled by
default: retry once at half the selected side, bounded to 256-512 px. For example,
1024 retries at 512, while 512 retries at 256. It resamples the same Source and
model mask; the Source-resolution output and post-generation mask controls remain
independent. Status reports the actual working dimensions and the memory retry.

Asynchronous weight prefetch is disabled to reduce concurrent RAM/device commit.
If the retry also fails, Status suggests 512/256 and points to the complete worker
log instead of dumping the tensor-loading progress. Closing unused memory-heavy
apps can help; changing result Grow / Erode or Blur cannot fix this inference error.

## Qualification

The following earlier generation runs used the worker mask path and matching
available at capture time. Current Nuke nodes use the separated native mask path
described above.

On the RTX 4070 Laptop 8 GB / 32 GB RAM, six tasks on one street frame with two
seeds each completed for both Mage and FLUX.2 Klein Q4 + LanPaint. At a maximum
512-pixel crop side Mage's median was 12.35 s and FLUX's 30.96 s; both peaked at
5534 MiB total device use, including background applications. These measure
different complete inference backends, not equalized model kernels. They exclude
the new boundary pass. Repaired results require shot-specific visual review.

A full original 2048x1080 street frame and its original person mask were also
tested at a 2048x1088 padded diffusion canvas, seed 42, four steps. Passing the
full-size reference directly failed before sampling with `Reference image
exceeds the tokenized context` after 128.78 s. Bounding the reference to 1024
completed in 41.36 s, peaking at 7457 MiB total device use out of 8188 MiB.
The production service with mask overlap, color matching, Auto boundary repair
and inward softness completed in 51.69 s (inference 41.45 s), peak 7375 MiB,
with OOM retry disabled. Background use varied between runs. This qualifies
this frame/aspect ratio, not every possible 2048-square canvas or concurrent
GPU workload. Architectural lines and source shadows can still reveal edits.
The same production request with Neural boundary repair completed in 95.34 s,
peak 7374 MiB. Both results preserve every source byte outside the original
mask. On this shot Neural softened pavement detail without clearly improving
the result. These were experimental boundary tests; the final node instead uses
native mask compositing and Lab Color Match with both boundary passes disabled.

Fresh installation was exercised in a separate path containing spaces and
Cyrillic. Both environments passed `pip check` without PyTorch installed. The
existing SHA-verified weight bundle was reused rather than downloaded again.
A real masked Mage generation then completed through the fresh installation.
Nuke 16 qualification also exercised source/mask export, authenticated HTTP
generation, patch loading, script save/reopen and a PNG render. The terminal
test inlines Nuke's background dispatcher; it does not automate the desktop.
The final Russian-prompt Nuke run completed in 25.1 s including translation,
updated the visible prompt to English, kept its original text for Restore
Original, and reopened with Mage and Auto Color Match selected. Ordinary alpha
feathering was used, without experimental boundary passes.

## Licenses

Mage and its Q6 conversion are MIT, the Qwen3-VL encoder is Apache-2.0, and
stable-diffusion.cpp / llama.cpp are MIT. NVIDIA CUDA redistributables retain
NVIDIA's own terms. Model Manager downloads these optional artifacts; they are
not embedded in the repository. Preserve license notices when redistributing.
The Mage worker does not import LanPaint or the GPL FLUX worker.

Native cached-patch qualification on the 1024 x 540 plate rendered four looks
in 0.22-0.25 s each including PNG output, with no server calls or cache writes.
Real Nuke integration checks also verify signed grow/erode, exact agreement with
ordinary Gaussian Blur, unchanged exported model pixels and payloads, crop coverage
clipping, Source precision/alpha, missing partners and save/reopen.
