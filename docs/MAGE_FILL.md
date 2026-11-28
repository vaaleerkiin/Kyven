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

## Masks and boundaries

Mage receives the source image as both the initial image and an editing
reference. The inference mask constrains latent predictions during every sampling
evaluation. Final source compositing independently enforces the original artist
alpha, including fractional alpha and unchanged float Source pixels outside it.
Inference masks are binary; output alpha remains independent.

Mage uses ordinary inward alpha feathering with **Result Edge Softness**, default
4 source pixels. The node does not run experimental boundary reconstruction or
a second neural pass. Legacy experimental knobs are retained invisibly so saved
scripts load, but Mage requests from the node force those passes off.

**Color Matching** and **Color Match Strength** are on the main tab:

- **Auto** calibrates generated RGB to clean Source context. Explicit removal
  prompts also match continuous, locally flat background surfaces.
- **Context** calibrates exposure/color while preserving new subject colors.
- **Background** forces the removal-style background matching.
- **Off** disables color correction. Strength 0 also disables it.

Paired, held-out-validated tone curves correct nonlinear gamma drift when they
outperform an affine exposure/white-balance fit. Removal matching also estimates
each flat boundary side independently and carries a smooth RGB offset inward.
It does not warp or blur generated image detail. **Color Match Radius** controls
that local transition (0 disables it). If the context is insufficient or
uncorrelated, the node reports that Color Match was skipped.

### Tight masks and visible outlines

**Smooth Removal Edges** prepares Auto crop, Context Padding 384, Processing
Resolution 1024, Result Mask Grow 32, Result Edge Softness 24, Generation Overlap
0 and Auto Color Match. Click Generate again to apply it. This explicitly lets
the result change RGB in an additional 32-source-pixel band around the input
alpha. Outside that expanded permission band Source stays unchanged; Source
alpha and other channels are preserved. Keeping model-only overlap at zero
aligns the sampling constraint with the expanded blend boundary.

New nodes still default to **Result Mask Grow 0**, preserving exact artist alpha.
Positive grow uses the dilated nonzero mask support as the result permission.
The result/patch mask output includes the expanded feathered area. Preview Model
Mask shows the combined result permission, Model Grow and Generation Overlap.
Changing edge/color settings requires Generate again; the node says so.

Simply feathering a tight subject matte inward can restore the old subject's
edge from Source. In the measured street test, inward softness 48 produced a
visible person ghost. Expanded margin 32 with softness 24 moved the transition
onto surrounding background and avoided that outline. A real Nuke run of the
preset at 1024 completed in 59.6 s including Russian translation. More context
alone, a LaMa boundary pass and multiband blending were also tested; the latter
two were not adopted because of blur/ghosting in this shot. Generated structural
mismatches can remain; the preset is not a universal geometry reconstruction fix.

Color estimation excludes the masked source subject and its expanded fringe.
Color Match cannot repair incorrect geometry or reconstruct a hidden shadow.
For structural mismatches, try another seed, more context, or a larger processing
resolution. An undersized mask cannot remove source pixels outside its alpha.

The default processing side is 512. This limits the crop, not the output frame.
Mage's Processing Resolution range now reaches 2048; FLUX retains 1024.
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

## Qualification

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
standard alpha feathering and Color Match with both boundary passes disabled.

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
