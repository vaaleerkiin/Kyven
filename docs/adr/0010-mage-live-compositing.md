# ADR 0010: native Mage color and edge adjustments

Status: accepted for the owner-requested responsive Mage controls (2026-11-28).

## Context

Color Match and edge softness previously participated in the generation request.
An artist had to process again to see these changes. The owner requested responsive
Nuke controls, exploration of the ComfyUI DLSS5 Lab matcher, a resolution menu
through Mage's supported maximum, and removal of FLUX Base from normal selection.

## Decision

The optional Mage request sets `fill.native_postprocess=true`. The backend fits
Reinhard mean/std transfer in D65 Lab using paired clean context, excluding the
inference mask and an eight-pixel contamination fringe. Samples are bounded;
the NumPy implementation adds no model dependency and remains CPU-capable.

Alongside the base result, the backend atomically publishes private straight-RGBA
raw and fully matched patches. The paired sidecar is named
`<patch stem>.lab.png`. Both carry crop coverage in alpha and are required manifest
artifacts. Private RGB retains generated crop context, while public output remains
restricted to result permission. Legacy requests retain their existing cache
identity when native postprocessing is false.

Native Nuke Colorspace/MergeExpression operators blend luminance and chroma in
Lab. A separate post-generation block applies signed FilterErode to input alpha,
then ordinary Gaussian Blur and clips only to generated crop coverage. No automatic
erosion or multiplication by original permission follows the blur. Preview Model
Mask and the exported model-mask Write share the same binary input-alpha graph;
legacy growth, threshold and overlap controls are bypassed for native Mage. These controls submit no jobs and do not change the generation
payload or cached bytes. The Group preserves Source alpha/AOVs and exact float
Source outside the final blend alpha. Old RGB patches use the legacy graph until regenerated.
Frame-dependent file checks enable paired patches, including after cache reconnect
and script reopen. Missing partners fall back to the readable legacy branch.

`Create Base Result Read` reads the base cache; a Write connected to the Group
saves the selected live look. Artists must generate again after changing inference
inputs, prompt, seed, working resolution or crop. The resolution dropdown maps to
the existing numeric pixel field: Mage offers 256-preview through 2048, FLUX through
1024, and existing custom sizes remain representable.

FLUX Base is retired from normal catalog listing, offline Model Manager and new
nodes. Its trusted metadata and provider remain available for saved legacy nodes.
No model or render is deleted. A hidden backend flag replaces old numeric model
index assumptions in the graph. API 49 prevents a new adapter from sending the
new request contract to an older server.

## Evidence and limits

Native tests exercise pixel changes, separate L/chroma amounts, mask preview,
outside-mask float/alpha preservation, unchanged generation requests and files,
resolution mapping, legacy selection and save/reopen. A real RTX 4070 Laptop
512-side Mage request completed in about 20 seconds. Four cached Nuke looks on
the 1024 × 540 plate took 0.22–0.25 seconds each including full PNG output; this
is a single-machine observation, not a Viewer frame-rate guarantee.

The new pair adds disk/cache artifacts, not extra inference passes. Nuke holds
additional image buffers for Lab transforms. Matching cannot correct geometry or
an unmasked shadow; feathering tight removal masks can reveal the old silhouette.
The 2048 menu value is a model-side limit, not an 8 GB memory guarantee. Existing
OOM retry and CPU inference remain available. Core LaMa/SAM/Refine profiles are unchanged.

## References

- [DLSS5 Color Matcher source](https://github.com/Strix-3D/ComfyUI-DLSS5-Enhancer-v15/blob/main/nodes.py)
- [Official Mage native resolution range](https://github.com/microsoft/Mage/blob/main/mage_flow/README.md)
- [Mage artist guide](../MAGE_FILL.md)
