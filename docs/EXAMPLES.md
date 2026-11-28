# Real image examples

[Project README](../README.md#real-examples) · [Visual node guide](NODES.md) ·
[Documentation home](README.md)

The README comparisons were captured on 2026-11-28 using the project's existing
local street qualification photograph, resized once to **1024 × 540**. The
[source PNG](images/examples/source.png) is the exact input used by these runs.
The source is already sRGB. No generated background, matte or detail was retouched
for the documentation. Layouts add labels and resize previews; the linked raw
results retain their capture dimensions.

Segment, Refine, Inpaint, Generative Fill and DLSS use the actual Kyven engine
services and installed providers. These five examples exercise inference directly,
without automating the Nuke properties UI or its HTTP adapter. WAFT uses the
deployed gizmo and native Nuke Inference in **NukeX 16.0v4**. Hardware was an
**RTX 4070 Laptop 8 GB / 32 GB RAM**.

[Recorded requests, model metadata and image hashes](images/examples/capture.json)
make the capture inspectable. Timings in that file include cache reuse where
`cache_hit` is true; they are not a comparative performance benchmark. The images
demonstrate these particular runs, not universal image quality or temporal stability.

## Segment

**SAM 2.1 Small**, balanced profile, full-frame processing, Fill Holes off. Five
positive points select the whole person, using top-left image coordinates:
`(595,200)`, `(605,325)`, `(605,495)`, `(526,395)`, `(686,391)`. The first README
panel overlays these exact points; the second is the model's alpha. The third
multiplies Source by that alpha over a checkerboard to show the cutout.

[Source](images/examples/source.png) · [Alpha](images/examples/segment-alpha.png) ·
[Edge Guide](images/examples/segment-edge-guide.png) · [Cutout preview](images/examples/segment-cutout.png)

## Refine

**ViTMatte Small Composition-1k** consumes the same Source and Segment alpha.
Generate Trimap is on, foreground erosion **5 px**, background dilation **8 px**,
tiles **512 px**, overlap **64 px**, known-tile skipping on. The README shows the
same crop `x=515..710, y=145..390` in Source, prepared trimap and refined alpha.
The raw refined matte contains **10,609 pixels with fractional alpha**.

[Source](images/examples/source.png) · [Coarse alpha](images/examples/segment-alpha.png) ·
[Actual trimap](images/examples/refine-trimap.png) · [Refined alpha](images/examples/refine-alpha.png)

## Inpaint

**Big-LaMa Native**, Standard, Auto crop with **128 px** context, model grow **0**,
edge softness **0**. The removal mask is the real Segment silhouette dilated by
**14 px** at the 1024 × 540 source resolution; this is an explicit mask-preparation
step. The model estimates the hidden storefront and pavement. Its real geometric
and texture artifacts remain visible in the published result.

The capture check verified byte-identical Source RGB outside the removal mask;
**65,717 selected pixels** changed.

[Source](images/examples/source.png) · [Removal mask](images/examples/removal-mask.png) ·
[Actual result](images/examples/inpaint-result.png)

## Generative Fill

**Mage-Flow Edit Turbo Q6**, seed **42**, **4 steps**, Processing Resolution **512**,
Auto crop with **256 px** context, model grow **0**, Generation Overlap **32 px**,
Result Mask Grow **0**, inward edge softness **4 px**, Auto Color Match. Seamless
Blend and experimental boundary repair are off, matching the current Mage node path.
It uses the same prepared removal mask as Inpaint.

Exact instruction:

```text
Remove the man. Restore the storefront and pavement behind him. Preserve the white car and the original lighting.
```

The actual generated storefront and pavement are shown without retouch. The
capture verified Source bytes outside the permission mask stayed identical;
**65,252 selected pixels** changed. This example does not establish consistency
between independently generated video frames.

[Source](images/examples/source.png) · [Permission mask](images/examples/removal-mask.png) ·
[Actual result](images/examples/generative-fill-result.png)

## DLSS

The input photograph is converted from sRGB to scene-linear RGB and written as a
32-bit float EXR with opaque alpha. **Kyven DLSS Runtime**, Temporal **off**,
Processing Size **1**, Intensity **1**, Preserve Source Color **on**, HDR Rolloff
**on**, no Motion/Depth/Mask guides. The output EXR is converted to sRGB for its
README preview; it is not an exposure-adjusted comparison.

Source and Result use the same detail crop `x=490..730, y=135..405`. Difference
is `clip(abs(result_sRGB8 - source_sRGB8) * 8, 0, 255)`. It visualizes where the
display pixels changed rather than claiming every change improves the image.
The float EXR stays in the local capture work directory.

[Source](images/examples/source.png) · [Actual result, sRGB preview](images/examples/dlss-result-srgb.png)

## WAFT SmartVector

**Deployed WAFT-a1 FP16 v5**, five iterations, Processing Resolution **512**,
Grid Suppression **1**, Edge Threshold **0.25**, Motion Vectors output. Native
Nuke Transform creates a controlled photographic pan: frame 2 has zero offset,
frame 3 has **+8 source pixels horizontally**. Adjacent frames are evaluated by
the real deployed WAFT Inference operators.

The measured interior median is **forward.u = 8.0145 px**,
**forward.v = 0.0010 px**, excluding a 32 px border. The flow preview maps
`atan2(v,u)` to HSV hue and `clip(magnitude / 16, 0, 1)` to brightness, with full
saturation. Its near-uniform color is the expected visualization of this pan.
This is real inference on photographic pixels with controlled motion; it is not
a recorded moving-person sequence and does not qualify occlusions or deformation.

[Frame 2](images/examples/waft-frame-2.png) · [Frame 3](images/examples/waft-frame-3.png) ·
[Actual forward-flow visualization](images/examples/waft-forward-flow.png) ·
[Float forward vectors, NPZ](images/examples/waft-forward-vectors.npz)

## Reproduce the capture

Requires the existing installed SAM, ViTMatte, Big-LaMa and Mage weights, Mage's
private runtime, prepared DLSS runtime and the complete deployed WAFT folder.
The script does not download missing models automatically. Run from the checkout:

```powershell
.venv/Scripts/python.exe scripts/capture_node_examples.py
& 'C:\Program Files\Nuke16.0v4\Nuke16.0.exe' --nukex -i -t --gpu scripts/capture_waft_example.py
.venv/Scripts/python.exe scripts/build_node_examples.py
```

The first command defaults to the checked-in source. `--source path/to/plate.png`
can select the local original during capture; the point coordinates are specific
to this street frame. Use a separate Nuke process so the artist's open script is
not touched. Inference and native EXR work files remain under
`.runtime/documentation-real/`. The final command copies selected results into
`docs/images/examples/` and rebuilds the six Markdown preview PNGs under
`docs/images/nodes/`. All README image references are relative repository paths
and work on GitHub without a local server.
