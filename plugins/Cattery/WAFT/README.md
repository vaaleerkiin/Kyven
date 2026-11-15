# WAFT Cattery for Nuke

Native WAFT-a1 with five iterations, packaged as `.cat` models and tested in
NukeX 16.0v4. The graph uses Nuke's built-in Inference and VectorToMotion nodes.
No external Python, PyTorch installation, compiler, Kyven Server or Model Manager
is needed to use the plugin.

Package version 4 includes adjustable vector grid suppression, GPU FP16 by
default, SDPA attention and corrected output bounds after VectorToMotion.
Output resolution and bounds follow the connected source.

## Install

GitHub **Code > Download ZIP** includes both model binaries. Extract it and follow
the setup below; Git and Git LFS are unnecessary for ZIP installation.

In a Kyven checkout, both models are versioned through Git LFS. Install Git LFS
before cloning; for an existing checkout run `git lfs install` and `git lfs pull`
from the repository root. Connect Kyven through your `.nuke/init.py`:

```python
import nuke
nuke.pluginAddPath('D:/Kyven/hosts/nuke')
```

Replace the path with your installation, restart Nuke and choose
**Kyven > WAFT SmartVector**, or Tab > WAFT.

For standalone use, copy this complete `WAFT` folder into `.nuke/Cattery/`.
The final path must be `.nuke/Cattery/WAFT/cat.json`. Restart Nuke and choose
**Cattery > Optical Flow > WAFT**. Keep one active WAFT installation on the plugin
path. See [INSTALL.md](INSTALL.md).

## Use

WAFT uses the same Kyven logo and header style as the other Kyven tools, with
native sections for Model and Performance, Input, Output, and Vector Cleanup.
All branding assets are included in this portable folder.

Connect an image sequence to the single input. Set **Input Colorspace** to the
source colorspace; the internal graph converts frames to sRGB for inference.
Adjacent source frames must be available. At sequence boundaries, neighboring
frame selection follows the Read node's range settings.

**Processing Resolution** limits the longest working edge to **512 (Low VRAM)**
(default), 768, 1024, or Full Source. Smaller frames are not enlarged. The vector
field is restored to the source dimensions, with displacement scaled independently
in X and Y. Project format does not determine output dimensions. Lower working
resolution uses less memory but can lose fine details and small motion.

**Precision** selects one of two embedded models:

- **GPU FP16 (Fast)**: `WAFT.cat`. Weights, convolutions and attention use FP16;
  flow accumulation, warping coordinates and output vectors remain FP32. Attention
  uses SDPA and full-resolution upsampling occurs after the final iteration.
- **FP32 (Compatible)**: `WAFT_FP32.cat`, for comparison and CPU use. Turning off
  **Use GPU if available** automatically selects FP32.

Only the selected model is used during normal operation. Nuke Inference's native
`halfPrecision` switch stays off because reduced precision is built into the
FP16 model. FP16 and FP32 can produce small numerical differences.

**Grid Suppression** ranges from 0 to 1, with default 1. Each vector pair is
bilaterally filtered before resizing to reduce fine spatial ripple while limiting
mixing across motion boundaries. Zero bypasses filtering. This is vector
postprocessing, not a change to model weights; small real movements can be smoothed.

**Edge Threshold** defaults to 0.25 working-resolution pixels. Lower values preserve
smaller motion boundaries; higher values suppress stronger ripple. Source RGB is
not used to weight this filter.

On frame 77 of the original test clip, high-frequency residual RMS in one wall
region fell by about 52%; motion-mask IoU at a one-pixel threshold was 0.999987.
This describes one frame and region, not a general guarantee. A synthetic vector
step from 3 to 8 pixels was preserved with less than 0.01-pixel error.

## Outputs and caching

| Output Mode | Channels | Use |
| --- | --- | --- |
| Motion Vectors | `forward.u`, `forward.v`, `backward.u`, `backward.v` | Native motion consumers such as VectorBlur, through VectorToMotion |
| Smart Vectors | `smartvector_f01_v01.n_u`, `.n_v`, `.p_u`, `.p_v` | Smart-vector workflows such as VectorDistort |

Only vector channels are output. To retain source RGB, copy the vector layers into
the source branch with Copy/MergeCopy. Render to EXR with **channels = all**.
WAFT uses ordinary Nuke evaluation and Write rendering, independently of Kyven's
server cache controls.

## Memory and updates

GPU FP16 with Processing Resolution 512 is the recommended starting point.
Previous PyTorch tests on an RTX 4070 Laptop at 512 x 288 measured about 296 MiB
peak allocated tensors for FP16 and 491 MiB for FP32. These figures cover one
frame pair, not Nuke's total VRAM. Nuke also needs both flow directions, image
buffers and application memory. Reduce working resolution after a memory error;
CPU FP32 is available but slower.

After updating the package, restart Nuke and create a new WAFT node. Existing
nodes can retain their previous internal graph.

## Package and development

Keep `WAFT.cat`, `WAFT_FP32.cat`, `WAFT.gizmo`, `cattery_waft.py`, `cat.json`,
icons, `LICENSE-WAFT.txt` and the validation reports together. This folder can
be copied into another Cattery installation without splitting its contents.

In Kyven, development sources and recovery instructions are under
`development/waft/README.md`; the main user guide is `docs/WAFT.md`.
Development environments, checkpoint copies, intermediate exports and renders
are ignored; the two deployed models are versioned through Git LFS.

Source: [Princeton WAFT](https://github.com/princeton-vl/WAFT), waftv2 branch,
a1 architecture. The authors' recommended checkpoint has Google Drive ID
`1CxzBQx0iSg6AyIgt6MF0ROlF_cAeZLPC`. Weight loading is strict. The export supports
dynamic frame dimensions with five fixed iterations. Source provenance and the
checkpoint SHA-256 are retained in the development workspace.
