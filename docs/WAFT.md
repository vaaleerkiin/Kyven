# WAFT SmartVector

[Documentation home](README.md) · [Install](INSTALLATION.md) ·
[Nuke workflow](NUKE.md) · [Development recovery](../development/waft/README.md)

WAFT-a1 estimates optical flow between adjacent frames and outputs either Motion
Vectors or Smart Vectors. It is a self-contained Cattery plugin bundled alongside
Kyven: calculations use Nuke's native Inference nodes, with no Kyven Server or
Model Manager download. Package version 4 uses five model iterations.

## First use

1. Use Nuke 16; this package was tested in NukeX 16.0v4.
2. Check `plugins/Cattery/WAFT/` contains both `WAFT.cat` and `WAFT_FP32.cat`,
   `WAFT.gizmo`, `cattery_waft.py`, `cat.json`, icons and `LICENSE-WAFT.txt`.
   Both models are included in the repository through Git LFS. Install Git LFS
   before cloning; for an existing checkout run `git lfs install` and `git lfs pull`
   in the repository root. The files must be binary models (about 124 and 244 MiB),
   not small LFS pointer text. You can also [rebuild from source](../development/waft/README.md).
3. Connect Kyven once through your existing `.nuke/init.py`:

   ```python
   import nuke
   nuke.pluginAddPath('D:/Kyven/hosts/nuke')
   ```

   Replace `D:/Kyven` with your installation path, then restart Nuke.
4. Choose **Nodes > Kyven > WAFT SmartVector**, or Tab > WAFT. Connect the source
   image sequence to its single input. Adjacent source frames must be available.
5. Set **Input Colorspace** to the source image's actual colorspace. The internal
   graph converts it to sRGB for inference.
6. Start with **Processing Resolution = 512 (Low VRAM)** and
   **Precision = GPU FP16 (Fast)**. Choose the required **Output Mode** below.
7. Render with an ordinary Nuke Write node: use EXR and **channels = all**.
   Read the result back and check that the vector channels are present.

For WAFT alone, no external Python environment, pip packages, compiler or Kyven
server installation is required. Core Kyven nodes still use their standard setup.

## Controls

| Control | Behavior |
| --- | --- |
| Processing Resolution | Limits the longest working edge to 512, 768, 1024, or Full Source. Smaller images are not enlarged. |
| Precision | GPU FP16 selects `WAFT.cat`; FP32 selects `WAFT_FP32.cat`. |
| Use GPU if available | Turning this off selects the FP32 model for CPU rendering. |
| Grid Suppression | 0 returns raw flow; 1 is the default vector smoothing strength. Smoothing can remove small real movements. |
| Edge Threshold | Default 0.25 working pixels; lower values preserve smaller motion boundaries, higher values smooth across larger differences. |
| Output Mode | Select Motion Vectors or Smart Vectors. |

The output format and bounds follow the connected source, including when the
project format differs. Vector lengths are rescaled independently in X and Y after
reduced-resolution processing. Start at 512; increase only when detail and available
memory justify it. These controls do not change the trained model weights.

## Outputs

| Mode | Channels | Example use |
| --- | --- | --- |
| Motion Vectors | `forward.u`, `forward.v`, `backward.u`, `backward.v` | Native motion-vector consumers such as VectorBlur |
| Smart Vectors | `smartvector_f01_v01.n_u`, `.n_v`, `.p_u`, `.p_v` | Smart-vector workflows such as VectorDistort |

Only vector channels are output. To keep source RGB, copy the vector layers into
the source branch using Copy/MergeCopy. Write all channels to EXR to preserve them.
At sequence boundaries, adjacent-frame behavior follows the Read node's range
settings. WAFT uses ordinary Nuke evaluation and Write rendering; Kyven's server
Process/Cache buttons and Folder/Name/Version cache contract do not apply.

## Hardware and limits

GPU FP16 is the recommended starting mode. FP32 provides a CPU option but can be
slow. Full Source needs more memory and can lose its practical advantage on large
frames. Lower Processing Resolution after an out-of-memory error.

Prior PyTorch measurements on an RTX 4070 Laptop at 512 x 288 recorded about
296 MiB peak allocated tensors for FP16 and 491 MiB for FP32. Those measurements
cover one frame pair, not Nuke's total VRAM usage. Nuke needs both flow directions,
image buffers and its own application memory. The core Kyven 4 GB support target
and server GPU queue do not guarantee WAFT's runtime memory use.

Motion estimation can lose fine detail and small motion at reduced resolution.
Inspect vector quality on representative shot frames; grid suppression is
postprocessing and can smooth real changes. FP16 and FP32 can differ numerically.

## Portable folder, updates and development

To use the plugin independently, copy the complete `plugins/Cattery/WAFT/` folder
into `%USERPROFILE%/.nuke/Cattery/`. The final path must be
`.nuke/Cattery/WAFT/cat.json`, with both models beside it. Restart Nuke and look
under **Cattery > Optical Flow > WAFT**. Keep one active installation to avoid
loading an old gizmo or callback module from another plugin path.

After replacing the package, restart Nuke and create a new WAFT node. Existing
nodes can retain their previous internal graph. For deployment, include the entire
plugin folder, both model files, icons and license; `development/` is unnecessary.
Never distribute only the gizmo or expect the server installer to restore its models.

[Development recovery](../development/waft/README.md) describes how to recreate
Python 3.10 dependencies, download the checksum-verified checkpoint, export both
models, test the graph and build the standalone archive. Sources are retained in
Git. Both deployed `.cat` files are versioned with Git LFS; regenerated environments,
checkpoints, exports, archives and development model copies remain ignored.
See also [ADR 0008](adr/0008-standalone-waft.md).
