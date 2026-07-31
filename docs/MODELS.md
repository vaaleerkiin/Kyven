# Kyven Tools model selection and safety

Kyven treats model implementations and model sizes as selectable providers. A node stores a
stable `model_id`, not a hard-coded Python class or checkpoint path.

## Segment catalog

| Model ID | Display name | Parameters | Download | VRAM guidance |
| --- | --- | ---: | ---: | ---: |
| `sam2.1-tiny` | SAM 2.1 Tiny | 38.9 M | 156 MB | 4 GB |
| `sam2.1-small` | SAM 2.1 Small | 46.0 M | 184 MB | 6 GB |
| `sam2.1-base-plus` | SAM 2.1 Base+ | 80.8 M | 324 MB | 8 GB |
| `sam2.1-large` | SAM 2.1 Large | 224.4 M | 898 MB | 12 GB+ |

The VRAM column is conservative Kyven guidance, not an upstream guarantee. A warning does not
block selection: the artist may try any installed model and later use the Resource Manager's
fallback behavior. On the tested RTX 4070 Laptop GPU with 8,188 MiB, Small is the default.

Only one segmentation model remains resident. Selecting another variant unloads the previous
one before loading the requested model.

## Commands

```text
kyven models list --models-dir models
kyven models download sam2.1-small --models-dir models
```

Downloads are written through a temporary file, checked against trusted byte size and SHA-256,
then moved atomically into the model directory. Every entry records its source and license.

## Refine catalog

| Model ID | Display name | Parameters | Download | VRAM guidance |
| --- | --- | ---: | ---: | ---: |
| `vitmatte-small-composition-1k` | ViTMatte Small | 25.8 M | 103 MB | 4 GB |

The checkpoint is pinned to an official Hugging Face revision and verified with SHA-256 before it
is moved into `models`. Its model card is Apache-2.0; the original ViTMatte implementation is MIT.
Refine unloads a resident SAM model before loading ViTMatte, which keeps the 4–8 GB workflow viable.

```text
kyven models download vitmatte-small-composition-1k --models-dir models
```

## Inpaint catalog

| Model ID | Display name | Download | Hardware guidance |
| --- | --- | ---: | --- |
| `lama-2025jan-onnx` | LaMa ONNX Fast | 93 MB | CPU; fixed 512 input; fastest Live mode |
| `big-lama-native` | Big-LaMa Native | 196 MB | CPU or GPU; native ROI detail; 4 GB+ recommended |

LaMa ONNX is downloaded from a pinned OpenCV Hugging Face revision and runs through ONNX Runtime.
Big-LaMa Native is the resolution-robust TorchScript model derived from the original LaMa project;
unlike the fast export it does not shrink every ROI into a fixed 512 x 512 canvas. Both downloads
are pinned by exact byte size and SHA-256, use Apache-2.0 model code/weights, and permit commercial
use. PowerPaint is intentionally not included: its roughly 15 GB multi-file runtime conflicts with
Kyven's portable, efficient installation target.

## Planned provider families

Depth is a roadmap item and is not yet part of the catalog or installer. Planned Depth
work currently favors commercially safe Small variants for interactive and temporal processing;
non-commercial Base/Large checkpoints must not appear as safe defaults.

See [Kyven Tools vision and roadmap](ROADMAP.md) for candidate models and proposed controls. Adding
a name to that roadmap does not make a checkpoint approved: catalog inclusion still requires a
pinned source, license metadata, byte size, SHA-256, commercial-use status, and hardware guidance.
