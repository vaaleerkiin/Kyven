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
block selection: the artist may try any installed model and later use the Model Manager's
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

Inside Nuke, use `Kyven > Model Manager...` or the **Model Manager...** button in any Kyven node.
It lists all catalog models with task, download size, installation state, and a VRAM warning. Model
downloads run outside the Nuke UI thread with a progress window and Cancel button. Removing a model
waits for active inference to finish, unloads it safely, and deletes only its verified checkpoint.

## Refine catalog

| Model ID | Display name | Parameters | Download | VRAM guidance |
| --- | --- | ---: | ---: | ---: |
| `vitmatte-small-composition-1k` | ViTMatte Small | 25.8 M | 103 MB | 4 GB |
| `vitmatte-base-distinctions-646` | ViTMatte Base | 96.7 M | 369 MB | 8 GB+ |

ViTMatte officially provides Small and Base backbones; it does not publish a Medium variant. Kyven
keeps Small Composition-1k as the portable default and offers Base Distinctions-646 as the heavier
quality option. Both checkpoints are pinned to official Hugging Face revisions and verified with
SHA-256 before they are moved into `models`. Their model cards are Apache-2.0; the original
ViTMatte implementation is MIT.
Refine unloads a resident SAM model before loading ViTMatte, which keeps the 4–8 GB workflow viable.

```text
kyven models download vitmatte-small-composition-1k --models-dir models
kyven models download vitmatte-base-distinctions-646 --models-dir models
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
use.

Big-LaMa's `Refined` quality mode uses the same verified checkpoint. It performs multi-scale feature
optimization and therefore changes runtime and memory use, not the catalog download. Standard is the
portable default; Refined is best used on a tight ROI with an 8 GB GPU or as a slow CPU fallback.
