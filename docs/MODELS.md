# Kyven Tools model selection and safety

[Documentation home](README.md) · [Install](INSTALLATION.md) · [Segment](SEGMENT.md) ·
[Refine](REFINE.md) · [Inpaint](INPAINT.md) · [DLSS](DLSS.md)

Kyven's server-backed tools treat model implementations and model sizes as selectable providers. A server-backed node stores a
stable `model_id`, not a hard-coded Python class or checkpoint path.

## Embedded WAFT models

Standalone [WAFT SmartVector](WAFT.md) uses files in `plugins/Cattery/WAFT/`,
outside the server model catalog: `WAFT.cat` (GPU FP16, about 124 MiB) and
`WAFT_FP32.cat` (FP32/CPU, about 244 MiB). Both are embedded in the complete local
plugin package and must travel with its gizmo. Model Manager does not install or
remove them. Both models are versioned using Git LFS and arrive with a normal
LFS-enabled clone. For an existing checkout run `git lfs pull`; see [WAFT](WAFT.md).

Use Processing Resolution 512 as the starting setting. Lower working resolution
reduces memory needs; CPU selects FP32 and is slower. Standalone WAFT has its own
Nuke memory usage and does not inherit the server's GPU queue or 4 GB guarantee.
The preserved source/model license is BSD-3-Clause; see the plugin's `LICENSE-WAFT.txt`.

## Segment catalog

| Model ID | Display name | Parameters | Download | VRAM guidance |
| --- | --- | ---: | ---: | ---: |
| `sam2.1-tiny` | SAM 2.1 Tiny | 38.9 M | 156 MB | 4 GB |
| `sam2.1-small` | SAM 2.1 Small | 46.0 M | 184 MB | 6 GB |
| `sam2.1-base-plus` | SAM 2.1 Base+ | 80.8 M | 324 MB | 8 GB |
| `sam2.1-large` | SAM 2.1 Large | 224.4 M | 898 MB | 12 GB+ |
| `hq-sam2.1-large` | HQ-SAM 2.1 Large (Beta) | 224.7 M | 899 MB | 12 GB+ |

The VRAM column is conservative Kyven guidance, not an upstream guarantee. A warning does not
block selection: the artist may try any installed model and later use the Model Manager's
fallback behavior. On the tested RTX 4070 Laptop GPU with 8,188 MiB, Small is the default.

Only one segmentation model remains resident. Selecting another variant unloads the previous
one before loading the requested model.

HQ-SAM 2.1 Large is the official SysCV beta checkpoint for higher-quality boundaries in images and
video propagation. It uses the HQ-SAM 2 runtime and is optional; Tiny and Small remain the portable
choices. The checkpoint and runtime are Apache-2.0, pinned to official upstream revisions, and CPU
inference is supported but impractically slow for most production shots.
When installed from Nuke, Model Manager checks the private Kyven environment, installs the pinned
HQ-SAM 2 runtime when needed, restarts only Kyven Server, and then downloads and verifies the
checkpoint. No separate installer step is required.

## Commands

The legacy MOSS catalog entry remains available for managing existing installations.
Its Edit weights have unconfirmed commercial permission and it is retired from the node selector.

```text
kyven models list --models-dir models
kyven models download sam2.1-small --models-dir models
kyven models download hq-sam2.1-large --models-dir models
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
| `mematte-vits-aim500` | MEMatte ViT-S AIM-500 | 29.2 M | 111 MB | 4 GB+ |
| `mematte-vitb-composition-1k` | MEMatte ViT-B Composition-1k | 110.1 M | 420 MB | 8 GB+ |

ViTMatte officially provides Small and Base backbones; it does not publish a Medium variant. Kyven
keeps Small Composition-1k as the portable default and offers Base Distinctions-646 as the heavier
quality option. Both checkpoints are pinned to official Hugging Face revisions and verified with
SHA-256 before they are moved into `models`. Their model cards are Apache-2.0; the original
ViTMatte implementation is MIT.
Refine unloads a resident SAM model before loading ViTMatte, which keeps the 4–8 GB workflow viable.

MEMatte is the high-resolution alternative. Its adaptive token router sends only important tokens
through global attention. Kyven maps Low Memory, Balanced, and Quality profiles to progressively
larger token limits while retaining the same RGB + trimap contract, output modes, cache, and tiling
controls as ViTMatte. ViT-S AIM-500 is the lighter real-world option; ViT-B Composition-1k is the
heavier quality option.

```text
kyven models download vitmatte-small-composition-1k --models-dir models
kyven models download vitmatte-base-distinctions-646 --models-dir models
kyven models download mematte-vits-aim500 --models-dir models
kyven models download mematte-vitb-composition-1k --models-dir models
```

In Nuke, Model Manager has **Set as Default**. It stores one default each for Segment, Refine,
Inpaint and Generative Fill. The selection is applied only when a new node is created; existing scripts and nodes keep
their explicit model selection. DLSS uses its own managed runtime rather than one of these selectable default models; see [DLSS setup](DLSS.md#setup).

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

## Generative Fill catalog

| Model ID | Display name | Starting Nuke settings | Setup |
| --- | --- | --- | --- |
| `mage-flow-edit-turbo-q6` | Mage-Flow Edit Turbo Q6 | Factory default, 512 px / 4 steps | Private native worker and approximately 7.69 GB verified bundle |
| `flux2-klein-4b-q4-lanpaint` | FLUX.2 Klein 4B Q4 + LanPaint | 512 px / 4 steps | Separate Torch/LanPaint worker and approximately 10.83 GB bundle |

FLUX Base is retired from the normal catalog and new node selector. Its pinned
entry/provider remain available for existing scripts; no weight or render is
deleted. New nodes offer Mage and distilled FLUX only.

These are optional heavier profiles. Recorded GPU qualification used an 8 GB RTX
4070 Laptop and 32 GB RAM; it does not establish a 4 GB guarantee. CPU inference
is selectable. Mage supports a working side up to 2048, FLUX up to 1024, subject
to available memory. **Install / Repair Model** prepares the selected model and
its dependencies; **Install Required Components** skips weights. Mage-only setup
does not require Torch, SAM or FLUX. Each backend uses a separate private runtime.

Mage/native workers use MIT components and an Apache-2.0 encoder; CUDA artifacts
retain NVIDIA's terms. FLUX weights are Apache-2.0 and its LanPaint worker is
GPL-3.0. The retired MOSS Edit model has unconfirmed commercial terms and is not
in the normal node selector. See [Generative Fill](GENERATIVE_FILL.md),
[Mage qualification](MAGE_FILL.md#qualification) and
[third-party notices](../THIRD_PARTY_NOTICES.md).
