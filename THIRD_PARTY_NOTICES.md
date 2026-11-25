# Third-party notices

Kyven's server model weights are downloaded separately. The standalone WAFT
Cattery models are included in the repository through Git LFS.

The standalone WAFT bundle under `plugins/Cattery/WAFT/` includes FP16
and FP32 model binaries versioned through Git LFS. Its Princeton Vision & Learning Lab
BSD-3-Clause license is preserved in `LICENSE-WAFT.txt` within that folder.
Development sources, upstream notices and recovery instructions are preserved
under `development/waft/`. See [standalone plugin packaging](plugins/README.md).

Built-in provider metadata currently references:

| Component | Purpose | License | Source |
| --- | --- | --- | --- |
| SAM 2.1 code and official checkpoints | Promptable segmentation | Apache-2.0 | https://github.com/facebookresearch/sam2 |
| HQ-SAM 2 code and HQ-SAM 2.1 Large checkpoint | High-quality image and video segmentation | Apache-2.0 | https://github.com/SysCV/sam-hq/tree/main/sam-hq2 |
| ViTMatte original code | Alpha matting architecture | MIT | https://github.com/hustvl/ViTMatte |
| ViTMatte Small Composition-1k checkpoint | Alpha refinement | Apache-2.0 | https://huggingface.co/hustvl/vitmatte-small-composition-1k |
| ViTMatte Base Distinctions-646 checkpoint | Alpha refinement | Apache-2.0 | https://huggingface.co/hustvl/vitmatte-base-distinctions-646 |
| MEMatte inference code and ViT-S AIM-500 / ViT-B Composition-1k checkpoints | Memory-efficient alpha refinement | MIT | https://github.com/linyiheng123/MEMatte |
| OpenCV LaMa ONNX checkpoint | Object removal / inpainting | Apache-2.0 | https://huggingface.co/opencv/inpainting_lama |
| Big-LaMa architecture and TorchScript checkpoint | Native-resolution object removal | Apache-2.0 | https://github.com/advimman/lama |
| ONNX Runtime | LaMa inference runtime | MIT | https://github.com/microsoft/onnxruntime |
| Hugging Face Transformers | ViTMatte runtime implementation | Apache-2.0 | https://github.com/huggingface/transformers |
| Safetensors | Verified model weight loading | Apache-2.0 | https://github.com/huggingface/safetensors |
| PyTorch | Optional inference runtime | BSD-style | https://github.com/pytorch/pytorch |
| Pillow | Image I/O | HPND | https://github.com/python-pillow/Pillow |
| NumPy | Array processing | BSD-3-Clause | https://github.com/numpy/numpy |
| OpenEXR Python module (optional) | Scene-linear DLSS EXR transport | BSD-3-Clause | https://github.com/AcademySoftwareFoundation/openexr |

Optional DLSS transport targets the protocol of
[KJzzzKJ/DLSS5-for-Nuke](https://github.com/KJzzzKJ/DLSS5-for-Nuke) at commit
`d61307ef4543c8991668730a4cab8019b3cb0096`. Its project-authored worker/shim source is MIT-licensed
(copyright 2026 Kai(KJ)). The build helper preserves its LICENSE with locally built binaries.
The maintained worker derivative under `native/dlss` preserves the upstream MIT license and attribution.
Kyven builds and packages `KyvenDLSSWorker.exe` and its caller shim with a SHA-256 manifest.
The owner-requested preview distribution includes the pinned NVIDIA Neural Rendering component
`nvngx_dlssnr.dll`. Its licensing is separate; Kyven's Apache-2.0 license does not relicense it.

The optional Model Manager DLSS install can download the pinned community release
`KJzzzKJ/DLSS5-for-Nuke` v1.0.0 and extract only `runtime/nvngx_dlssnr.dll`, after presenting its
source and terms. Exact archive/member checksums are recorded in `resources/dlss/vendor-source.json`.
The archive contains this binary although the current upstream README describes a narrower runtime
policy. Permission for Kyven to redistribute that NVIDIA binary has not been verified; the file is
included in the owner-requested preview ZIP/wheel, with its source, exact size and SHA-256 recorded.
The package does not install the upstream worker, ReShade or RenoDX.

## Optional Generative Fill

FLUX.2 Klein 4B and its VAE, plus the shared Qwen3-4B checkpoint distributed
by Tongyi-MAI, are Apache-2.0. Their selected files
and the Unsloth Q4_K_M derivative are pinned by revision, size and SHA-256 in
`src/kyven/resources/generative_fill/models.json`; weights are not committed.

LanPaint at `2d7912f9a5efe5ece8de334c7ca18317b8288c39` and LanPaint-Diffusers
at `a2f6483acb678ecb0b6268a311d55d6d51f37c4f` are GPL-3.0:
[LanPaint](https://github.com/scraed/LanPaint),
[LanPaint-Diffusers](https://github.com/charrywhite/LanPaint-diffusers).
The optional worker `src/kyven/resources/generative_fill/worker.py` is explicitly
GPL-3.0-only and its complete source and license ship together. This exception
does not relicense the Apache-2.0 transport or Nuke adapter. Redistribution of
the worker/dependencies must include their required licenses and corresponding
source. Diffusers, Transformers and Accelerate use Apache-2.0; GGUF uses MIT.

Every future provider must record separate code and model-weight licensing metadata before it
can be enabled in the default distribution.

Generative Fill seamless compositing uses opencv-python-headless 4.14.0.94 in its
private worker. Preserve the OpenCV package license and bundled third-party
notices when redistributing the runtime.

### Experimental MOSS backend

MOSS-Image-SFT from OpenMOSS and its published quantized conversions are pinned
in `src/kyven/resources/generative_fill/moss_models.json`. The base is Apache-2.0.
The separately downloaded `chinoll/Moss-Image-Edit` adapter does not state an
explicit weights license; commercial use and redistribution are not confirmed.
Its author-provided editing code is GPL-3.0 and its pinned complete inference
sources and license are downloaded with the adapter. Kyven's separate MOSS worker
and tensor-format loader are GPL-3.0-only, beside the worker license.

The standalone `comfy-kitchen` 0.2.41 tensor-kernel library is Apache-2.0:
https://github.com/Comfy-Org/comfy-kitchen. This backend does not require the
ComfyUI application, graph execution or server. Preserve all dependency notices
when distributing a prepared runtime. Experimental model metadata must not be
interpreted as a commercial license grant for the Edit adapter.
