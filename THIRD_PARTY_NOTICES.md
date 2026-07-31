# Third-party notices

Kyven does not commit or redistribute model weights in this repository.

Built-in provider metadata currently references:

| Component | Purpose | License | Source |
| --- | --- | --- | --- |
| SAM 2.1 code and official checkpoints | Promptable segmentation | Apache-2.0 | https://github.com/facebookresearch/sam2 |
| ViTMatte original code | Alpha matting architecture | MIT | https://github.com/hustvl/ViTMatte |
| ViTMatte Small Composition-1k checkpoint | Alpha refinement | Apache-2.0 | https://huggingface.co/hustvl/vitmatte-small-composition-1k |
| OpenCV LaMa ONNX checkpoint | Object removal / inpainting | Apache-2.0 | https://huggingface.co/opencv/inpainting_lama |
| ONNX Runtime | LaMa inference runtime | MIT | https://github.com/microsoft/onnxruntime |
| Hugging Face Transformers | ViTMatte runtime implementation | Apache-2.0 | https://github.com/huggingface/transformers |
| Safetensors | Verified model weight loading | Apache-2.0 | https://github.com/huggingface/safetensors |
| PyTorch | Optional inference runtime | BSD-style | https://github.com/pytorch/pytorch |
| Pillow | Image I/O | HPND | https://github.com/python-pillow/Pillow |
| NumPy | Array processing | BSD-3-Clause | https://github.com/numpy/numpy |

Every future provider must record separate code and model-weight licensing metadata before it
can be enabled in the default distribution.
