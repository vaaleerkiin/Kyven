# SPDX-License-Identifier: GPL-3.0-only
"""Stream the published INT8/W4A8 weights into standalone PyTorch models.

Only the Apache-2.0 tensor-kernel library is used; no ComfyUI runtime or server.
Never materialize a full BF16 copy of either model.
"""
from __future__ import annotations

import json
import re

import torch
from accelerate import init_empty_weights
from safetensors import safe_open


def load_checkpoint(model, path, *, language=False):
    import comfy_kitchen as ck
    from comfy_kitchen.tensor import AsymW4A8Int8Layout, QuantizedTensor, TensorWiseINT8Layout
    # The CUDA-13 extension is optional. CUDA 12.8/CPU use the Torch backend.
    if torch.version.cuda is None or tuple(map(int, torch.version.cuda.split(".")[:2])) < (13, 0):
        ck.registry.disable("cuda")
    ck.registry.disable("triton")
    loaded = set()
    with safe_open(str(path), framework="pt", device="cpu") as file:
        keys = set(file.keys())
        for name, parameter in list(model.named_parameters()):
            key = name
            if language and key.startswith("model."):
                key = "model.language_model." + key[len("model."):]
            split = None
            if key not in keys:
                match = re.search(r"\.proj\.([qkv])\.weight$", key)
                if match:
                    split = "qkv".index(match.group(1))
                    key = key[:match.start()] + ".proj.qkv.weight"
            if key not in keys:
                raise ValueError(f"Published MOSS checkpoint is missing {name} ({key}).")
            weight = file.get_tensor(key)
            parent, _, leaf = name.rpartition(".")
            module = model.get_submodule(parent) if parent else model
            quant_key = key.removesuffix("weight") + "comfy_quant"
            if quant_key in keys:
                options = json.loads(bytes(file.get_tensor(quant_key).tolist()))
                extra = options.get("params", {})
                option = {**extra, **options}.get
                prefix = key.removesuffix("weight")
                if options["format"] == "int8_tensorwise":
                    scale = file.get_tensor(prefix+"weight_scale")
                    if split is not None:
                        width = parameter.shape[0]
                        weight = weight[split*width:(split+1)*width]
                        if scale.numel() > 1:
                            scale = scale[split*width:(split+1)*width]
                    params = TensorWiseINT8Layout.Params(scale=scale, orig_dtype=torch.bfloat16,
                        orig_shape=tuple(parameter.shape), convrot=bool(option("convrot", False)),
                        convrot_groupsize=int(option("convrot_groupsize", 256)))
                    layout = "TensorWiseINT8Layout"
                elif options["format"] == "asym_w4a8_int8":
                    if split is not None:
                        raise ValueError("Fused W4A8 slicing is not supported.")
                    scale = file.get_tensor(prefix+"weight_s_rel")
                    if scale.dtype == torch.uint8:
                        scale = scale.view(torch.float8_e4m3fn)
                    params = AsymW4A8Int8Layout.Params(scale=scale,
                        s_channel=file.get_tensor(prefix+"weight_s_channel"),
                        codebook=file.get_tensor(prefix+"weight_codebook") if prefix+"weight_codebook" in keys else None,
                        orig_dtype=torch.bfloat16, orig_shape=tuple(parameter.shape),
                        group_size=int(option("group_size", 16)),
                        convrot_groupsize=int(option("convrot_groupsize", 256)))
                    layout = "AsymW4A8Int8Layout"
                else:
                    raise ValueError(f"Unsupported MOSS quantization: {options['format']}")
                value = QuantizedTensor(weight, layout, params)
            else:
                if split is not None:
                    width = parameter.shape[0]
                    weight = weight[split*width:(split+1)*width]
                dtype = torch.float32 if language and name.endswith(("A_log", "dt_bias")) else torch.bfloat16
                value = weight.to(dtype)
                if tuple(value.shape) != tuple(parameter.shape):
                    raise ValueError(f"MOSS tensor shape mismatch: {name}")
            module._parameters[leaf] = torch.nn.Parameter(value, requires_grad=False)
            loaded.add(name)
        # Persistent buffers such as rotary frequencies must retain FP32 precision.
        for name, buffer in list(model.named_buffers()):
            key = "model.language_model."+name[len("model."):] if language and name.startswith("model.") else name
            if key in keys:
                parent, _, leaf = name.rpartition(".")
                module = model.get_submodule(parent) if parent else model
                module._buffers[leaf] = file.get_tensor(key).to(dtype=buffer.dtype)
            elif buffer.is_meta:
                raise ValueError(f"MOSS has an uninitialized buffer: {name}")
    if any(p.is_meta for p in model.parameters()):
        raise ValueError("MOSS quantized loading left uninitialized parameters.")
    return model.eval(), len(loaded)


def load_transformer(root):
    from moss_edit.components import load_moss_module
    module = load_moss_module(root)
    config = json.loads((root / "transformer/config.json").read_text())
    with init_empty_weights():
        model = module.MOSSImageTransformer.from_config(config)
    return load_checkpoint(model, root / "transformer/moss_sft_int8.safetensors")[0]


def load_language_model(root):
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    config = AutoConfig.from_pretrained(root / "text_encoder", local_files_only=True).text_config
    config._attn_implementation = "sdpa"
    with init_empty_weights():
        model = Qwen3_5ForCausalLM(config)
    model = load_checkpoint(model, root / "text_encoder/qwen3.5_w4a8.safetensors", language=True)[0]
    tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer", local_files_only=True)
    return model, tokenizer


def place_transformer(model, device, *, reserve_bytes=2*1024**3):
    """Keep a bounded subset of layers resident; stream remaining leaf operations.

    MOSS Edit calls projection modules directly rather than Block.forward, so
    ordinary block offload hooks would never run. Hooks must cover its leaves.
    """
    if device.type != "cuda":
        return {"resident_layers": len(model.layers), "offloaded_layers": 0}
    for name, child in model.named_children():
        if name != "layers":
            child.to(device)
    free, _ = torch.cuda.mem_get_info(device)
    budget = max(0, free - reserve_bytes)
    resident = offloaded = 0

    def storage_bytes(parameter):
        raw = getattr(parameter, "_qdata", parameter)
        return raw.numel()*raw.element_size()

    def onload(module, args):
        module.to(device)

    def offload(module, args, output):
        module.to("cpu")

    for layer in model.layers:
        size = sum(storage_bytes(p) for p in layer.parameters())
        if size <= budget:
            layer.to(device)
            budget -= size
            resident += 1
            continue
        offloaded += 1
        covered = []
        for name, child in layer.named_modules():
            if any(name.startswith(prefix+".") for prefix in covered):
                continue
            if not list(child.parameters(recurse=False)):
                continue
            if hasattr(child, "lora_A"):
                covered.append(name)
            child.register_forward_pre_hook(onload)
            child.register_forward_hook(offload)
    return {"resident_layers": resident, "offloaded_layers": offloaded}
