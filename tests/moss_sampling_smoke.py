"""Run with the private MOSS interpreter after installing its verified code bundle."""
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src/kyven/resources/generative_fill"),
               str(ROOT / "models/moss-sft-int8-edit-lanpaint/edit_runtime")]

import torch
from accelerate import init_empty_weights
from comfy_kitchen.tensor import AsymW4A8Int8Layout, TensorWiseINT8Layout
from LanPaint.lanpaint import LanPaint
from lanpaint_pipeline.utils import make_current_times
from moss_edit.components import Condition, load_moss_module
from moss_edit.lora import inject_lora
from moss_edit.model import MossEditModel
from moss_quantized import load_checkpoint, place_transformer
from moss_worker import MossDenoiser, Runtime, edit_reference
from safetensors.torch import save_file

torch.set_num_threads(4)
torch.manual_seed(17)
module = load_moss_module(ROOT / "models/moss-sft-int8-edit-lanpaint")
model = module.MOSSImageTransformer(dim=64, ffn_dim=128, num_heads=8, num_layers=2,
                                  text_dim=32, freq_dim=16, latent_channels=8).bfloat16().eval()
state = {k: v.clone().contiguous() for k, v in model.state_dict().items()}
for layer in range(2):
    prefix = f"layers.{layer}.proj."
    weights = torch.cat([state.pop(prefix + axis + ".weight") for axis in "qkv"])
    data, params = TensorWiseINT8Layout.quantize(weights, per_channel=True,
                                               convrot=True, convrot_groupsize=16)
    state[prefix + "qkv.weight"] = data
    state[prefix + "qkv.weight_scale"] = params.scale
    config = json.dumps({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 16})
    state[prefix + "qkv.comfy_quant"] = torch.tensor(list(config.encode()), dtype=torch.uint8)
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "tiny.safetensors"
    save_file(state, path)
    with init_empty_weights():
        loaded = module.MOSSImageTransformer.from_config(model.config)
    loaded, _ = load_checkpoint(loaded, path)
    # Exercise the compact text-encoder weight format on CPU as well.
    dense = torch.randn(64, 64).bfloat16()
    data, params = AsymW4A8Int8Layout.quantize(dense, group_size=16, convrot_groupsize=16)
    compressed = {"weight": data, "weight_s_rel": params.scale,
        "weight_s_channel": params.s_channel,
        "comfy_quant": torch.tensor(list(json.dumps({"format": "asym_w4a8_int8",
            "group_size": 16, "convrot_groupsize": 16}).encode()), dtype=torch.uint8)}
    if params.codebook is not None:
        compressed["weight_codebook"] = params.codebook
    text_path = Path(directory)/"text.safetensors"
    save_file(compressed, text_path)
    with init_empty_weights():
        text_linear = torch.nn.Linear(64, 64, bias=False)
    text_linear, _ = load_checkpoint(text_linear, text_path)
    inputs = torch.randn(2, 64).bfloat16()
    assert torch.isfinite(text_linear(inputs)).all()
inject_lora(loaded, rank=2, alpha=2, param_dtype=torch.bfloat16)
loaded.requires_grad_(False)
editor = MossEditModel(loaded).eval()
source = torch.randn(1, 8, 3, 3)
positive = Condition(torch.randn(1, 4, 32).bfloat16(), torch.ones(1, 4, dtype=torch.bool))
negative = Condition(torch.zeros(1, 2, 32).bfloat16(), torch.ones(1, 2, dtype=torch.bool))


def sample(keep):
    with torch.no_grad():
        runtime = SimpleNamespace(editor=editor, forward_calls=0, device=source.device)
        reference = edit_reference(source, keep, "remove")
        assert torch.equal(reference * keep, source * keep)
        assert torch.count_nonzero(reference * (1-keep)) == 0
        assert edit_reference(source, keep, "edit") is source
        assert edit_reference(source, keep, "add") is source
        assert torch.equal(edit_reference(source, keep, "replace"), reference)
        # Reference stamps remain valid after leaf onload/offload as well.
        cache = Runtime.reference_cache(runtime, reference)
        assert cache is not None
        denoiser = MossDenoiser(runtime, reference, positive, negative, cache, 3.0)
        sampler = LanPaint(Model=denoiser, NSteps=1, IS_FLOW=True, IS_FLUX=False,
                           Friction=15.0, Lambda=8.0, Beta=1.0, StepSize=0.2)
        noise = torch.randn(source.shape, generator=torch.Generator().manual_seed(42)).to(source.device)
        latents = noise.clone()
        scheduler = module.MOSSImageScheduler()
        scheduler.set_timesteps(3, device=source.device)
        torch.manual_seed(42)
        for i, timestep in enumerate(scheduler.timesteps):
            level = scheduler.sigmas[i].item()
            x0 = sampler(x=latents, latent_image=source, noise=noise, sigma=torch.tensor([level], device=source.device),
                latent_mask=keep, current_times=make_current_times(level, source.device),
                model_options={}, seed=42)
            latents = scheduler.step((latents-x0)/level, timestep, latents, return_dict=False)[0]
        return latents, runtime.forward_calls


keep = torch.ones(1, 1, 3, 3)
keep[:, :, 1, 1] = 0
masked, calls = sample(keep)
full, _ = sample(torch.zeros_like(keep))
assert torch.isfinite(masked).all()
known_error = float(((masked-source)*keep).abs().max())
assert known_error < 1e-5, known_error
assert float((masked[:, :, 1, 1]-source[:, :, 1, 1]).abs().max()) > 0.01
assert not torch.allclose(masked[:, :, 1, 1], full[:, :, 1, 1])
if torch.cuda.is_available():
    from unittest.mock import patch
    with patch("torch.cuda.mem_get_info", return_value=(2*1024**3, 8*1024**3)):
        placement = place_transformer(loaded, torch.device("cuda"))
    assert placement["offloaded_layers"] == 2
    source, positive, negative = source.cuda(), positive.to("cuda"), negative.to("cuda")
    masked_gpu, _ = sample(keep.cuda())
    assert torch.isfinite(masked_gpu).all()
    assert float(((masked_gpu-source)*keep.cuda()).abs().max()) < 1e-5
    with patch("torch.cuda.mem_get_info", return_value=(0, 8*1024**3)):
        assert Runtime.reference_cache(SimpleNamespace(editor=editor, device=source.device), source) is None
print(json.dumps({"ok": True, "known_latent_error": known_error, "forwards": calls,
    "mask_changes_raw_generated_latents": True, "rgb_compositing_used": False,
    "real_moss_edit_reference_attention": True, "published_int8_layout": True,
    "cuda_leaf_offload": torch.cuda.is_available(), "bounded_reference_cache": True}))
