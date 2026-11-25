"""Private-runtime native learned-mask/CPU/offload qualification; synthetic inputs."""
import argparse
import gc
import json
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/kyven/resources/generative_fill"))

import torch
from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    ZImageControlNetModel,
    ZImageTransformer2DModel,
)
from zimage_worker import Runtime, control_condition, place_models, predict_velocity

parser = argparse.ArgumentParser()
parser.add_argument("--full-model", action="store_true")
args = parser.parse_args()
torch.manual_seed(17)
torch.set_num_threads(4)
model = ZImageTransformer2DModel(dim=256, n_layers=2, n_heads=16, n_kv_heads=16,
    cap_feat_dim=32, axes_dims=[4, 6, 6], axes_lens=[64, 64, 64]).bfloat16().eval()
control = ZImageControlNetModel(control_layers_places=[0, 1], control_refiner_layers_places=[0, 1],
    control_in_dim=33, add_control_noise_refiner="control_noise_refiner", dim=256,
    n_heads=16, n_kv_heads=16).bfloat16().eval()
# A new ControlNet has zero output projections; emulate nonzero learned residuals.
with torch.no_grad():
    for value in control.parameters():
        if value.ndim == 2 and torch.count_nonzero(value) == 0:
            value.normal_(std=0.02)
control = ZImageControlNetModel.from_transformer(control, model)
source = torch.randn(1, 16, 4, 4)
edit = torch.zeros(1, 1, 32, 32)
edit[:, :, 8:24, 8:24] = 1
condition = control_condition(source.bfloat16(), edit.bfloat16())
assert condition.shape == (1, 33, 1, 4, 4)
assert torch.count_nonzero(condition[:, :16]) == 0
positive, negative = torch.randn(4, 32).bfloat16(), torch.zeros(2, 32).bfloat16()
noise = torch.randn_like(source)


def denoise(model, control, source, condition, positive, negative):
    scheduler = FlowMatchEulerDiscreteScheduler(shift=6)
    scheduler.sigma_min = 0
    scheduler.set_timesteps(3, device=source.device)
    latents = noise.to(source.device).clone()
    with torch.no_grad():
        for i, timestep in enumerate(scheduler.timesteps):
            velocity = predict_velocity(model, control, latents, (1000-timestep.reshape(1))/1000,
                                        positive, negative, condition, 4.0, 0.9)
            latents = scheduler.step(velocity, timestep, latents, return_dict=False)[0].float()
    return latents


with torch.no_grad():
    level = torch.tensor([0.5])
    a = predict_velocity(model, control, noise, level, positive, negative, condition, 4.0, 0.9)
    changed = condition.clone()
    changed[:, 16] = 1-changed[:, 16]
    b = predict_velocity(model, control, noise, level, positive, negative, changed, 4.0, 0.9)
assert not torch.allclose(a, b), "Learned mask channel must change raw denoiser output"
result = denoise(model, control, source, condition, positive, negative)
assert torch.isfinite(result).all()
if torch.cuda.is_available():
    with patch("torch.cuda.mem_get_info", return_value=(2*1024**3, 8*1024**3)):
        placement = place_models(model, control, torch.device("cuda"))
    assert placement["offloaded_blocks"] > 0
    result = denoise(model, control, source.cuda(), condition.cuda(), positive.cuda(), negative.cuda())
    assert torch.isfinite(result).all()
del model, control
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
report = {"ok": True, "learned_mask_changes_raw_denoiser": True,
    "extra_source_latent_projection": False, "rgb_compositing_used": False,
    "native_condition_channels": 33, "cuda_offload": torch.cuda.is_available()}
if args.full_model:
    runtime = Runtime(ROOT/"models/z-image-base-q4-control-inpaint", "cuda")
    runtime.load_models()
    x = torch.randn(1, 16, 16, 16, device="cuda")
    c = torch.zeros(1, 33, 1, 16, 16, device="cuda", dtype=torch.bfloat16)
    p = torch.randn(4, 2560, device="cuda", dtype=torch.bfloat16)
    n = torch.zeros(2, 2560, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        a = predict_velocity(runtime.transformer, runtime.controlnet, x, torch.tensor([0.5], device="cuda"), p, n, c, 4, 0.9)
        c[:, 16, :, 4:12, 4:12] = 1
        b = predict_velocity(runtime.transformer, runtime.controlnet, x, torch.tensor([0.5], device="cuda"), p, n, c, 4, 0.9)
    difference = float((a-b).abs().max())
    assert torch.isfinite(a).all() and torch.isfinite(b).all() and difference > 1e-5
    report.update(full_checkpoint_mask_difference=difference,
                  peak_vram_bytes=torch.cuda.max_memory_allocated(), **runtime.placement)
print(json.dumps(report))
