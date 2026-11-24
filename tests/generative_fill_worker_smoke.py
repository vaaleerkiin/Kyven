"""Opt-in real CPU pipeline smoke using tiny random weights, no downloads.

Run with .runtime/generative-fill/venv/Scripts/python.exe. This verifies shapes,
mask polarity, padding and the actual LanPaint/Diffusers adapter, not image quality.
"""
from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path

import torch
from diffusers import (
    AutoencoderKLFlux2,
    FlowMatchEulerDiscreteScheduler,
    Flux2KleinPipeline,
    Flux2Transformer2DModel,
)
from PIL import Image
from transformers import BatchEncoding, Qwen3Config, Qwen3ForCausalLM


class TinyTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"]

    def __call__(self, prompt, **kwargs):
        return BatchEncoding({"input_ids": torch.ones((1, 8), dtype=torch.long),
                              "attention_mask": torch.ones((1, 8), dtype=torch.long)})


def main() -> None:
    root = Path(__file__).parents[1]
    spec = importlib.util.spec_from_file_location("fill_worker",
        root / "src/kyven/resources/generative_fill/worker.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    runtime = module.Runtime(root, "cpu")
    dtype = runtime.dtype
    runtime.lm = Qwen3ForCausalLM(Qwen3Config(vocab_size=64, hidden_size=32,
        intermediate_size=64, num_hidden_layers=28, num_attention_heads=2,
        num_key_value_heads=1, head_dim=16)).to(dtype).eval()
    runtime.tokenizer = TinyTokenizer()
    transformer = Flux2Transformer2DModel(in_channels=16, num_layers=1,
        num_single_layers=1, num_attention_heads=2, attention_head_dim=32,
        joint_attention_dim=96, axes_dims_rope=(8, 8, 8, 8)).to(dtype).eval()
    vae = AutoencoderKLFlux2(block_out_channels=(32, 32, 32, 32), latent_channels=4,
        down_block_types=("DownEncoderBlock2D",) * 4,
        up_block_types=("UpDecoderBlock2D",) * 4, sample_size=128).to(dtype).eval()
    runtime.pipe = Flux2KleinPipeline(
        scheduler=FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True),
        vae=vae, transformer=transformer, text_encoder=runtime.lm,
        tokenizer=runtime.tokenizer, is_distilled=True)
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        Image.new("RGB", (73, 51), (100, 120, 80)).save(folder / "source.png")
        mask = Image.new("L", (73, 51), 0)
        mask.paste(255, (20, 10, 45, 35))
        mask.save(folder / "mask.png")
        settings = {"prompt": "a vase", "seed": 42, "steps": 3, "working_size": 256,
            "thinking_steps": 1, "step_size": 0.2, "friction": 15, "strength": 8,
            "beta": 1, "early_stop": 1, "vae_tile_size": 128, "oom_retry": False}
        result = runtime.fill({"settings": settings, "source": str(folder / "source.png"),
            "mask": str(folder / "mask.png"), "output": str(folder / "result.png")})
        assert Image.open(folder / "result.png").size == (73, 51)
        assert result["metadata"]["working_size"] == [80, 64]
        assert result["metadata"]["device"] == "cpu"
        print(json.dumps({"ok": True, **result}))


if __name__ == "__main__":
    main()
