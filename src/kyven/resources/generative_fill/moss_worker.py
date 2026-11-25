# SPDX-License-Identifier: GPL-3.0-only
"""MOSS Edit with LanPaint constraints during diffusion, in an isolated process."""
from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from worker import Runtime as KleinRuntime
from worker import emit


def release() -> None:
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def edit_reference(source_latents, keep_mask, operation):
    """Hide the old subject for destructive edits; retain context for attention."""
    return source_latents * keep_mask if operation in {"remove", "replace"} else source_latents


class MossDenoiser:
    def __init__(self, runtime, reference, positive, negative, cache, guidance):
        self.inner_model = self.model_sampling = self
        self.runtime, self.reference = runtime, reference
        self.positive, self.negative = positive, negative
        self.cache, self.guidance = cache, guidance

    def noise_scaling(self, sigma, noise, latent_image):
        return (1-sigma)*latent_image + sigma*noise

    def __call__(self, x, t, model_options=None, seed=None):
        level = t.flatten()[0].float()
        def predict(condition):
            self.runtime.forward_calls += 1
            return self.runtime.editor(hidden_states=x, timestep=level,
                encoder_hidden_states=condition.embeds,
                encoder_attention_mask=condition.attention_mask,
                reference_latents=[self.reference], cache=self.cache).float()
        conditional = predict(self.positive)
        unconditional = predict(self.negative)
        guided = unconditional + self.guidance*(conditional-unconditional)
        return x.float()-level*guided, x.float()-level*conditional


class Runtime:
    stitch = staticmethod(KleinRuntime.stitch)
    enhance = KleinRuntime.enhance
    edit_operation = KleinRuntime.edit_operation
    cleanup = staticmethod(release)

    def __init__(self, models: Path, device: str):
        from kyven.server.bootstrap import reset_inherited_dll_directory
        reset_inherited_dll_directory()
        import torch
        self.torch, self.dtype = torch, torch.bfloat16
        self.lm = self.tokenizer = self.pipe = self.english_mask = None
        self.operations = {}
        self.translations = {}
        self.condition_cache = {}
        self.condition_cache_hit = False
        self.operation_system_prompt = (
            "Label the image edit: REMOVE (erase, restore background), ADD (insert), "
            "REPLACE (change to a new subject), EDIT (other changes). Replacing an object "
            "with empty background is REMOVE. Return the label only."
        )
        self.language_label = "Qwen3.5-9B W4A8"
        self.models = models
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else
                                   "cpu" if device == "auto" else device)
        self.encoder = self.editor = self.vae = None
        self.forward_calls = 0
        torch.set_num_threads(8)
        sys.path.insert(0, str(models / "edit_runtime"))

    def language_model(self):
        if self.lm is None:
            from moss_quantized import load_language_model
            emit("progress", progress=0.03, message="Loading compact MOSS Qwen3.5 encoder in RAM")
            self.lm, self.tokenizer = load_language_model(self.models)

    def image_prompt(self, prompt, seed):
        if prompt in self.translations:
            return self.translations[prompt]
        result = KleinRuntime.image_prompt(self, prompt, seed)
        if len(self.translations) >= 64:
            self.translations.pop(next(iter(self.translations)))
        self.translations[prompt] = result
        return result

    def encode_prompt(self, prompt):
        self.condition_cache_hit = prompt in self.condition_cache
        if self.condition_cache_hit:
            emit("progress", progress=0.08, message="Reusing cached MOSS text embeddings")
            self.lm = self.tokenizer = None
            release()
            return self.condition_cache[prompt]
        import torch
        from moss_edit.components import TextComponents, encode_prompts
        # The 9B encoder stays on CPU, and is released before loading the DiT.
        emit("progress", progress=0.03, message="Loading MOSS Qwen3.5 encoder on CPU")
        self.language_model()
        components = TextComponents(self.lm.model, self.tokenizer, torch.device("cpu"), torch.device("cpu"))
        emit("progress", progress=0.08, message="Encoding edit instruction")
        with torch.no_grad():
            conditions = encode_prompts(components, [prompt], negative_prompts=[""],
                                        max_sequence_length=512, device="cpu")
        del components
        self.lm = self.tokenizer = None
        release()
        if len(self.condition_cache) >= 4:
            self.condition_cache.pop(next(iter(self.condition_cache)))
        self.condition_cache[prompt] = conditions
        return conditions

    def load_model(self, reference_tokens=0):
        import torch
        from moss_edit.lora import inject_lora, load_adapter
        from moss_edit.model import MossEditModel
        from moss_quantized import load_transformer, place_transformer
        if self.editor is not None:
            return
        emit("progress", progress=0.16, message="Loading ready INT8 MOSS SFT and BF16 Edit adapter")
        config = json.loads((self.models / "edit/adapter_config.json").read_text())
        first = next(iter(config["modules"].values()))
        model = load_transformer(self.models)
        inject_lora(model, rank=first["rank"], alpha=first["alpha"],
                    dropout=first["dropout"], param_dtype=torch.bfloat16,
                    targets=config["targets"])
        load_adapter(model, self.models / "edit")
        # LoRA uses BF16 Parameters, not Linear children; only base Linear weights
        # are quantized. No full BF16 DiT is ever moved onto an 8 GB GPU.
        model.requires_grad_(False).eval()
        self.quantized_linears = sum(hasattr(p, "_qdata") for p in model.parameters())
        cache_bytes = 4 * model.config.num_layers * model.config.dim * reference_tokens
        # Smaller canvases can trade a few resident layers for reusable reference
        # KV. Reserve it before placement so the later cache budget can fit.
        cache_reserve = cache_bytes if cache_bytes <= 768*1024**2 else 0
        self.placement = place_transformer(model, self.device,
            reserve_bytes=2*1024**3 + cache_reserve)
        self.editor = MossEditModel(model, gradient_checkpointing=False).eval()
        release()

    def reference_cache(self, reference):
        """Reuse reference attention when it fits beside streamed layers/activations."""
        import torch
        if self.device.type == "cuda":
            config = self.editor.config
            estimated = 4 * config.num_layers * config.dim * reference.shape[-2] * reference.shape[-1]
            free, _ = torch.cuda.mem_get_info(self.device)
            if free < estimated + 1536*1024**2:
                return None
        cache = None
        try:
            cache = self.editor.prepare_cache([reference])
        except torch.cuda.OutOfMemoryError:
            emit("progress", progress=0.19, message="Reference cache exceeds VRAM; sampling without it")
        release()
        return cache

    def fill(self, payload):
        import numpy as np
        import torch
        from LanPaint.lanpaint import LanPaint
        from lanpaint_pipeline.utils import make_current_times
        from moss_edit.components import load_moss_module
        from moss_edit.processing import decode_latents, encode_image, image_to_tensor
        from PIL import Image

        settings = dict(payload["settings"])
        # Release a resident DiT before text encoding to bound both RAM and VRAM.
        if self.editor is not None:
            self.editor = None
            release()
        original_prompt = settings["prompt"]
        settings["prompt"], translated = self.image_prompt(original_prompt, settings["seed"])
        operation = self.edit_operation(settings["prompt"])
        conditions = self.encode_prompt(settings["prompt"])
        original = Image.open(payload["source"]).convert("RGB")
        mask = Image.open(payload["mask"]).convert("L")
        maximum = min(settings["working_size"], 512) if payload.get("low_memory") else settings["working_size"]
        sizes = [maximum]
        if settings["oom_retry"] and maximum > 256:
            sizes.append(max(256, min(512, maximum // 2)))
        module = load_moss_module(self.models)
        for attempt, size in enumerate(sizes):
            try:
                if self.device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                scale = min(1.0, size / max(original.size))
                content = tuple(max(1, round(v * scale)) for v in original.size)
                canvas = tuple(max(16, math.ceil(v / 16) * 16) for v in content)
                pixels = np.pad(np.asarray(original.resize(content, Image.Resampling.LANCZOS)),
                    ((0, canvas[1]-content[1]), (0, canvas[0]-content[0]), (0, 0)), mode="edge")
                source = Image.fromarray(pixels)
                selected = np.zeros((canvas[1], canvas[0]), dtype=np.float32)
                selected[:content[1], :content[0]] = np.asarray(mask.resize(content, Image.Resampling.BOX)) > 0
                emit("progress", progress=0.12, message="Encoding source as MOSS Edit reference and masked diffusion constraint")
                self.vae = module.AutoencoderKLFlux2.from_pretrained(self.models / "vae",
                    torch_dtype=torch.float32, local_files_only=True).requires_grad_(False).eval()
                self.vae.enable_tiling()
                self.vae.enable_slicing()
                with torch.no_grad():
                    self.vae.to(self.device)
                    y = encode_image(self.vae, image_to_tensor(source), sample_posterior=False).to(self.device)
                    self.vae.to("cpu")
                release()
                keep = 1.0 - torch.nn.functional.adaptive_max_pool2d(
                    torch.from_numpy(selected).to(self.device)[None, None], y.shape[-2:])
                reference = edit_reference(y, keep, operation)
                self.load_model(reference.shape[-2] * reference.shape[-1])
                scheduler = module.MOSSImageScheduler.from_pretrained(self.models / "scheduler", local_files_only=True)
                scheduler.set_timesteps(settings["steps"], device=self.device)
                generator = torch.Generator("cpu").manual_seed(settings["seed"])
                noise = torch.randn(y.shape, generator=generator, dtype=torch.float32).to(self.device)
                latents = noise.clone()
                positive, negative = conditions.positive.to(self.device), conditions.negative.to(self.device)
                guidance = float(settings.get("moss_guidance", 4.0))

                with torch.no_grad():
                    cache = self.reference_cache(reference)
                    reference_cache_enabled = cache is not None
                    sampler = LanPaint(Model=MossDenoiser(self, reference, positive, negative, cache, guidance), NSteps=settings["thinking_steps"],
                        Friction=settings["friction"], Lambda=settings["strength"], Beta=settings["beta"],
                        StepSize=settings["step_size"], IS_FLUX=False, IS_FLOW=True)
                    self.forward_calls = 0
                    torch.manual_seed(settings["seed"])
                    for i, timestep in enumerate(scheduler.timesteps):
                        level = scheduler.sigmas[i].item()
                        emit("progress", progress=0.2+0.65*i/len(scheduler.timesteps),
                             message=f"MOSS masked diffusion {i+1}/{len(scheduler.timesteps)}")
                        x0 = sampler(x=latents, latent_image=y, noise=noise,
                            sigma=torch.tensor([level], device=self.device), latent_mask=keep,
                            current_times=make_current_times(level, self.device), model_options={},
                            seed=settings["seed"],
                            n_steps=0 if len(scheduler.timesteps)-i <= settings["early_stop"] else None)
                        latents = scheduler.step((latents-x0)/max(level, 1e-6), timestep,
                                                 latents, return_dict=False)[0].float()
                    known_error = float(((latents-y)*keep).abs().max())
                    # Final latent conditioning is part of the sampler, before
                    # VAE decoding and before any RGB compositing occurs.
                    latents = latents * (1-keep) + y*keep
                    self.editor = None
                    del sampler, cache, positive, negative
                    release()
                    emit("progress", progress=0.9, message="Decoding raw MOSS masked generation")
                    decoded = decode_latents(self.vae.to(self.device), latents)
                    image = Image.fromarray(((decoded[0].float().cpu().clamp(-1,1)+1)*127.5)
                        .round().to(torch.uint8).permute(1,2,0).numpy())
                image.crop((0,0,*content)).resize(original.size, Image.Resampling.LANCZOS).save(payload["output"])
                peak = torch.cuda.max_memory_allocated() if self.device.type == "cuda" else 0
                self.vae = None
                release()
                return {"metadata": {"backend": "moss-sft-int8-edit-lanpaint", "quantization": "int8_convrot",
                    "text_quantization": "w4a8_gptq", "original_prompt": original_prompt, "auto_translated": translated,
                    "text_conditioning_cache_hit": self.condition_cache_hit,
                    "reference_cache_enabled": reference_cache_enabled,
                    "quantized_linears": self.quantized_linears, "edit_operation": operation,
                    "effective_prompt": settings["prompt"], "working_size": list(canvas),
                    "steps": settings["steps"], "seed": settings["seed"], "thinking_steps": settings["thinking_steps"],
                    "peak_vram_bytes": peak, "device": str(self.device), "oom_retried": attempt > 0,
                    "mask_conditioned_during_sampling": True, "reference_conditioning": "MOSS Edit t0 prefix",
                    "selected_subject_hidden_in_reference": operation in {"remove", "replace"},
                    "known_latent_error_before_final_projection": known_error,
                    "denoiser_forwards": self.forward_calls, "raw_output_before_rgb_composite": True, **self.placement}}
            except torch.cuda.OutOfMemoryError:
                self.editor = self.vae = None
                release()
                if attempt + 1 == len(sizes):
                    raise
                emit("progress", progress=0.1, message="MOSS GPU memory exhausted; retrying with smaller canvas")


def main():
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="strict")
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    runtime = None
    for line in sys.stdin:
        try:
            with contextlib.redirect_stdout(sys.stderr):
                if runtime is None:
                    runtime = Runtime(args.models, args.device)
                payload = json.loads(line)
                if payload["action"] == "fill":
                    result = runtime.fill(payload)
                elif payload["action"] == "stitch":
                    result = runtime.stitch(payload)
                elif payload["action"] == "enhance":
                    result = runtime.enhance(payload["settings"])
                else:
                    raise ValueError("Unknown MOSS worker action.")
            emit("result", **result)
        except Exception as exc:  # noqa: BLE001 - process protocol must return a structured error
            traceback.print_exc(file=sys.stderr)
            emit("error", message=str(exc), detail=traceback.format_exc())


if __name__ == "__main__":
    main()
