# SPDX-License-Identifier: GPL-3.0-only
"""Native 33-channel learned inpainting with Z-Image Base and ControlNet Lite.

Separate process only. Uses the Diffusers Z-Image ControlNet components and the
official [zero control, known mask, masked-image latent] conditioning contract.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import re
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from worker import Runtime as KleinRuntime
from worker import emit


def release():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def control_condition(masked_latents, edit_mask):
    """The trained 33-channel contract, including the actual artist selection."""
    import torch
    known = torch.nn.functional.interpolate(1-edit_mask, size=masked_latents.shape[-2:], mode="nearest")
    return torch.cat([torch.zeros_like(masked_latents), known, masked_latents], dim=1).unsqueeze(2)


def place_models(transformer, controlnet, device):
    """Budget resident blocks; stream complete remaining blocks, including shared refiners."""
    import torch
    groups = [*transformer.noise_refiner, *transformer.context_refiner,
              *transformer.layers, *controlnet.control_layers, *controlnet.control_noise_refiner]
    groups = list({id(group): group for group in groups}.values())
    if device.type != "cuda":
        return {"resident_blocks": len(groups), "offloaded_blocks": 0}
    grouped = {id(module) for group in groups for module in group.modules()}
    visited = set()
    for model in (transformer, controlnet):
        for module in model.modules():
            if id(module) in visited or id(module) in grouped:
                continue
            visited.add(id(module))
            # Only direct Parameters: moving a parent would move every block.
            for name, value in list(module._parameters.items()):
                if value is not None:
                    module._parameters[name] = torch.nn.Parameter(value.to(device), requires_grad=False)
            for name, value in list(module._buffers.items()):
                if value is not None:
                    module._buffers[name] = value.to(device)
    free, _ = torch.cuda.mem_get_info(device)
    budget = max(0, free - 2*1024**3)
    resident = streamed = 0

    def onload(module, args):
        module.to(device)

    def offload(module, args, output):
        module.to("cpu")

    for group in groups:
        size = sum(p.untyped_storage().nbytes() for p in group.parameters())
        if size <= budget:
            group.to(device)
            budget -= size
            resident += 1
        else:
            group.register_forward_pre_hook(onload)
            group.register_forward_hook(offload)
            streamed += 1
    return {"resident_blocks": resident, "offloaded_blocks": streamed}


def predict_velocity(transformer, controlnet, latents, level, positive, negative, condition, guidance, scale):
    """Sequential CFG bounds temporary memory on an 8 GB GPU."""
    import torch
    inputs = list(latents.to(transformer.dtype).unsqueeze(2).unbind(0))

    def predict(embeddings):
        controls = controlnet(inputs, level, [embeddings], condition, conditioning_scale=scale)
        result = transformer(inputs, level, [embeddings], controlnet_block_samples=controls)[0]
        return torch.stack(result).squeeze(2).float()

    pos, neg = predict(positive), predict(negative)
    return -(pos + guidance*(pos-neg))


class Runtime:
    stitch = staticmethod(KleinRuntime.stitch)
    enhance = KleinRuntime.enhance
    language_model = KleinRuntime.language_model
    image_prompt = KleinRuntime.image_prompt
    cleanup = staticmethod(release)

    def __init__(self, models: Path, device: str):
        from kyven.server.bootstrap import reset_inherited_dll_directory
        reset_inherited_dll_directory()
        import torch
        self.torch, self.dtype = torch, torch.bfloat16
        self.models = self.root = models
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else
                                   "cpu" if device == "auto" else device)
        self.lm = self.tokenizer = self.pipe = self.english_mask = None
        self.transformer = self.controlnet = self.vae = None
        self.plans, self.conditions = {}, {}
        self.language_label = "Qwen3-4B"
        torch.set_num_threads(8)

    def prepare_text(self, prompt, negative):
        key = (prompt, negative)
        if key not in self.plans:
            emit("progress", progress=0.04, message="Preparing the desired masked content with Qwen")
            english, _ = self.image_prompt(prompt, 0)
            response = self.enhance({"prompt": english, "system_prompt":
                "Treat the user text as an image-editing brief. Return an English visual caption "
                "of the FINAL PHOTOGRAPH, a negative prompt and operation (remove/add/replace/edit). "
                "The caption must describe visible subject/background and photographic appearance "
                "in 40-90 words. Start with A photograph of. Never describe an editing action, "
                "mask, selection, reconstruction, unchanged/intact objects, integration or before/after. "
                "For removal describe an EMPTY background where the subject stood. Put the removed "
                "subject only in negative. Example: Remove a man from a sidewalk in front of a shop. "
                "Caption: A photograph of an empty sidewalk in front of a shop. The storefront and "
                "pavement continue naturally across the scene, with consistent perspective and "
                "continuous surfaces. Natural photographic lighting, realistic subdued colors, "
                "soft detail and consistent depth of field. Negative: person, man, human figure. "
                "For addition/replacement describe the NEW subject situated on the provided surface "
                "in the scene, with its requested color/size and appropriate photographic lighting. "
                "Do not invent specific architecture, props or weather absent from the brief. "
                "Objects are part of one scene, never product cards or inset photos unless requested. "
                "ADD/EDIT normally have empty negative. Return JSON: caption, negative, operation.",
                "seed": 0, "max_new_tokens": 256, "temperature": 0.0, "top_p": 0.9},
                output_contract="Return only one valid JSON object with caption, negative, operation. No markdown.")
            match = re.search(r"\{.*\}", response["prompt"], flags=re.DOTALL)
            plan = json.loads(match.group() if match else response["prompt"])
            plan["instruction"] = english
            if (plan.get("operation") not in {"remove", "add", "replace", "edit"}
                    or not isinstance(plan.get("caption"), str) or not plan["caption"].strip()
                    or not isinstance(plan.get("instruction"), str) or not plan["instruction"].strip()
                    or not isinstance(plan.get("negative"), str)):
                raise ValueError("Qwen could not prepare the masked-content description. Shorten the editing brief.")
            if any(c.isalpha() and not c.isascii() for name in ("instruction", "caption", "negative") for c in plan[name]):
                raise ValueError("Qwen did not return an English editing brief. Retry Improve Prompt.")
            if re.search(r"\b(added|removed|replaced|reconstructed|selection)\b|masked region|selected area|inside the mask",
                         plan["caption"], flags=re.IGNORECASE):
                corrected = self.enhance({"prompt": plan["instruction"]+"\nDraft caption: "+plan["caption"],
                    "system_prompt": "Write a 40-90 word visual description starting with A photograph of. "
                    "Describe a final photograph, never reconstruction, removal or intact objects. "
                    "Remove editing/selection terminology. Objects belong in the same continuous "
                    "scene, not on a poster, inset, card or pasted panel unless explicitly requested. "
                    "Preserve appearance and scene details. Return only the English visual caption.",
                    "seed": 0, "max_new_tokens": 192, "temperature": 0.0, "top_p": 0.9},
                    output_contract="Return only the visual caption, no editing instructions or commentary.")
                plan["caption"] = corrected["prompt"]
            if not plan["caption"].strip():
                raise ValueError("Qwen returned an empty target caption. Shorten the editing brief and retry.")
            if plan["operation"] in {"add", "edit"}:
                # Never automatically put the requested new subject into negative.
                plan["negative"] = ""
            else:
                plan["negative"] = re.sub(r"^(no|without|avoid)\s+", "", plan["negative"], flags=re.IGNORECASE)
            if plan["operation"] in {"add", "replace"}:
                plan["caption"] += " Integrated into the surrounding scene, matching its perspective, lighting and focus."
            if negative:
                translated, _ = self.image_prompt(negative, 0)
                plan["negative"] = ", ".join(filter(None, [plan["negative"], translated]))
            if len(self.plans) >= 64:
                self.plans.pop(next(iter(self.plans)))
            self.plans[key] = plan
        plan = self.plans[key]
        encoded_key = (plan["caption"], plan["negative"])
        hit = encoded_key in self.conditions
        if not hit:
            self.language_model()
            embeddings = []
            emit("progress", progress=0.1, message="Encoding target caption and negative prompt on CPU")
            for text in encoded_key:
                formatted = self.tokenizer.apply_chat_template([{"role": "user", "content": text}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=True)
                tokens = self.tokenizer(formatted, max_length=512, truncation=True, return_tensors="pt")
                with self.torch.no_grad():
                    result = self.lm(**tokens, output_hidden_states=True)
                embeddings.append(result.hidden_states[-2][0].detach().cpu())
                del result
            if len(self.conditions) >= 4:
                self.conditions.pop(next(iter(self.conditions)))
            self.conditions[encoded_key] = embeddings
        self.lm = self.tokenizer = None
        release()
        return plan, self.conditions[encoded_key], hit

    def load_models(self):
        from diffusers import (
            GGUFQuantizationConfig,
            ZImageControlNetModel,
            ZImageTransformer2DModel,
        )
        emit("progress", progress=0.18, message="Loading Z-Image Base Q4 and trained 33-channel Inpaint ControlNet")
        self.transformer = ZImageTransformer2DModel.from_single_file(
            str(self.models / "transformer/z-image-base-Q4_K_M.gguf"), config=str(self.models),
            subfolder="transformer", torch_dtype=self.dtype, local_files_only=True,
            quantization_config=GGUFQuantizationConfig(compute_dtype=self.dtype))
        self.controlnet = ZImageControlNetModel.from_single_file(
            str(self.models / "controlnet/model.safetensors"),
            config=str(Path(__file__).with_name("zimage_control_config.json")),
            torch_dtype=self.dtype, local_files_only=True)
        from safetensors import safe_open
        with safe_open(str(self.models / "controlnet/model.safetensors"), framework="pt") as file:
            if set(file.keys()) != set(self.controlnet.state_dict()):
                raise ValueError("Z-Image ControlNet Lite checkpoint does not match its native configuration.")
        self.controlnet = ZImageControlNetModel.from_transformer(self.controlnet, self.transformer)
        self.transformer.requires_grad_(False).eval()
        self.controlnet.requires_grad_(False).eval()
        self.placement = place_models(self.transformer, self.controlnet, self.device)
        release()

    def fill(self, payload):
        import numpy as np
        import torch
        from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
        from PIL import Image
        self.transformer = self.controlnet = self.vae = None
        release()
        settings = payload["settings"]
        plan, embeddings, text_hit = self.prepare_text(settings["prompt"], settings.get("zimage_negative_prompt", ""))
        original = Image.open(payload["source"]).convert("RGB")
        mask = Image.open(payload["mask"]).convert("L")
        maximum = min(settings["working_size"], 512) if payload.get("low_memory") else settings["working_size"]
        sizes = [maximum]
        if settings["oom_retry"] and maximum > 256:
            sizes.append(max(256, min(512, maximum//2)))
        for attempt, size in enumerate(sizes):
            try:
                if self.device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                scale = min(1.0, size/max(original.size))
                content = tuple(max(1, round(v*scale)) for v in original.size)
                canvas = tuple(max(16, math.ceil(v/16)*16) for v in content)
                pixels = np.pad(np.asarray(original.resize(content, Image.Resampling.LANCZOS)),
                    ((0, canvas[1]-content[1]), (0, canvas[0]-content[0]), (0, 0)), mode="edge")
                selected = np.zeros((canvas[1], canvas[0]), dtype=np.float32)
                selected[:content[1], :content[0]] = np.asarray(mask.resize(content, Image.Resampling.BOX)) > 0
                image = torch.from_numpy(pixels.copy()).permute(2, 0, 1)[None].to(self.device, self.dtype)/127.5-1
                edit = torch.from_numpy(selected)[None, None].to(self.device, self.dtype)
                emit("progress", progress=0.14, message="Encoding masked source for native Inpaint")
                self.vae = AutoencoderKL.from_pretrained(self.models / "vae", torch_dtype=self.dtype,
                    local_files_only=True).requires_grad_(False).eval().to(self.device)
                if self.vae.config.latent_channels != 16 or self.vae.config.shift_factor is None:
                    raise ValueError("Native Z-Image requires its 16-channel VAE. Reinstall its bundle in Model Manager.")
                self.vae.enable_tiling()
                self.vae.enable_slicing()
                with torch.no_grad():
                    def encode(value):
                        latent = self.vae.encode(value).latent_dist.mode()
                        return (latent-self.vae.config.shift_factor)*self.vae.config.scaling_factor
                    masked = encode(image*(1-edit))
                    condition = control_condition(masked, edit)
                self.vae.to("cpu")
                release()
                self.load_models()
                scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(self.models / "scheduler", local_files_only=True)
                scheduler.sigma_min = 0.0
                scheduler.set_timesteps(settings["steps"], device=self.device)
                noise = torch.randn(masked.shape, generator=torch.Generator("cpu").manual_seed(settings["seed"])).to(self.device)
                latents = noise.clone()
                positive, negative = [row.to(self.device) for row in embeddings]
                with torch.no_grad():
                    for i, timestep in enumerate(scheduler.timesteps):
                        emit("progress", progress=0.22+0.64*i/len(scheduler.timesteps),
                             message=f"Z-Image native masked diffusion {i+1}/{len(scheduler.timesteps)}")
                        level = (1000-timestep.reshape(1))/1000
                        velocity = predict_velocity(self.transformer, self.controlnet, latents, level,
                            positive, negative, condition, settings.get("zimage_guidance", 3.0),
                            settings.get("zimage_control_scale", 0.9))
                        latents = scheduler.step(velocity, timestep, latents, return_dict=False)[0].float()
                    self.transformer = self.controlnet = None
                    release()
                    emit("progress", progress=0.9, message="Decoding raw native masked generation")
                    self.vae.to(self.device)
                    decoded = self.vae.decode(latents.to(self.dtype)/self.vae.config.scaling_factor
                                              + self.vae.config.shift_factor, return_dict=False)[0]
                    output = ((decoded[0].float().clamp(-1, 1)+1)*127.5).round().byte().permute(1, 2, 0).cpu().numpy()
                Image.fromarray(output).crop((0, 0, *content)).resize(original.size, Image.Resampling.LANCZOS).save(payload["output"])
                peak = torch.cuda.max_memory_allocated() if self.device.type == "cuda" else 0
                self.vae = None
                release()
                return {"metadata": {"backend": "z-image-base-q4-control-inpaint", "quantization": "Q4_K_M",
                    "native_inpaint_condition_channels": int(condition.shape[1]), "mask_conditioned_during_sampling": True,
                    "known_latent_projection_each_step": False,
                    "raw_output_before_rgb_composite": True, "original_prompt": settings["prompt"],
                    "effective_prompt": plan["instruction"], "model_caption": plan["caption"], "negative_prompt": plan["negative"],
                    "edit_operation": plan["operation"], "auto_translated": any(c.isalpha() and not c.isascii() for c in settings["prompt"]),
                    "text_conditioning_cache_hit": text_hit, "working_size": list(canvas), "steps": settings["steps"],
                    "seed": settings["seed"], "device": str(self.device), "peak_vram_bytes": peak,
                    "oom_retried": attempt > 0, **self.placement}}
            except torch.cuda.OutOfMemoryError:
                self.transformer = self.controlnet = self.vae = None
                release()
                if attempt+1 == len(sizes):
                    raise
                emit("progress", progress=0.12, message="Z-Image GPU memory exhausted; retrying a smaller canvas")


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
                elif payload["action"] == "enhance":
                    result = runtime.enhance(payload["settings"])
                elif payload["action"] == "stitch":
                    result = runtime.stitch(payload)
                else:
                    raise ValueError("Unknown native Inpaint worker action.")
            emit("result", **result)
        except Exception as exc:  # noqa: BLE001 - structured process protocol
            traceback.print_exc(file=sys.stderr)
            emit("error", message=str(exc), detail=traceback.format_exc())


if __name__ == "__main__":
    main()
