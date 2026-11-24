# SPDX-License-Identifier: GPL-3.0-only
"""Optional standalone worker for FLUX.2 Klein/LanPaint; no host dependency.

This file is separately licensed from the Apache-2.0 Kyven transport. The complete
source is shipped beside its license. Communication is newline-delimited JSON.
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

PROTOCOL = sys.stdout


def emit(event: str, **values) -> None:
    PROTOCOL.write(json.dumps({"event": event, **values}, ensure_ascii=False) + "\n")
    PROTOCOL.flush()


def english_suppression_ids(tokenizer, vocabulary_size: int) -> list[int]:
    """Exclude non-English scripts and incomplete UTF-8 byte tokens at decoding.

    A system instruction alone is insufficient for this encoder checkpoint: it
    can switch from English to Chinese mid-answer. ASCII words/punctuation and
    EOS remain available; inputs can still contain any language.
    """
    allowed = set()
    for token, index in tokenizer.get_vocab().items():
        if index >= vocabulary_size:
            continue
        decoded = tokenizer.convert_tokens_to_string([token])
        if decoded and all(c in "\n\r\t" or 32 <= ord(c) < 127 for c in decoded):
            allowed.add(index)
    allowed.add(tokenizer.eos_token_id)
    for token in ("<think>", "</think>", "<|im_start|>", "<|endoftext|>"):
        allowed.discard(tokenizer.convert_tokens_to_ids(token))
    return [index for index in range(vocabulary_size) if index not in allowed]


class Runtime:
    @staticmethod
    def stitch(payload: dict) -> dict:
        """Gradient-domain reconstruction with clean background boundary data."""
        import cv2
        import numpy as np
        from PIL import Image

        source = np.array(Image.open(payload["source"]).convert("RGB"))
        generated = np.array(Image.open(payload["generated"]).convert("RGB"))
        selected = np.array(Image.open(payload["mask"]).convert("L")) > 0
        if source.shape != generated.shape or source.shape[:2] != selected.shape:
            raise ValueError("Seamless blend inputs must have matching dimensions.")
        info = {"applied": False}
        result = generated.copy()
        if np.any(selected) and np.any(~selected):
            from scipy.sparse import coo_matrix
            from scipy.sparse.linalg import splu

            clean = cv2.inpaint(np.ascontiguousarray(source[..., ::-1]),
                               selected.astype(np.uint8) * 255, 5, cv2.INPAINT_NS)[..., ::-1]
            # Solve every permitted pixel, including the first boundary pixel.
            # No erosion, reflected source subject, or alpha feather is involved.
            index = np.full(selected.shape, -1, dtype=np.int32)
            count = int(selected.sum())
            index[selected] = np.arange(count, dtype=np.int32)
            degree = np.zeros(count, dtype=np.float64)
            rhs = np.zeros((count, 3), dtype=np.float64)
            rows, columns, data = [], [], []
            g, c = generated.astype(np.float64), clean.astype(np.float64)
            directions = [((slice(1, None), slice(None)), (slice(None, -1), slice(None))),
                          ((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
                          ((slice(None), slice(1, None)), (slice(None), slice(None, -1))),
                          ((slice(None), slice(None, -1)), (slice(None), slice(1, None)))]
            for here, neighbor in directions:
                valid = index[here] >= 0
                ids = index[here][valid]
                others = index[neighbor][valid]
                degree[ids] += 1
                inside = others >= 0
                gradient = (g[here] - g[neighbor])[valid]
                gradient[~inside] = (c[here] - c[neighbor])[valid][~inside]
                rhs[ids] += gradient
                rhs[ids[~inside]] += source[neighbor][valid][~inside]
                rows.append(ids[inside])
                columns.append(others[inside])
                data.append(-np.ones(int(inside.sum())))
            rows.append(np.arange(count))
            columns.append(np.arange(count))
            data.append(degree)
            matrix = coo_matrix((np.concatenate(data),
                (np.concatenate(rows), np.concatenate(columns))), shape=(count, count)).tocsc()
            solution = splu(matrix, permc_spec="MMD_AT_PLUS_A").solve(rhs)
            residual = float(np.max(abs(matrix @ solution - rhs)))
            result[selected] = np.clip(np.rint(solution), 0, 255).astype(np.uint8)
            info = {"applied": True, "method": "clean-boundary-poisson",
                    "solver_residual": residual}
        result[~selected] = source[~selected]
        Image.fromarray(result).save(payload["output"])
        return {"metadata": info}

    def __init__(self, models: Path, device: str) -> None:
        import torch
        self.torch = torch
        torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
        self.root = models
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Select CPU or install the CUDA runtime.")
        self.dtype = torch.bfloat16 if self.device == "cpu" or torch.cuda.is_bf16_supported() else torch.float16
        self.lm = self.tokenizer = self.pipe = None
        self.english_mask = None
        self.operations = {}
        self.forwards = 0

    def cleanup(self) -> None:
        gc.collect()
        if self.device == "cuda":
            self.torch.cuda.empty_cache()

    def language_model(self) -> None:
        if self.lm is not None:
            return
        from transformers import AutoTokenizer, Qwen3ForCausalLM
        emit("progress", progress=0.14, message="Loading shared Qwen3-4B in system RAM")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.root / "tokenizer", local_files_only=True, use_fast=True)
        self.lm = Qwen3ForCausalLM.from_pretrained(
            self.root / "text_encoder", torch_dtype=self.dtype, local_files_only=True,
            attn_implementation="sdpa", low_cpu_mem_usage=True,
        ).eval()

    def enhance(self, settings: dict, *, output_contract: str | None = None) -> dict:
        torch = self.torch
        # Make room in VRAM even if the preceding action was image generation.
        if self.pipe is not None:
            self.pipe.transformer.to("cpu")
            self.pipe.vae.to("cpu")
        self.cleanup()
        self.language_model()
        contract = output_contract or (
            "Return only the final English editing brief. Preserve the requested action: "
            "removal must remain removal, adding must remain adding, and replacement must "
            "retain its new subject. Do not turn a removal request into a description of a person.")
        messages = [{"role": "system", "content": settings["system_prompt"] +
                     "\nYour final answer must be entirely in English. Never switch languages. "
                     + contract},
                    {"role": "user", "content": settings["prompt"]}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=False)
        inputs = self.tokenizer(text, return_tensors="pt")
        if inputs.input_ids.shape[1] > 2048:
            raise ValueError("Prompt and system instruction exceed 2048 tokens. Shorten them and retry.")
        emit("progress", progress=0.3, message="Qwen3-4B is processing the editing brief on CPU")
        torch.manual_seed(settings["seed"])
        temperature = settings["temperature"]
        kwargs = {"max_new_tokens": settings["max_new_tokens"], "do_sample": temperature > 0,
                  "pad_token_id": self.tokenizer.eos_token_id, "use_cache": True}
        from transformers import LogitsProcessor, LogitsProcessorList
        if self.english_mask is None:
            self.english_mask = torch.zeros(self.lm.config.vocab_size, dtype=torch.bool)
            self.english_mask[english_suppression_ids(self.tokenizer, self.lm.config.vocab_size)] = True
        blocked = self.english_mask

        class EnglishOnly(LogitsProcessor):
            def __call__(self, input_ids, scores):
                return scores.masked_fill(blocked.to(scores.device), -float("inf"))

        kwargs["logits_processor"] = LogitsProcessorList([EnglishOnly()])
        if temperature > 0:
            kwargs.update(temperature=temperature, top_p=settings["top_p"])
        with torch.inference_mode():
            output = self.lm.generate(**inputs, **kwargs)
        result = self.tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        result = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip().strip('"')
        return {"prompt": result, "metadata": {"model": "Qwen3-4B", "device": "cpu",
                                                "shared_text_encoder_weights": True}}

    def pipeline(self) -> None:
        if self.pipe is not None:
            return
        from diffusers import (
            Flux2KleinPipeline,
            Flux2Transformer2DModel,
            GGUFQuantizationConfig,
        )
        self.language_model()
        emit("progress", progress=0.18, message="Loading FLUX.2 Klein 4B Q4")
        transformer = Flux2Transformer2DModel.from_single_file(
            str(self.root / "flux-2-klein-4b-Q4_K_M.gguf"),
            config=str(self.root), subfolder="transformer", local_files_only=True,
            quantization_config=GGUFQuantizationConfig(compute_dtype=self.dtype),
            torch_dtype=self.dtype,
        )
        self.pipe = Flux2KleinPipeline.from_pretrained(
            self.root, transformer=transformer, text_encoder=self.lm,
            tokenizer=self.tokenizer, torch_dtype=self.dtype, local_files_only=True,
        )
        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()

    def image_prompt(self, prompt: str, seed: int) -> tuple[str, bool]:
        translated = any(c.isalpha() and not c.isascii() for c in prompt)
        if not translated:
            return prompt, False
        emit("progress", progress=0.13, message="Translating the prompt into English with Qwen3-4B")
        result = self.enhance({
            "prompt": prompt, "system_prompt":
            "Translate the user's image-editing brief into natural English. Preserve its "
            "meaning and requested details exactly. Do not expand it, invent details or "
            "describe a different scene. Return only the English translation.",
            "seed": seed, "max_new_tokens": 256, "temperature": 0.0, "top_p": 0.9,
        })
        return result["prompt"], True

    def edit_operation(self, prompt: str) -> str:
        if prompt in self.operations:
            return self.operations[prompt]
        result = self.enhance({"prompt": prompt, "system_prompt":
            "Classify an image editing request. Output exactly one label: REMOVE, ADD, "
            "REPLACE or EDIT. REMOVE means an object disappears and the existing background "
            "is restored, including replacing a person with empty surroundings. ADD means "
            "a new object appears. REPLACE means an existing subject becomes a different "
            "subject. EDIT means other changes. Examples: replace the person with background "
            "= REMOVE; replace the man with a woman = REPLACE; add a vase = ADD.",
            "seed": 0, "max_new_tokens": 8, "temperature": 0.0, "top_p": 0.9},
            output_contract="Return only REMOVE, ADD, REPLACE or EDIT. Do not rewrite the brief.")
        match = re.search(r"\b(REMOVE|ADD|REPLACE|EDIT)\b", result["prompt"].upper())
        operation = match.group(1).lower() if match else "edit"
        if len(self.operations) >= 64:
            self.operations.pop(next(iter(self.operations)))
        self.operations[prompt] = operation
        return operation

    def fill(self, payload: dict) -> dict:
        import numpy as np
        from lanpaint_pipeline import LanPaintConfig, LanPaintInpaintPipeline
        from lanpaint_pipeline.adapters.flux_klein import FluxKleinAdapter
        from PIL import Image

        torch = self.torch
        settings = dict(payload["settings"])
        original_prompt = settings["prompt"]
        settings["prompt"], translated = self.image_prompt(original_prompt, settings["seed"])
        operation = (self.edit_operation(settings["prompt"])
                     if (settings.get("color_match_mode") == "auto"
                         or settings.get("seamless_blend") == "auto") else "edit")
        self.pipeline()
        runtime = self

        class StagedAdapter(FluxKleinAdapter):
            @property
            def device(self):
                return torch.device(runtime.device)

            @property
            def dtype(self):
                return runtime.dtype

            def encode_prompt(self, prompt, negative_prompt, device):
                self.pipe.transformer.to("cpu")
                self.pipe.vae.to("cpu")
                runtime.cleanup()
                emit("progress", progress=0.22, message="Encoding prompt with Qwen3-4B on CPU")
                with torch.inference_mode():
                    positive, _ = self.pipe.encode_prompt(
                        prompt=prompt, device=torch.device("cpu"),
                        max_sequence_length=512,
                    )
                self.set_conditioning(positive.to(device=device, dtype=runtime.dtype))
                return self._prompt_bundle

            def encode_and_prepare(self, *args, **kwargs):
                self.pipe.transformer.to("cpu")
                self.pipe.vae.to(runtime.device)
                result = super().encode_and_prepare(*args, **kwargs)
                self.pipe.vae.to("cpu")
                runtime.cleanup()
                self.pipe.transformer.to(runtime.device)
                return result

            def predict_x0(self, *args, **kwargs):
                runtime.forwards += 1
                emit("progress", progress=min(0.88, 0.3 + runtime.forwards * 0.55 /
                     max(1, settings["steps"] * (settings["thinking_steps"] + 1))),
                     message=f"Generating masked patch (model evaluation {runtime.forwards})")
                return super().predict_x0(*args, **kwargs)

            def mask_to_latent_space(self, mask_pixel_keep):
                unpacked = self.pipe._unpack_latents_with_ids(
                    self._y_packed, self._ref_image_ids)
                # Nearest-neighbor sampling can mark partially selected latent
                # cells as KEEP, retaining pieces of the old silhouette.
                selected = torch.nn.functional.adaptive_max_pool2d(
                    1.0 - mask_pixel_keep, unpacked.shape[-2:])
                return (1.0 - selected).to(torch.float32).reshape(1, -1, 1)

            def decode_latents(self, latents):
                if not torch.isfinite(latents).all().item():
                    raise RuntimeError("LanPaint produced nonfinite values. Reduce Langevin Step Size and retry.")
                self.pipe.transformer.to("cpu")
                runtime.cleanup()
                self.pipe.vae.to(runtime.device)
                emit("progress", progress=0.9, message="Decoding generated patch")
                try:
                    return super().decode_latents(latents)
                finally:
                    self.pipe.vae.to("cpu")
                    runtime.cleanup()

        with Image.open(payload["source"]) as image:
            original = image.convert("RGB")
        with Image.open(payload["mask"]) as image:
            edit_mask = image.convert("L")
        maximum = min(settings["working_size"], 512) if payload.get("low_memory") else settings["working_size"]
        sizes = [maximum]
        if settings["oom_retry"] and maximum > 256:
            sizes.append(max(256, min(512, maximum // 2)))
        retried = False
        for attempt, maximum in enumerate(sizes):
            scale = min(1.0, maximum / max(original.size))
            content = tuple(max(1, round(v * scale)) for v in original.size)
            canvas = tuple(max(16, math.ceil(v / 16) * 16) for v in content)
            # Preserve aspect ratio; pad to the latent grid instead of stretching.
            resized = original.resize(content, Image.Resampling.LANCZOS)
            source = Image.fromarray(np.pad(np.asarray(resized),
                ((0, canvas[1] - content[1]), (0, canvas[0] - content[0]), (0, 0)), mode="edge"))
            # Keep any thin selected feature alive when reducing resolution.
            reduced_mask = edit_mask.resize(content, Image.Resampling.BOX)
            selected = np.asarray(reduced_mask) > 0
            mask = Image.new("L", canvas, 255)  # upstream: white = keep, black = edit
            mask.paste(Image.fromarray(np.where(selected, 0, 255).astype(np.uint8)), (0, 0))
            self.pipe.vae.tile_sample_min_size = settings["vae_tile_size"]
            self.pipe.vae.tile_latent_min_size = max(16, settings["vae_tile_size"] // 8)
            adapter = StagedAdapter(self.pipe)
            config = LanPaintConfig(n_steps=settings["thinking_steps"],
                step_size=settings["step_size"], friction=settings["friction"],
                chara_lambda=settings["strength"], beta=settings["beta"],
                early_stop=settings["early_stop"], cfg_big=1.0, blend_overlap=0)
            self.forwards = 0
            if self.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            try:
                result = LanPaintInpaintPipeline(adapter, config)(
                    prompt=settings["prompt"], image=source, mask_image=mask,
                    width=canvas[0], height=canvas[1], guidance_scale=1.0,
                    num_inference_steps=settings["steps"], seed=settings["seed"],
                ).images[0]
                result.crop((0, 0, *content)).resize(original.size, Image.Resampling.LANCZOS).save(payload["output"])
                peak = torch.cuda.max_memory_allocated() if self.device == "cuda" else 0
                return {"metadata": {"working_size": list(canvas), "oom_retried": retried,
                    "peak_vram_bytes": peak, "device": self.device, "seed": settings["seed"],
                    "steps": settings["steps"], "thinking_steps": settings["thinking_steps"],
                    "original_prompt": original_prompt, "effective_prompt": settings["prompt"],
                    "auto_translated": translated,
                    "edit_operation": operation,
                    "text_encoder_device": "cpu", "quantization": "Q4_K_M"}}
            except torch.cuda.OutOfMemoryError:
                self.pipe.transformer.to("cpu")
                self.pipe.vae.to("cpu")
                adapter = None
                self.cleanup()
                if attempt + 1 == len(sizes):
                    raise RuntimeError("Not enough VRAM for Generative Fill, even at the reduced size.") from None
                retried = True
                emit("progress", progress=0.3, message="VRAM limit reached; retrying a smaller crop")
        raise RuntimeError("No generation attempt completed.")


def main() -> None:
    # -I ignores PYTHONIOENCODING. Windows redirected pipes otherwise use the
    # ANSI code page and can turn Cyrillic into unpaired surrogate characters.
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="strict")
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", required=True, type=Path)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    runtime = None
    for line in sys.stdin:
        try:
            with contextlib.redirect_stdout(sys.stderr):
                if runtime is None:
                    runtime = Runtime(args.models, args.device)
                payload = json.loads(line)
                action = payload["action"]
                if action == "enhance":
                    result = runtime.enhance(payload["settings"])
                elif action == "fill":
                    result = runtime.fill(payload)
                elif action == "stitch":
                    result = runtime.stitch(payload)
                else:
                    raise ValueError("Unknown worker action")
            emit("result", **result)
        except Exception as exc:  # noqa: BLE001 - IPC boundary returns structured errors
            traceback.print_exc(file=sys.stderr)
            emit("error", message=str(exc), detail=type(exc).__name__)


if __name__ == "__main__":
    main()
