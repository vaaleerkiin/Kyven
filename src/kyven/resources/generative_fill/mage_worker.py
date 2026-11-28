# SPDX-License-Identifier: MIT
"""JSON-lines worker. Mage inference and boundary repair stay outside Nuke."""

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def emit(event, **values):
    print(json.dumps(dict(event=event, **values), ensure_ascii=False), flush=True)


def is_memory_error(error):
    """Recognize managed host/device budgets as well as allocator failures."""
    return bool(re.search(
        r"out of memory|cudaMalloc|(?:RAM|VRAM) (?:graph|weight) capacity exceeded|"
        r"allocat[^\n]*(?:fail|unable)|(?:fail|unable)[^\n]*allocat|"
        r"cannot fit (?:params|graph|buffer)", str(error), re.IGNORECASE))


class Runtime:
    def __init__(self, models, device):
        self.models = Path(models)
        self.device = "cuda" if device == "auto" else device
        self.bin = self.models / "runtime/bin"
        self.bin.mkdir(parents=True, exist_ok=True)
        for archive in sorted((self.models / "runtime").glob("*.zip")):
            destination = self.bin / ("llama" if archive.name.startswith("llama-") else "sd")
            destination.mkdir(exist_ok=True)
            marker = destination / (archive.name + ".ready")
            if marker.exists():
                continue
            with zipfile.ZipFile(archive) as bundle:
                for info in bundle.infolist():
                    target = (destination / info.filename).resolve()
                    if not target.is_relative_to(destination.resolve()):
                        raise ValueError("Runtime archive contains an unsafe path")
                bundle.extractall(destination)
            marker.write_text(str(archive.stat().st_size))
        self.cli = next((self.bin / "sd").rglob("sd-cli.exe"))
        self.llama = next((self.bin / "llama").rglob("llama-completion.exe"))
        self.environment = dict(
            os.environ, PATH=str(self.bin) + os.pathsep + os.environ.get("PATH", "")
        )

    def command(self, source, mask, output, prompt, seed, steps, strength=1.0):
        from PIL import Image

        with Image.open(source) as image:
            width, height = image.size
            # Bound Qwen3-VL image tokens independently of the diffusion canvas.
            # The original -i image and mask retain their processing resolution.
            reference = Path(output).with_name(Path(output).stem + "-reference.png")
            if max(image.size) > 1024:
                image = image.convert("RGB")
                image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                image.save(reference)
            else:
                reference = source
        return [
            str(self.cli),
            "--diffusion-model",
            str(self.models / "mage-flow-edit-turbo-Q6_K.gguf"),
            "--llm",
            str(self.models / "Qwen3VL-4B-Instruct-Q4_K_M.gguf"),
            "--llm_vision",
            str(self.models / "mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf"),
            "--vae",
            str(self.models / "mage_flow_vae_bf16.safetensors"),
            "--backend",
            "cpu" if self.device == "cpu" else "te=cpu,vae=cuda0,diffusion=cuda0",
            "--offload-to-cpu",
            "--disable-prefetch",
            "--diffusion-fa",
            "-t",
            "8",
            "-i",
            str(source),
            "-r",
            str(reference),
            "--mask",
            str(mask),
            "-W",
            str(width),
            "-H",
            str(height),
            "--steps",
            str(steps),
            "--strength",
            str(strength),
            "--cfg-scale",
            "1",
            "--sampling-method",
            "euler",
            "-s",
            str(seed),
            "-p",
            prompt,
            "-o",
            str(output),
        ]

    def run(self, command, timeout=1800, text_only=False):
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE if text_only else subprocess.STDOUT,
            env=dict(
                self.environment,
                PATH=str(Path(command[0]).parent) + os.pathsep + self.environment["PATH"],
            ),
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        try:
            output, errors = process.communicate(timeout=timeout)
        except BaseException:
            process.kill()
            process.wait()
            raise
        text = output.decode("utf-8", errors="replace")
        if errors:
            print(errors.decode("utf-8", errors="replace"), file=sys.stderr, flush=True)
        print(text, file=sys.stderr, flush=True)
        if process.returncode:
            raise RuntimeError(f"Mage runtime failed ({process.returncode}): {text[-3000:]}")
        return text

    def enhance(self, settings):
        system = settings.get(
            "system_prompt", "Translate the editing instruction to English. Preserve its meaning."
        )
        prompt = (
            "<|im_start|>system\n"
            + system
            + "\nReturn only the English editing instruction.<|im_end|>\n<|im_start|>user\n"
            + settings["prompt"]
            + "<|im_end|>\n<|im_start|>assistant\n"
        )
        with tempfile.TemporaryDirectory(prefix="kyven-mage-text-") as directory:
            path = Path(directory) / "prompt.txt"
            path.write_text(prompt, encoding="utf-8")
            text = self.run(
                [
                    str(self.llama),
                    "-m",
                    str(self.models / "Qwen3VL-4B-Instruct-Q4_K_M.gguf"),
                    "-f",
                    str(path),
                    "-n",
                    str(settings.get("max_new_tokens", 192)),
                    "-ngl",
                    "0",
                    "-t",
                    "8",
                    "--no-display-prompt",
                    "--simple-io",
                    "--no-conversation",
                    "--seed",
                    str(settings.get("seed", 42)),
                    "--temp",
                    str(settings.get("temperature", 0.6)),
                    "--top-p",
                    str(settings.get("top_p", 0.9)),
                    "-c",
                    "2048",
                ],
                settings.get("timeout_seconds", 600),
                text_only=True,
            )
        text = (
            re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).split("<|im_end|>")[0].strip()
        )
        text = text.replace("[end of text]", "").strip()
        if not text:
            raise RuntimeError("Prompt improvement returned no text")
        return {"prompt": text, "metadata": {"text_encoder_device": "cpu"}}

    def fill(self, payload):
        import numpy as np
        from PIL import Image

        settings = payload["settings"]
        prompt = settings["prompt"]
        auto_translated = bool(re.search(r"[\u0400-\u04ff]", prompt))
        if auto_translated:
            prompt = self.enhance({"prompt": prompt, "temperature": 0, "seed": settings["seed"]})[
                "prompt"
            ]
        original = Image.open(payload["source"]).convert("RGB")
        mask = Image.open(payload["mask"]).convert("L")
        maximum = (
            min(settings["working_size"], 512)
            if payload.get("low_memory")
            else settings["working_size"]
        )
        sizes = [maximum]
        if settings["oom_retry"] and maximum > 256:
            sizes.append(max(256, min(512, maximum // 2)))
        started = time.monotonic()
        # Conservative removal detection: only an explicit opening action
        # enables background continuity; additions keep their intended colors.
        operation = "remove" if re.match(
            r"^\s*(?:please\s+)?(?:remove|erase|delete|clean\s+up)\b", prompt, re.IGNORECASE
        ) else "edit"
        with tempfile.TemporaryDirectory(prefix="kyven-mage-fill-") as directory:
            root = Path(directory)
            for attempt, maximum in enumerate(sizes):
                scale = min(1, maximum / max(original.size))
                content = tuple(max(1, round(v * scale)) for v in original.size)
                canvas = tuple(math.ceil(v / 16) * 16 for v in content)
                image = np.array(original.resize(content, Image.Resampling.LANCZOS))
                image = np.pad(
                    image,
                    ((0, canvas[1] - content[1]), (0, canvas[0] - content[0]), (0, 0)),
                    mode="edge",
                )
                Image.fromarray(image).save(root / "source.png")
                small_mask = Image.new("L", canvas)
                small_mask.paste(
                    mask.resize(content, Image.Resampling.BOX).point(lambda v: 255 if v > 0 else 0)
                )
                small_mask.save(root / "mask.png")
                emit(
                    "progress", progress=0.25, message="Mage is generating inside the selected mask"
                )
                try:
                    self.run(
                        self.command(
                            root / "source.png",
                            root / "mask.png",
                            root / "raw.png",
                            prompt,
                            settings["seed"],
                            settings["steps"],
                        ),
                        settings["timeout_seconds"],
                    )
                except RuntimeError as exc:
                    if not is_memory_error(exc):
                        raise
                    if attempt + 1 == len(sizes):
                        raise MemoryError(
                            f"Mage ran out of RAM or VRAM at {canvas[0]} x {canvas[1]} px. "
                            "Use Processing Resolution 512 or 256 and close unused memory-heavy apps. "
                            "Enable Retry Smaller After GPU OOM for automatic recovery. "
                            "Details are in .runtime/mage-fill/worker.log."
                        ) from exc
                    emit(
                        "progress",
                        progress=0.3,
                        message=f"RAM/VRAM limit reached; retrying at {sizes[attempt + 1]} px",
                    )
                    continue
                with Image.open(root / "raw.png") as raw:
                    raw.crop((0, 0, *content)).resize(original.size, Image.Resampling.LANCZOS).save(
                        payload["output"]
                    )
                return {
                    "metadata": {
                        "working_size": list(canvas),
                        "reference_max_side": min(1024, max(canvas)),
                        "seed": settings["seed"],
                        "steps": settings["steps"],
                        "device": self.device,
                        "quantization": "Q6_K",
                        "mask_during_sampling": True,
                        "effective_prompt": prompt,
                        "original_prompt": settings["prompt"],
                        "auto_translated": auto_translated,
                        "oom_retried": bool(attempt),
                        "seconds": time.monotonic() - started,
                        "edit_operation": operation,
                        "runtime": "stable-diffusion.cpp",
                    }
                }

    def stitch(self, payload):
        import cv2
        import numpy as np
        from PIL import Image

        from kyven.generative_fill.boundary import repair_boundary

        settings = payload.get("settings", {})
        source = np.array(Image.open(payload["source"]).convert("RGB"))
        generated = np.array(Image.open(payload["generated"]).convert("RGB"))
        selected = np.array(Image.open(payload["mask"]).convert("L")) > 0
        width = settings.get("seam_width", 24)
        mode = settings.get("seam_repair", "auto")
        if mode == "off":
            Image.fromarray(generated).save(payload["output"])
            return {"metadata": {"applied": False}}
        neural = mode == "neural" and selected.any() and not selected.all()
        if neural:
            emit("progress", progress=0.94, message="Repairing the mask boundary with Mage")
            composite = np.where(selected[..., None], generated, source).astype(np.uint8)
            distance = cv2.distanceTransform(selected.astype(np.uint8), cv2.DIST_L2, 5)
            ring = selected & (distance <= width)
            with tempfile.TemporaryDirectory(prefix="kyven-mage-seam-") as directory:
                root = Path(directory)
                height, wide = selected.shape
                scale = min(1, settings.get("working_size", 512) / max(height, wide))
                content = (max(1, round(wide * scale)), max(1, round(height * scale)))
                canvas = tuple(math.ceil(v / 16) * 16 for v in content)
                image = Image.fromarray(composite).resize(content, Image.Resampling.LANCZOS)
                padded = Image.fromarray(
                    np.pad(
                        np.array(image),
                        ((0, canvas[1] - content[1]), (0, canvas[0] - content[0]), (0, 0)),
                        mode="edge",
                    )
                )
                padded.save(root / "source.png")
                mask = Image.new("L", canvas)
                mask.paste(
                    Image.fromarray((ring * 255).astype(np.uint8)).resize(
                        content, Image.Resampling.NEAREST
                    )
                )
                mask.save(root / "mask.png")
                self.run(
                    self.command(
                        root / "source.png",
                        root / "mask.png",
                        root / "repaired.png",
                        "Repair visible seams. Align architectural edges and surface lines across the selected boundary. Match surrounding light, texture and color. Preserve the current scene and objects.",
                        settings.get("seed", 42),
                        12,
                        0.3,
                    ),
                    settings.get("timeout_seconds", 1800),
                )
                with Image.open(root / "repaired.png") as image:
                    repaired = np.array(
                        image.crop((0, 0, *content)).resize(
                            (wide, height), Image.Resampling.LANCZOS
                        )
                    )
                # The second neural pass may touch only the inward boundary ring.
                generated[ring] = repaired[ring]
        result, metadata = repair_boundary(generated, source, selected, width)
        metadata.update(neural_pass=neural, mode=mode, preserve_edge_softness=True)
        Image.fromarray(result).save(payload["output"])
        return {"metadata": metadata}


def main():
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    runtime = None
    for line in sys.stdin:
        try:
            payload = json.loads(line)
            if runtime is None:
                runtime = Runtime(args.models, args.device)
            result = getattr(runtime, payload["action"])(
                payload.get("settings", {}) if payload["action"] == "enhance" else payload
            )
            emit("result", **result)
        except Exception as exc:  # noqa: BLE001 - structured IPC error boundary
            import traceback

            traceback.print_exc(file=sys.stderr)
            emit("error", message=str(exc), detail=type(exc).__name__)


if __name__ == "__main__":
    main()
