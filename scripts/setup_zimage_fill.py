"""Install the isolated native Z-Image inpaint runtime outside Nuke."""
from __future__ import annotations

import argparse
import importlib.metadata
import os
import subprocess
import sys
import venv
from pathlib import Path

PROBE = (
    "from kyven.server.bootstrap import reset_inherited_dll_directory; "
    "reset_inherited_dll_directory(); "
    "import torch, cv2, gguf, scipy; "
    "from diffusers import ZImageControlNetInpaintPipeline, ZImageControlNetModel, "
    "ZImageTransformer2DModel, GGUFQuantizationConfig; "
    "from transformers import Qwen3ForCausalLM, AutoTokenizer; "
    "from importlib.metadata import version; "
    "assert version('diffusers') == '0.38.0'; "
    "assert version('transformers') == '4.57.6'; "
    "assert version('gguf') == '0.19.0'; "
    "print('Z-Image Native Inpaint runtime ready:', torch.__version__)"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-models", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        subprocess.run([sys.executable, "-I", "-c", PROBE], check=True)
        return
    root = Path(__file__).resolve().parents[1]
    core = root / (".venv/Scripts/python.exe" if os.name == "nt" else ".venv/bin/python")
    if not core.is_file():
        raise SystemExit("Install the Kyven base runtime from Model Manager first.")
    if Path(sys.prefix).resolve() != (root / ".venv").resolve():
        subprocess.run([str(core), "-I", str(Path(__file__).resolve()), *sys.argv[1:]], check=True)
        return
    environment = root / ".runtime/zimage-fill/venv"
    venv.EnvBuilder(with_pip=True).create(environment)
    executable = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    suffix = "Lib/site-packages" if os.name == "nt" else f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
    core_site, private_site = root / ".venv" / suffix, environment / suffix
    if not (core_site / "torch").is_dir():
        raise SystemExit("Install the Kyven base CUDA runtime first.")
    (private_site / "kyven_core_torch.pth").write_text(
        "import sys; sys.path.append(" + ascii(str(core_site)) + "); sys.path.insert(0, " +
        ascii(str(root / "src")) + ")\n", encoding="ascii")
    constraints = environment / "shared-torch-constraints.txt"
    constraints.write_text("\n".join(
        f"{name}=={importlib.metadata.version(name)}" for name in ("torch", "torchvision")
    ) + "\n", encoding="utf-8")
    subprocess.run([str(executable), "-m", "pip", "install", "--upgrade",
                    "--constraint", str(constraints), "-r", str(root / "requirements/zimage-fill.txt")], check=True)
    subprocess.run([str(executable), "-I", "-c", PROBE], check=True)
    if args.download_models:
        sys.path.insert(0, str(root / "src"))
        from kyven.models.catalog import ModelCatalog
        ModelCatalog.builtin().download("z-image-base-q4-control-inpaint", root / "models",
                                       progress=lambda n, total: print(f"{n}/{total}", flush=True))
    print("Z-Image Native Inpaint runtime is ready.")


if __name__ == "__main__":
    main()
