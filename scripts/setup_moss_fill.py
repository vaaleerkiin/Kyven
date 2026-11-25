"""Install the optional standalone worker; never modifies the core environment."""
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
    "import torch, cv2, scipy; from comfy_kitchen.tensor import QuantizedTensor, TensorWiseINT8Layout, AsymW4A8Int8Layout; "
    "from diffusers import AutoencoderKLFlux2; "
    "from transformers import AutoModel, AutoTokenizer, Qwen3_5Model; "
    "from LanPaint.lanpaint import LanPaint; "
    "from lanpaint_pipeline.pipeline import LanPaintInpaintPipeline; "
    "from importlib.metadata import version; "
    "assert version('diffusers') == '0.38.0'; "
    "assert version('transformers') == '5.12.1'; "
    "assert version('comfy-kitchen') == '0.2.41'; "
    "print('MOSS Fill runtime ready:', torch.__version__)"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-models", action="store_true")
    parser.add_argument("--check", action="store_true", help="Verify the current interpreter without installing")
    args = parser.parse_args()
    if args.check:
        subprocess.run([sys.executable, "-I", "-c", PROBE], check=True)
        return
    root = Path(__file__).resolve().parents[1]
    core_python = root / (".venv/Scripts/python.exe" if os.name == "nt" else ".venv/bin/python")
    if not core_python.is_file():
        raise SystemExit("Install the Kyven core runtime first (install.cmd).")
    if Path(sys.prefix).resolve() != (root / ".venv").resolve():
        subprocess.run([str(core_python), "-I", str(Path(__file__).resolve()), *sys.argv[1:]], check=True)
        return
    environment = root / ".runtime/moss-fill/venv"
    venv.EnvBuilder(with_pip=True).create(environment)
    executable = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    # Reuse the already installed, pinned core Torch wheel. The private site's
    # packages are first, so its Transformers cannot replace core ViTMatte's pin.
    core_site = root / (".venv/Lib/site-packages" if os.name == "nt"
                        else f".venv/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages")
    private_site = environment / ("Lib/site-packages" if os.name == "nt"
                                  else f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages")
    if not (core_site / "torch").is_dir():
        raise SystemExit("Install Kyven's CUDA runtime first (install.cmd).")
    # Python 3.10-3.12 may read .pth using the Windows ANSI code page. Keep the
    # file ASCII and escape the path so Cyrillic/non-ASCII folders still work.
    (private_site / "kyven_core_torch.pth").write_text(
        "import sys; sys.path.append(" + ascii(str(core_site)) + "); sys.path.insert(0, " +
        ascii(str(root / "src")) + ")\n", encoding="ascii")
    constraints = environment / "shared-torch-constraints.txt"
    constraints.write_text("\n".join(
        f"{name}=={importlib.metadata.version(name)}" for name in ("torch", "torchvision")
    ) + "\n", encoding="utf-8")
    subprocess.run([str(executable), "-m", "pip", "install", "--upgrade",
                    "--constraint", str(constraints),
                    "-r", str(root / "requirements/moss-fill.txt")], check=True)
    # Only these archives bypass resolution: their upstream metadata contains a
    # Git URL. All of their runtime dependencies are resolved by the first step.
    subprocess.run([str(executable), "-m", "pip", "install", "--no-deps",
                    "-r", str(root / "requirements/generative-fill-lanpaint.txt")], check=True)
    subprocess.run([str(executable), "-I", "-c", PROBE], check=True)
    if args.download_models:
        sys.path.insert(0, str(root / "src"))
        from kyven.models.catalog import ModelCatalog
        ModelCatalog.builtin().download("moss-sft-int8-edit-lanpaint", root / "models",
                                       progress=lambda n, total: print(f"{n}/{total}", flush=True))
    print("MOSS Fill runtime is ready.")


if __name__ == "__main__":
    main()
