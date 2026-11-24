"""Install the optional standalone worker; never modifies the core environment."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import venv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-models", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    environment = root / ".runtime/generative-fill/venv"
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
    (private_site / "kyven_core_torch.pth").write_text(str(core_site) + "\n", encoding="utf-8")
    subprocess.run([str(executable), "-m", "pip", "install", "--ignore-installed",
                    "--no-deps", "-r", str(root / "requirements/generative-fill.txt")], check=True)
    subprocess.run([str(executable), "-m", "pip", "install", "--ignore-installed",
                    "--no-deps", "huggingface-hub==0.36.2", "tokenizers==0.22.2",
                    "regex", "tqdm", "requests", "filelock", "packaging", "pyyaml",
                    "psutil", "importlib-metadata"], check=True)
    subprocess.run([str(executable), "-c",
                    ("import torch, diffusers, transformers, LanPaint, lanpaint_pipeline, cv2; "
                    "print('Generative Fill runtime:', torch.__version__, diffusers.__version__, "
                    "transformers.__version__)" )], check=True)
    if args.download_models:
        sys.path.insert(0, str(root / "src"))
        from kyven.models.catalog import ModelCatalog
        ModelCatalog.builtin().download("flux2-klein-4b-q4-lanpaint", root / "models",
                                       progress=lambda n, total: print(f"{n}/{total}", flush=True))
    print("Generative Fill runtime is ready.")


if __name__ == "__main__":
    main()
