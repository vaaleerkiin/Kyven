"""Install a private Mage transport environment; model files use Model Manager."""

import argparse
import os
import subprocess
import sys
import venv
from pathlib import Path

PROBE = "import numpy, PIL, cv2, scipy; from scipy.sparse.linalg import splu; print('Mage runtime ready')"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--download-models", action="store_true")
    args = parser.parse_args()
    if args.check:
        if (Path(sys.prefix) / "Lib/site-packages/kyven_core.pth").exists():
            raise SystemExit("Legacy shared Mage runtime detected. Use Install / Repair Model.")
        subprocess.run([sys.executable, "-I", "-c", PROBE], check=True)
        return
    if os.name != "nt":
        raise SystemExit("The pinned Mage runtime currently supports Windows x64.")
    root = Path(__file__).resolve().parents[1]
    environment = root / ".runtime/mage-fill/venv"
    venv.EnvBuilder(with_pip=True).create(environment)
    python = environment / "Scripts/python.exe"
    site = environment / "Lib/site-packages"
    # The worker adds its own source path. Share no core site-packages or
    # distribution metadata with this minimal environment (including upgrades).
    (site / "kyven_core.pth").unlink(missing_ok=True)
    subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "numpy>=1.26,<3",
            "Pillow>=10,<13",
            "opencv-python-headless>=4.10,<5",
            "scipy>=1.14,<2",
        ],
        check=True,
    )
    subprocess.run([str(python), "-I", "-c", PROBE], check=True)
    if args.download_models:
        sys.path.insert(0, str(root / "src"))
        from kyven.models.catalog import ModelCatalog

        ModelCatalog.builtin().download(
            "mage-flow-edit-turbo-q6",
            root / "models",
            progress=lambda n, total: print(f"{n}/{total}", flush=True),
        )


if __name__ == "__main__":
    main()
