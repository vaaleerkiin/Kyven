"""Transport to the optional MIT Mage worker."""

import os
import subprocess
from dataclasses import asdict

import numpy as np
from PIL import Image

from kyven.generative_fill.assets import bundle_ready, manifest_checksum
from kyven.generative_fill.models import MAGE_PROVIDER_ID
from kyven.generative_fill.provider import KleinLanPaintProvider
from kyven.inpaint.models import InpaintCapabilities


class MageProvider(KleinLanPaintProvider):
    auto_stitch_all_edits = True

    def __init__(self, models_dir, device="auto"):
        super().__init__(models_dir, device)
        self.models = models_dir.resolve() / MAGE_PROVIDER_ID
        self.runtime = self.runtime.with_name("mage-fill")
        self.python = self.runtime / (
            "venv/Scripts/python.exe" if os.name == "nt" else "venv/bin/python"
        )
        self.worker = self.worker.with_name("mage_worker.py")
        self._settings = {}

    @property
    def capabilities(self):
        return InpaintCapabilities(
            MAGE_PROVIDER_ID,
            "Mage-Flow Edit Turbo Q6",
            "1.4.0",
            manifest_checksum(MAGE_PROVIDER_ID),
            "MIT; Apache-2.0 encoder; NVIDIA CUDA redistributables",
            "https://github.com/microsoft/Mage/blob/main/LICENSE",
            True,
            8192,
        )

    def _bundle_ready(self):
        return bundle_ready(self.models, verify=True, model_id=MAGE_PROVIDER_ID)

    def predict(self, request, cancellation):
        self._settings = asdict(request.fill)
        return super().predict(request, cancellation)

    def unload(self):
        # Cancelling the IPC worker must also stop its active CUDA/LLM child.
        process = self._process
        if os.name == "nt" and process is not None and process.poll() is None:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW,
                check=False,
            )
        super().unload()

    def stitch(self, generated, source, mask, cancellation):
        if self._settings.get("seam_repair", "off") == "off":
            cancellation.raise_if_cancelled()
            return generated, {"applied": False, "mode": "off"}
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory(prefix="kyven-mage-boundary-") as directory:
            root = Path(directory)
            for name, pixels in (("generated", generated), ("source", source), ("mask", mask)):
                Image.fromarray(pixels).save(root / (name + ".png"))
            response = self._exchange(
                {
                    "action": "stitch",
                    "generated": str(root / "generated.png"),
                    "source": str(root / "source.png"),
                    "mask": str(root / "mask.png"),
                    "output": str(root / "output.png"),
                    "settings": self._settings,
                },
                cancellation,
                self._settings.get("timeout_seconds", 1800),
            )
            with Image.open(root / "output.png") as image:
                return np.asarray(image.convert("RGB")).copy(), response.get("metadata", {})
