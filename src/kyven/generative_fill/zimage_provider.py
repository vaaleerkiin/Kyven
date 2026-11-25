"""Thin transport to the isolated native Z-Image ControlNet Inpaint worker."""
import os
from pathlib import Path

from kyven.generative_fill.assets import bundle_ready, manifest_checksum
from kyven.generative_fill.models import ZIMAGE_PROVIDER_ID
from kyven.generative_fill.provider import KleinLanPaintProvider
from kyven.inpaint.models import InpaintCapabilities


class ZImageInpaintProvider(KleinLanPaintProvider):
    auto_stitch_all_edits = True

    def __init__(self, models_dir: Path, device: str = "auto") -> None:
        super().__init__(models_dir, device)
        self.models = models_dir.resolve() / ZIMAGE_PROVIDER_ID
        self.runtime = self.runtime.with_name("zimage-fill")
        self.python = self.runtime / ("venv/Scripts/python.exe" if os.name == "nt" else "venv/bin/python")
        self.worker = self.worker.with_name("zimage_worker.py")

    @property
    def capabilities(self) -> InpaintCapabilities:
        return InpaintCapabilities(
            ZIMAGE_PROVIDER_ID, "Z-Image Base Q4 + Native ControlNet Inpaint (Preview)", "0.2.0",
            manifest_checksum(ZIMAGE_PROVIDER_ID), "Apache-2.0 weights; GPL-3.0 optional worker",
            "https://huggingface.co/alibaba-pai/Z-Image-Fun-Controlnet-Union-2.1", True, 8192)

    def _bundle_ready(self) -> bool:
        return bundle_ready(self.models, verify=True, model_id=ZIMAGE_PROVIDER_ID)
