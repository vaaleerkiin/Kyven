"""Thin host-neutral transport to the isolated experimental MOSS worker."""
import os
from pathlib import Path

from kyven.generative_fill.assets import bundle_ready, manifest_checksum
from kyven.generative_fill.models import MOSS_PROVIDER_ID
from kyven.generative_fill.provider import KleinLanPaintProvider
from kyven.inpaint.models import InpaintCapabilities


class MossLanPaintProvider(KleinLanPaintProvider):
    def __init__(self, models_dir: Path, device: str = "auto") -> None:
        super().__init__(models_dir, device)
        self.models = models_dir.resolve() / MOSS_PROVIDER_ID
        self.runtime = self.runtime.with_name("moss-fill")
        self.python = self.runtime / ("venv/Scripts/python.exe" if os.name == "nt" else "venv/bin/python")
        self.worker = self.worker.with_name("moss_worker.py")

    @property
    def capabilities(self) -> InpaintCapabilities:
        return InpaintCapabilities(
            MOSS_PROVIDER_ID, "MOSS SFT INT8 + Edit + LanPaint (Experimental)", "0.1.2",
            manifest_checksum(MOSS_PROVIDER_ID),
            "Apache-2.0 base; Edit weights license unspecified; GPL-3.0 worker",
            "https://huggingface.co/chinoll/Moss-Image-Edit", True, 8192)

    def _bundle_ready(self) -> bool:
        return bundle_ready(self.models, verify=True, model_id=MOSS_PROVIDER_ID)
