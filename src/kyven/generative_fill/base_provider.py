"""Transport for the optional undistilled Klein Base worker."""
from kyven.generative_fill.assets import bundle_ready, manifest_checksum
from kyven.generative_fill.models import BASE_PROVIDER_ID
from kyven.generative_fill.provider import KleinLanPaintProvider
from kyven.inpaint.models import InpaintCapabilities


class KleinBaseProvider(KleinLanPaintProvider):
    auto_stitch_all_edits = True

    def __init__(self, models_dir, device="auto"):
        super().__init__(models_dir, device)
        self.models = models_dir.resolve() / BASE_PROVIDER_ID

    @property
    def capabilities(self):
        return InpaintCapabilities(BASE_PROVIDER_ID, "FLUX.2 Klein 4B Base Q4 + LanPaint", "0.1.1",
            manifest_checksum(BASE_PROVIDER_ID), "Apache-2.0 weights; GPL-3.0 optional worker",
            "https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B", True, 8192)

    def _bundle_ready(self):
        return bundle_ready(self.models, verify=True, model_id=BASE_PROVIDER_ID)
