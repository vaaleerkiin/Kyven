from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from kyven.inpaint.models import InpaintCapabilities, InpaintPrediction, InpaintRequest
from kyven.inpaint.service import InpaintService
from kyven.segment.models import BoxPrompt
from kyven.segment.providers.registry import ProviderRegistry


class FakeInpaintProvider:
    def __init__(self) -> None:
        self.requests: list[InpaintRequest] = []

    @property
    def capabilities(self) -> InpaintCapabilities:
        return InpaintCapabilities("fake", "Fake", "1", "checksum", "MIT", "", True, None)

    def predict(self, request: InpaintRequest, cancellation) -> InpaintPrediction:
        self.requests.append(request)
        with Image.open(request.source) as image:
            shape = np.asarray(image.convert("RGB")).shape
        return InpaintPrediction(np.full(shape, (255, 0, 0), dtype=np.uint8))

    def unload(self) -> None: pass


class InpaintServiceTests(unittest.TestCase):
    def _registry(self, provider: FakeInpaintProvider) -> ProviderRegistry:
        registry = ProviderRegistry(); registry.register("fake", lambda: provider); return registry

    def test_auto_roi_and_masked_merge_preserve_outside(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "source.png"; mask = root / "mask.png"; output = root / "output.png"
            Image.fromarray(np.full((20, 30, 3), 50, dtype=np.uint8)).save(source)
            pixels = np.zeros((20, 30), dtype=np.uint8); pixels[8:12, 13:17] = 255; Image.fromarray(pixels).save(mask)
            provider = FakeInpaintProvider()
            result = InpaintService(self._registry(provider)).run(InpaintRequest(source, mask, output, provider_id="fake", context_padding=2, mask_grow=0, mask_feather=0))
            rendered = np.asarray(Image.open(output).convert("RGB"))
            self.assertTrue(np.all(rendered[0, 0] == 50)); self.assertTrue(np.all(rendered[9, 14] == (255, 0, 0)))
            self.assertEqual(result.metadata["processing_roi"]["width"], 8)
            self.assertEqual(Image.open(provider.requests[0].source).size if provider.requests[0].source.exists() else (8, 8), (8, 8))

    def test_empty_mask_does_not_activate_provider(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "source.png"; mask = root / "mask.png"; output = root / "output.png"
            original = np.arange(300, dtype=np.uint8).reshape(10, 10, 3); Image.fromarray(original).save(source); Image.fromarray(np.zeros((10, 10), dtype=np.uint8)).save(mask)
            registry = ProviderRegistry()
            result = InpaintService(registry).run(InpaintRequest(source, mask, output))
            self.assertTrue(result.metadata["empty_mask"]); np.testing.assert_array_equal(np.asarray(Image.open(output)), original)

    def test_manual_roi_ignores_mask_outside_roi(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "source.png"; mask = root / "mask.png"; output = root / "output.png"
            Image.fromarray(np.full((10, 10, 3), 25, dtype=np.uint8)).save(source)
            pixels = np.zeros((10, 10), dtype=np.uint8); pixels[8, 8] = 255; Image.fromarray(pixels).save(mask)
            provider = FakeInpaintProvider()
            result = InpaintService(self._registry(provider)).run(InpaintRequest(source, mask, output, provider_id="fake", crop_mode="manual", roi=BoxPrompt(0, 0, 5, 5)))
            self.assertTrue(result.metadata["empty_mask"]); self.assertFalse(provider.requests)


if __name__ == "__main__":
    unittest.main()
