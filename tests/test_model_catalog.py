from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kyven.models.catalog import ModelCatalog


class ModelCatalogTests(unittest.TestCase):
    def test_builtin_catalog_exposes_all_sam21_sizes(self) -> None:
        models = ModelCatalog.builtin().list("segment")
        self.assertEqual(
            tuple(model.model_id for model in models),
            ("sam2.1-tiny", "sam2.1-small", "sam2.1-base-plus", "sam2.1-large"),
        )
        self.assertTrue(all(len(model.sha256) == 64 for model in models))
        self.assertTrue(all(model.commercial_use for model in models))

    def test_snapshot_reports_installation_and_vram_advice(self) -> None:
        model = ModelCatalog.builtin().get("sam2.1-small")
        with tempfile.TemporaryDirectory() as directory:
            snapshot = model.snapshot(Path(directory), available_vram_mb=4096)
        self.assertFalse(snapshot["installed"])
        self.assertFalse(snapshot["compatible"])

    def test_refine_catalog_is_commercial_and_low_memory_capable(self) -> None:
        models = ModelCatalog.builtin().list("refine")
        self.assertEqual(
            [model.model_id for model in models],
            ["vitmatte-small-composition-1k", "vitmatte-base-distinctions-646"],
        )
        self.assertTrue(models[0].commercial_use)
        self.assertTrue(models[0].supports_cpu)
        self.assertEqual(models[0].recommended_vram_mb, 4096)
        self.assertEqual(models[1].parameters_millions, 96.7)
        self.assertEqual(models[1].recommended_vram_mb, 8192)
        self.assertTrue(models[1].snapshot(Path("models"), 8188)["compatible"])

    def test_vitmatte_provider_is_lazy_and_declares_license(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = ModelCatalog.builtin().registry(Path(directory), "cpu")
            provider = registry.get("vitmatte-small-composition-1k")
            capabilities = provider.capabilities
        self.assertEqual(capabilities.license_name, "Apache-2.0")
        self.assertTrue(capabilities.supports_cpu)
        self.assertTrue(capabilities.supports_tiling)

    def test_inpaint_catalog_offers_fast_and_native_lama(self) -> None:
        models = ModelCatalog.builtin().list("inpaint")
        self.assertEqual(
            [model.model_id for model in models],
            ["lama-2025jan-onnx", "big-lama-native"],
        )
        self.assertTrue(all(model.commercial_use for model in models))
        self.assertEqual(models[1].size_bytes, 205669692)


if __name__ == "__main__":
    unittest.main()
