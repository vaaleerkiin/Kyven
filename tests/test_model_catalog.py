from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from kyven.hashing import cached_file_sha256, file_sha256
from kyven.models.catalog import ModelCatalog


class ModelCatalogTests(unittest.TestCase):
    def test_checkpoint_hash_cache_invalidates_when_file_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.bin"
            checkpoint.write_bytes(b"first")
            with mock.patch("kyven.hashing.file_sha256", wraps=file_sha256) as calculate:
                first = cached_file_sha256(checkpoint)
                self.assertEqual(cached_file_sha256(checkpoint), first)
                self.assertEqual(calculate.call_count, 1)
                checkpoint.write_bytes(b"second-longer")
                self.assertNotEqual(cached_file_sha256(checkpoint), first)
                self.assertEqual(calculate.call_count, 2)

    def test_builtin_catalog_exposes_sam21_and_hq_sam2(self) -> None:
        models = ModelCatalog.builtin().list("segment")
        self.assertEqual(
            tuple(model.model_id for model in models),
            (
                "sam2.1-tiny",
                "sam2.1-small",
                "sam2.1-base-plus",
                "sam2.1-large",
                "hq-sam2.1-large",
            ),
        )
        self.assertTrue(all(len(model.sha256) == 64 for model in models))
        self.assertTrue(all(model.commercial_use for model in models))

        hq = models[-1]
        self.assertEqual(hq.provider, "hq-sam2")
        self.assertEqual(hq.runtime, "hq-sam2")
        self.assertEqual(hq.size_bytes, 898844313)
        self.assertEqual(
            hq.sha256,
            "03b8be4b4f282c3d7365715a033918d9ac67cbc597660ccc274448aa1c8cdb05",
        )

    def test_hq_sam2_provider_is_lazy_and_has_hq_capabilities(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = ModelCatalog.builtin().registry(Path(directory), "cpu")
            capabilities = registry.get("hq-sam2.1-large").capabilities

        self.assertEqual(capabilities.model_family, "hq-sam2")
        self.assertEqual(capabilities.minimum_vram_mb, 12288)
        self.assertIn("SysCV/sam-hq", capabilities.license_url)

    def test_bundled_hq_sam2_config_is_discoverable_by_hydra(self) -> None:
        config_root = Path(__file__).parents[1] / "src" / "kyven" / "resources" / "models" / "sam2_configs"
        GlobalHydra.instance().clear()
        try:
            with initialize_config_dir(version_base="1.2", config_dir=str(config_root)):
                config = compose(config_name="sam2.1/sam2.1_hq_hiera_l.yaml")
        finally:
            GlobalHydra.instance().clear()

        self.assertEqual(config.model._target_, "sam2.modeling.sam2_hq_base.SAM2HQBase")

    def test_segment_provider_factory_accepts_catalog_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = ModelCatalog.builtin().registry(Path(directory), "cpu")
            provider = registry.get("sam2.1-small")

        self.assertEqual(provider.capabilities.provider_id, "sam2.1-small")

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
            [
                "vitmatte-small-composition-1k",
                "vitmatte-base-distinctions-646",
                "mematte-vits-aim500",
                "mematte-vitb-composition-1k",
            ],
        )
        self.assertTrue(models[0].commercial_use)
        self.assertTrue(models[0].supports_cpu)
        self.assertEqual(models[0].recommended_vram_mb, 4096)
        self.assertEqual(models[1].parameters_millions, 96.7)
        self.assertEqual(models[1].recommended_vram_mb, 8192)
        self.assertTrue(models[1].snapshot(Path("models"), 8188)["compatible"])
        self.assertEqual(models[2].provider, "mematte")
        self.assertEqual(
            models[2].sha256,
            "538cf871933dd0663a6bc28a4bec9b870cc2de909c7431b857b72ce427e82a09",
        )
        self.assertEqual(models[3].parameters_millions, 110.1)
        self.assertEqual(models[3].recommended_vram_mb, 8192)

    def test_mematte_provider_is_lazy_and_declares_license(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = ModelCatalog.builtin().registry(Path(directory), "cpu")
            capabilities = registry.get("mematte-vits-aim500").capabilities

        self.assertEqual(capabilities.provider_id, "mematte-vits-aim500")
        self.assertEqual(capabilities.license_name, "MIT")
        self.assertTrue(capabilities.supports_tiling)

    def test_vitmatte_provider_is_lazy_and_declares_license(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = ModelCatalog.builtin().registry(Path(directory), "cpu")
            provider = registry.get("vitmatte-small-composition-1k")
            capabilities = provider.capabilities
        self.assertEqual(capabilities.license_name, "Apache-2.0")
        self.assertTrue(capabilities.supports_cpu)
        self.assertTrue(capabilities.supports_tiling)

    def test_vitmatte_base_provider_uses_base_hardware_and_license_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = ModelCatalog.builtin().registry(Path(directory), "cpu")
            capabilities = registry.get("vitmatte-base-distinctions-646").capabilities

        self.assertEqual(capabilities.minimum_vram_mb, 8192)
        self.assertIn("vitmatte-base-distinctions-646", capabilities.license_url)

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
