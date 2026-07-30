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


if __name__ == "__main__":
    unittest.main()
