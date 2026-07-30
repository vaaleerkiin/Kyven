from __future__ import annotations

import sys
import unittest
from pathlib import Path

NUKE_ROOT = Path(__file__).parents[1] / "hosts" / "nuke"
sys.path.insert(0, str(NUKE_ROOT))

from kyven_nuke.node import _nuke_file_path
from kyven_nuke.payload import segment_payload


class NukePayloadTests(unittest.TestCase):
    def test_nuke_file_paths_use_forward_slashes(self) -> None:
        path = _nuke_file_path(Path("D:/Kyven/.runtime/source.1.png"))

        self.assertNotIn("\\\\", path)
        self.assertTrue(path.endswith("/source.1.png"))

    def test_coordinates_and_model_selection_are_translated(self) -> None:
        payload = segment_payload(
            source="C:/cache/source.png",
            output="C:/cache/matte.png",
            model_index=2,
            profile="balanced",
            image_height=1080,
            positive_enabled=True,
            positive_xy=(100, 200),
            negative_enabled=False,
            negative_xy=(0, 0),
            box_enabled=True,
            box=(10, 20, 300, 400),
        )
        self.assertEqual(payload["model_id"], "sam2.1-base-plus")
        self.assertEqual(payload["points"][0]["y"], 880.0)
        self.assertEqual(payload["box"], {"x0": 10.0, "y0": 680.0, "x1": 300.0, "y1": 1060.0})


if __name__ == "__main__":
    unittest.main()
