from __future__ import annotations

import sys
import unittest
from pathlib import Path

NUKE_ROOT = Path(__file__).parents[1] / "hosts" / "nuke"
sys.path.insert(0, str(NUKE_ROOT))

from kyven_nuke.node import (
    _nuke_file_path,
    _path_for_frame,
    _point_knob_names,
    _prompt_defaults,
)
from kyven_nuke.payload import segment_payload
from kyven_nuke.runtime import _server_environment


class NukePayloadTests(unittest.TestCase):
    def test_prompt_defaults_use_input_dimensions(self) -> None:
        center, box = _prompt_defaults(2048.0, 858.0)

        self.assertEqual(center, [1024.0, 429.0])
        self.assertEqual(box, [0.0, 0.0, 2048.0, 858.0])

    def test_dynamic_point_knob_names_are_stable(self) -> None:
        self.assertEqual(
            _point_knob_names("positive", 3),
            ["positive_point", "positive_point_2", "positive_point_3"],
        )

    def test_frame_pattern_resolves_to_zero_padded_path(self) -> None:
        path = _path_for_frame(Path("D:/cache/matte.%04d.png"), 12)

        self.assertEqual(path.name, "matte.0012.png")

    def test_server_environment_removes_nuke_python_overrides(self) -> None:
        environment = _server_environment(
            {"PATH": "keep", "PYTHONHOME": "Nuke", "PYTHONPATH": "Nuke/python"}
        )

        self.assertEqual(environment["PATH"], "keep")
        self.assertNotIn("PYTHONHOME", environment)
        self.assertNotIn("PYTHONPATH", environment)
        self.assertEqual(environment["PYTHONNOUSERSITE"], "1")

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
            positive_points=[(100, 200), (150, 250)],
            negative_points=[(500, 600)],
            box_enabled=True,
            box=(10, 20, 300, 400),
        )
        self.assertEqual(payload["model_id"], "sam2.1-base-plus")
        self.assertEqual(payload["points"][0]["y"], 880.0)
        self.assertEqual(
            [item["label"] for item in payload["points"]],
            ["positive", "positive", "negative"],
        )
        self.assertEqual(payload["box"], {"x0": 10.0, "y0": 680.0, "x1": 300.0, "y1": 1060.0})


if __name__ == "__main__":
    unittest.main()
