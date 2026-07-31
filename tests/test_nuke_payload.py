from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

NUKE_ROOT = Path(__file__).parents[1] / "hosts" / "nuke"
sys.path.insert(0, str(NUKE_ROOT))

from kyven_nuke.inpaint_node import INPAINT_OUTPUT_MODES
from kyven_nuke.live import affects_live_result
from kyven_nuke.node import (
    OUTPUT_MODES,
    _cache_root_path,
    _nuke_file_path,
    _path_for_frame,
    _place_knob_after,
    _point_knob_names,
    _prompt_defaults,
    _section_markup,
)
from kyven_nuke.payload import (
    inpaint_payload,
    refine_payload,
    roi_box,
    segment_payload,
    segment_video_payload,
)
from kyven_nuke.refine_node import REFINE_OUTPUT_MODES, _trimap_preview_paths
from kyven_nuke.runtime import _server_environment


class NukePayloadTests(unittest.TestCase):
    def test_invalid_or_inverted_roi_is_normalized_to_a_safe_rectangle(self) -> None:
        inverted = roi_box((900, 700, 100, 200), image_height=1080, image_width=1920)
        outside = roi_box((3000, 200, 4000, 800), image_height=1080, image_width=1920)

        self.assertEqual(
            inverted,
            {"x0": 100.0, "y0": 380.0, "x1": 900.0, "y1": 880.0},
        )
        self.assertEqual(
            outside,
            {"x0": 0.0, "y0": 0.0, "x1": 1920.0, "y1": 1080.0},
        )

    def test_trimap_preview_read_name_ends_with_nuke_frame(self) -> None:
        _input, output = _trimap_preview_paths(Path("D:/cache"), frame=67, revision=23)

        self.assertEqual(output.name, "trimap_preview_r23.0067.png")

    def test_live_invalidation_tracks_prompts_roi_and_refine_controls(self) -> None:
        self.assertTrue(affects_live_result("positive_point_3", "segment"))
        self.assertTrue(affects_live_result("prompt_box", "segment"))
        self.assertTrue(affects_live_result("processing_roi", "refine"))
        self.assertFalse(affects_live_result("foreground_radius", "refine"))
        self.assertFalse(affects_live_result("max_hole_area", "segment"))
        self.assertFalse(affects_live_result("output_mode", "segment"))
        self.assertFalse(affects_live_result("output_mode", "refine"))

    def test_refine_payload_translates_roi_and_trimap_controls(self) -> None:
        payload = refine_payload(
            source="D:/source.png",
            mask="D:/mask.png",
            output="D:/alpha.png",
            trimap_output="D:/trimap.png",
            model_index=0,
            profile="low_memory",
            image_width=1920,
            image_height=1080,
            roi_enabled=True,
            roi=(10, 20, 300, 400),
            generate_trimap=True,
            foreground_radius=8,
            background_radius=12,
            tile_size=512,
            tile_overlap=64,
        )
        self.assertEqual(payload["model_id"], "vitmatte-small-composition-1k")
        self.assertEqual(payload["roi"]["y0"], 680.0)
        self.assertEqual(payload["foreground_radius"], 8)
        self.assertTrue(payload["generate_trimap"])
        self.assertEqual(payload["tile_size"], 512)
        self.assertEqual(payload["trimap_output"], "D:/trimap.png")

    def test_section_markup_adds_compact_spacing_and_title(self) -> None:
        markup = _section_markup("OUTPUT")

        self.assertTrue(markup.startswith("<br>"))
        self.assertIn("#9fc7e8", markup)
        self.assertIn("<b>OUTPUT</b>", markup)

    def test_model_manager_is_moved_next_to_model_refresh(self) -> None:
        class Knob:
            def __init__(self, name: str) -> None:
                self._name = name

            def name(self) -> str:
                return self._name

        class Node:
            def __init__(self) -> None:
                self._knobs = {
                    name: Knob(name)
                    for name in (
                        "model_section",
                        "model",
                        "refresh_models",
                        "processing_section",
                        "kyven_status",
                        "open_model_manager",
                    )
                }

            def knobs(self) -> dict[str, Knob]:
                return self._knobs

            def __getitem__(self, name: str) -> Knob:
                return self._knobs[name]

            def removeKnob(self, knob: Knob) -> None:
                self._knobs.pop(knob.name())

            def addKnob(self, knob: Knob) -> None:
                self._knobs[knob.name()] = knob

        node = Node()
        _place_knob_after(node, "open_model_manager", "refresh_models")

        names = list(node.knobs())
        self.assertEqual(names.index("open_model_manager"), names.index("refresh_models") + 1)
        self.assertEqual(names[-1], "kyven_status")

    def test_node_cache_path_rejects_parent_traversal(self) -> None:
        class Knob:
            def value(self) -> str:
                return "../outside"

        class Node:
            def __getitem__(self, name: str) -> Knob:
                self.last_name = name
                return Knob()

        with (
            mock.patch(
                "kyven_nuke.node.config.cache_dir",
                return_value=Path("D:/Kyven/.runtime/nuke_cache"),
            ),
            self.assertRaises(RuntimeError),
        ):
            _cache_root_path(Node())

    def test_output_modes_match_internal_switch_inputs(self) -> None:
        self.assertEqual(
            OUTPUT_MODES,
            ("Matte", "Source + Alpha", "Cutout", "Source (Bypass)"),
        )
        self.assertEqual(
            REFINE_OUTPUT_MODES,
            (
                "Refined Matte",
                "Source + Refined Alpha",
                "Refined Cutout",
                "Trimap",
                "Source + Trimap Alpha",
                "Trimap Cutout",
                "Source (Bypass)",
            ),
        )
        self.assertEqual(
            INPAINT_OUTPUT_MODES,
            ("Result", "Patch", "Processed Mask", "Difference", "Source"),
        )

    def test_inpaint_payload_includes_processed_mask_controls(self) -> None:
        payload = inpaint_payload(
            source="D:/source.tif", mask="D:/mask.png", output="D:/result.png",
            mask_output="D:/processed.png", model_index=0, profile="balanced",
            image_width=1920, image_height=1080, crop_mode="manual",
            roi=(10, 20, 300, 400), context_padding=96, mask_grow=-2,
            blend_grow=1, mask_feather=3.5, edge_color_match=0.75, mask_threshold=0.25,
            invert_mask=True, mask_channel="alpha", processing_size=0,
        )
        self.assertEqual(payload["model_id"], "lama-2025jan-onnx")
        self.assertEqual(payload["mask_output"], "D:/processed.png")
        self.assertEqual(payload["roi"]["y0"], 680.0)
        self.assertEqual(payload["mask_grow"], -2)
        self.assertEqual(payload["blend_grow"], 1)
        self.assertEqual(payload["edge_color_match"], 0.75)
        self.assertEqual(payload["mask_channel"], "alpha")
        self.assertTrue(payload["invert_mask"])

    def test_video_payload_uses_key_frame_and_cpu_offload(self) -> None:
        payload = segment_video_payload(
            frames_dir="D:/cache/frames",
            output_pattern="D:/cache/matte.%04d.png",
            raw_output_pattern="D:/cache/raw_matte.%04d.png",
            model_index=1,
            profile="low_memory",
            image_width=1920,
            image_height=1080,
            positive_points=[(100, 200)],
            negative_points=[],
            box_enabled=False,
            box=(0, 0, 1920, 1080),
            first_frame=1,
            last_frame=100,
            key_frame=50,
            direction="both",
        )

        self.assertEqual(payload["key_frame"], 50)
        self.assertEqual(payload["direction"], "both")
        self.assertEqual(payload["raw_output_pattern"], "D:/cache/raw_matte.%04d.png")
        self.assertTrue(payload["offload_video_to_cpu"])
        self.assertEqual(payload["points"][0]["y"], 880.0)
        self.assertIsNone(payload["box"])
        self.assertIsNone(payload["roi"])
        self.assertTrue(payload["fill_holes"])
        self.assertEqual(payload["max_hole_area"], 2_048)
        self.assertEqual(payload["rois"], [])

    def test_video_payload_serializes_animated_roi_per_frame(self) -> None:
        payload = segment_video_payload(
            frames_dir="D:/cache/frames",
            output_pattern="D:/cache/matte.%04d.png",
            model_index=1,
            profile="balanced",
            image_width=100,
            image_height=100,
            positive_points=[(25, 40)],
            negative_points=[],
            box_enabled=True,
            box=(10, 20, 50, 60),
            first_frame=10,
            last_frame=11,
            key_frame=10,
            direction="forward",
            animated_rois=[
                (10, (10, 20, 50, 60)),
                (11, (20, 25, 70, 65)),
            ],
        )

        self.assertIsNone(payload["roi"])
        self.assertEqual(
            payload["rois"],
            [
                {"frame": 10, "x0": 10.0, "y0": 40.0, "x1": 50.0, "y1": 80.0},
                {"frame": 11, "x0": 20.0, "y0": 35.0, "x1": 70.0, "y1": 75.0},
            ],
        )

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
            {
                "PATH": "C:/Program Files/Nuke16.0v4;keep",
                "Path": "duplicate-that-must-not-survive",
                "SystemRoot": "C:/Windows",
                "PYTHONHOME": "Nuke",
                "PYTHONPATH": "Nuke/python",
                "QT_PLUGIN_PATH": "Nuke/qt",
                "TCL_LIBRARY": "Nuke/tcl",
            },
            Path("D:/Kyven/.venv/Scripts"),
        )

        self.assertEqual(
            environment["PATH"],
            "D:\\Kyven\\.venv\\Scripts;C:\\Windows\\System32;C:/Windows",
        )
        self.assertNotIn("PYTHONHOME", environment)
        self.assertNotIn("PYTHONPATH", environment)
        self.assertNotIn("QT_PLUGIN_PATH", environment)
        self.assertNotIn("TCL_LIBRARY", environment)
        self.assertNotIn("Path", environment)
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
            image_width=1920,
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
        self.assertIsNone(payload["box"])
        self.assertTrue(payload["fill_holes"])
        self.assertEqual(payload["max_hole_area"], 2_048)
        self.assertEqual(
            payload["roi"],
            {"x0": 10.0, "y0": 680.0, "x1": 300.0, "y1": 1060.0},
        )


if __name__ == "__main__":
    unittest.main()
