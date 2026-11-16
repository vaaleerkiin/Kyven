from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

NUKE_ROOT = Path(__file__).parents[1] / "hosts" / "nuke"
sys.path.insert(0, str(NUKE_ROOT))

from kyven_nuke.client import NukeKyvenClient, NukeKyvenClientError
from kyven_nuke.inpaint_node import (
    CATTERY_DEFAULT_INPUT_COLORSPACE,
    CATTERY_MODEL_COLORSPACE,
    INPAINT_OUTPUT_MODES,
    _colorspace_has_error,
    _inpaint_public_pattern,
    _wire_generated_patch,
    _wire_patch_over_source,
)
from kyven_nuke.inpaint_node import (
    MASK_INPUT as INPAINT_MASK_INPUT,
)
from kyven_nuke.inpaint_node import (
    SOURCE_INPUT as INPAINT_SOURCE_INPUT,
)
from kyven_nuke.inpaint_node import (
    _ensure_input_order as ensure_inpaint_input_order,
)
from kyven_nuke.inpaint_node import (
    _paths as inpaint_cache_paths,
)
from kyven_nuke.live import affects_live_result
from kyven_nuke.node import (
    OUTPUT_MODES,
    _begin_node_operation,
    _cache_path_component,
    _cache_paths,
    _cache_patterns,
    _cache_root_path,
    _cached_postprocess_frames,
    _clamped_bbox_values,
    _clear_sequence_frames,
    _ensure_double_slider,
    _ensure_unique_cache_uuid,
    _job_error_text,
    _missing_sequence_frames,
    _nuke_file_path,
    _operation_node,
    _path_for_frame,
    _place_knob_after,
    _point_knob_names,
    _prompt_defaults,
    _section_markup,
    _sequence_frame_range,
    _set_busy,
    _video_cache_paths,
)
from kyven_nuke.payload import (
    inpaint_payload,
    refine_payload,
    roi_box,
    segment_payload,
    segment_video_payload,
)
from kyven_nuke.refine_node import (
    MASK_INPUT as REFINE_MASK_INPUT,
)
from kyven_nuke.refine_node import (
    REFINE_OUTPUT_MODES,
    _refine_public_pattern,
    _trimap_preview_paths,
)
from kyven_nuke.refine_node import (
    SOURCE_INPUT as REFINE_SOURCE_INPUT,
)
from kyven_nuke.refine_node import (
    _cache_paths as refine_cache_paths,
)
from kyven_nuke.refine_node import (
    _cache_patterns as refine_cache_patterns,
)
from kyven_nuke.refine_node import (
    _ensure_input_order as ensure_refine_input_order,
)
from kyven_nuke.refine_node import (
    _payload as refine_node_payload,
)
from kyven_nuke.runtime import (
    HQ_SAM2_RUNTIME_URL,
    _listener_pids,
    _listening_legacy_ports,
    _runtime_install_command,
    _server_environment,
    ensure_model_runtime,
)


class NukePayloadTests(unittest.TestCase):
    def test_refine_payload_reads_animated_roi_at_requested_frame(self) -> None:
        class Knob:
            def __init__(self, value, animated=None):
                self.current = value
                self.animated = animated

            def value(self):
                return self.current

            def valueAt(self, frame, index=None):
                value = self.animated[frame] if self.animated else self.current
                return value if index is None else value[index]

            def getValue(self):
                return self.current

        class Node:
            def __init__(self):
                self.values = {
                    "model": Knob(0),
                    "profile": Knob("balanced"),
                    "roi_enabled": Knob(True),
                    "processing_roi": Knob(
                        (0, 0, 100, 100),
                        animated={1: (10, 20, 110, 120), 2: (30, 40, 230, 240)},
                    ),
                    "generate_trimap": Knob(True),
                    "foreground_radius": Knob(10),
                    "background_radius": Knob(15),
                    "tile_size": Knob(0),
                    "tile_overlap": Knob(64),
                }

            def __getitem__(self, name):
                return self.values[name]

        class Source:
            @staticmethod
            def width():
                return 1000

            @staticmethod
            def height():
                return 500

        payload = refine_node_payload(
            Node(),
            Source(),
            Path("source.png"),
            Path("mask.png"),
            Path("output.png"),
            Path("trimap.png"),
            frame=2,
        )
        self.assertEqual(payload["roi"], {"x0": 30.0, "y0": 260.0, "x1": 230.0, "y1": 460.0})

    def test_active_node_is_resolved_by_uuid_after_rename(self) -> None:
        class Knob:
            def __init__(self, value):
                self.current = value

            def value(self):
                return self.current

            def setValue(self, value):
                self.current = value

        class Node:
            def __init__(self):
                self.name = "BeforeRename"
                self.values = {
                    "cache_uuid": Knob("stable-uuid"),
                    "kyven_busy": Knob(False),
                    "kyven_job_id": Knob("job"),
                }

            def fullName(self):
                return self.name

            def knobs(self):
                return self.values

            def __getitem__(self, name):
                return self.values[name]

        node = Node()
        fake_nuke = mock.Mock()
        fake_nuke.toNode.return_value = None
        fake_nuke.allNodes.return_value = [node]
        self.assertTrue(_begin_node_operation(node))
        node.name = "AfterRename"
        with mock.patch("kyven_nuke.node._nuke", return_value=fake_nuke):
            self.assertIs(_operation_node("BeforeRename"), node)
            _set_busy("BeforeRename", False)
        self.assertFalse(node["kyven_busy"].value())
        self.assertEqual(node["kyven_job_id"].value(), "")

    def test_legacy_server_scan_only_returns_listening_known_ports(self) -> None:
        output = """TCP 127.0.0.1:8765 0.0.0.0:0 LISTENING 10
TCP 127.0.0.1:18788 0.0.0.0:0 LISTENING 11
TCP 127.0.0.1:9999 0.0.0.0:0 LISTENING 12
TCP 127.0.0.1:8766 0.0.0.0:0 ESTABLISHED 13"""
        self.assertEqual(_listening_legacy_ports(output), (8765,))

    def test_node_operation_claim_rejects_duplicate_start(self) -> None:
        class Knob:
            def __init__(self) -> None:
                self.current = False

            def value(self):
                return self.current

            def setValue(self, value):
                self.current = bool(value)

        class Node:
            def __init__(self) -> None:
                self.busy = Knob()

            def knobs(self):
                return {"kyven_busy": self.busy}

            def fullName(self):
                return "TestNodeOperationClaim"

            def __getitem__(self, name):
                return self.knobs()[name]

        node = Node()
        # A persisted busy knob from a crashed/reopened script is stale and must be recoverable.
        node.busy.current = True
        self.assertTrue(_begin_node_operation(node))
        self.assertFalse(_begin_node_operation(node))

    def test_missing_sequence_frames_checks_exact_expected_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pattern = Path(directory) / "frame.%04d.png"
            (Path(directory) / "frame.0001.png").touch()
            (Path(directory) / "unrelated.jpg").touch()
            (Path(directory) / "frame.0003.png").touch()
            self.assertEqual(_missing_sequence_frames(pattern, 1, 3), [2])
            _clear_sequence_frames(pattern, 1, 3)
            self.assertFalse((Path(directory) / "frame.0001.png").exists())
            self.assertFalse((Path(directory) / "frame.0003.png").exists())
            self.assertTrue((Path(directory) / "unrelated.jpg").exists())

    def test_inpaint_matches_cattery_lama_color_defaults(self) -> None:
        self.assertEqual(CATTERY_DEFAULT_INPUT_COLORSPACE, "Linear")
        self.assertEqual(CATTERY_MODEL_COLORSPACE, "sRGB")
        self.assertTrue(_colorspace_has_error("Error: (Linear) not found"))
        self.assertFalse(_colorspace_has_error("Linear"))

    def test_generated_patch_uses_composited_live_source_and_mask(self) -> None:
        class Copy:
            def __init__(self) -> None:
                self.inputs = {}

            def setInput(self, index, node) -> None:
                self.inputs[index] = node

        copy = Copy()
        result = object()
        mask = object()
        _wire_generated_patch(copy, result, mask)

        self.assertIs(copy.inputs[0], result)
        self.assertIs(copy.inputs[1], mask)

    def test_inpaint_result_wires_generated_patch_over_source(self) -> None:
        class Merge:
            def __init__(self) -> None:
                self.inputs = {}

            def setInput(self, index, node) -> None:
                self.inputs[index] = node

        merge = Merge()
        patch = object()
        source = object()
        _wire_patch_over_source(merge, patch, source)

        self.assertIs(merge.inputs[0], source)
        self.assertIs(merge.inputs[1], patch)

    def test_two_input_groups_migrate_source_left_without_swapping_media(self) -> None:
        class Knob:
            def __init__(self, value):
                self.current = value

            def value(self):
                return self.current

            def setValue(self, value):
                self.current = value

        class InputNode:
            def __init__(self, number):
                self.number = Knob(number)

            def __getitem__(self, name):
                self.assert_number(name)
                return self.number

            @staticmethod
            def assert_number(name):
                if name != "number":
                    raise KeyError(name)

        class Group:
            def __init__(self):
                self.connections = ["source-media", "mask-media"]

            def input(self, index):
                return self.connections[index]

            def setInput(self, index, value):
                self.connections[index] = value

            def begin(self):
                pass

            def end(self):
                pass

        for module, ensure, mask_name in (
            ("kyven_nuke.inpaint_node", ensure_inpaint_input_order, "Mask"),
            ("kyven_nuke.refine_node", ensure_refine_input_order, "MaskOrTrimap"),
        ):
            source = InputNode(0)
            mask = InputNode(1)
            fake_nuke = mock.Mock()
            fake_nuke.toNode.side_effect = lambda name, source=source, mask_name=mask_name, mask=mask: {
                "Source": source,
                mask_name: mask,
            }.get(name)
            group = Group()
            with mock.patch(f"{module}._nuke", return_value=fake_nuke):
                ensure(group)
            self.assertEqual(group.connections, ["mask-media", "source-media"])
            self.assertEqual(source.number.value(), 1)
            self.assertEqual(mask.number.value(), 0)

        self.assertEqual(INPAINT_SOURCE_INPUT, REFINE_SOURCE_INPUT)
        self.assertEqual(INPAINT_MASK_INPUT, REFINE_MASK_INPUT)

    def test_nuke_client_wait_retries_transient_poll_timeout(self) -> None:
        client = NukeKyvenClient("http://127.0.0.1:1", "token")
        client.job = mock.Mock(
            side_effect=[
                NukeKyvenClientError("temporary timeout"),
                {"status": "running"},
                {"status": "succeeded"},
            ]
        )

        result = client.wait("job", poll_seconds=0, timeout_seconds=1)

        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(client.job.call_count, 3)

    def test_nuke_client_wait_sends_cancellation_once(self) -> None:
        client = NukeKyvenClient("http://127.0.0.1:1", "token")
        client.cancel = mock.Mock()
        client.job = mock.Mock(
            side_effect=[
                {"status": "running"},
                {"status": "cancelled"},
            ]
        )

        result = client.wait(
            "job",
            poll_seconds=0,
            timeout_seconds=1,
            cancel_requested=lambda: True,
        )

        self.assertEqual(result["status"], "cancelled")
        client.cancel.assert_called_once_with("job")

    def test_int_control_is_upgraded_to_animated_double_slider_in_place(self) -> None:
        class Knob:
            def __init__(self, name: str, label: str = "", value: float = 0) -> None:
                self._name = name
                self._label = label
                self._value = value

            def name(self) -> str:
                return self._name

            def value(self) -> float:
                return self._value

            def toScript(self) -> str:
                return str(self._value)

        class DoubleKnob(Knob):
            def setLabel(self, label: str) -> None:
                self._label = label

            def setRange(self, minimum: float, maximum: float) -> None:
                self.range = (minimum, maximum)

            def setFlag(self, flag: int) -> None:
                self.flag = flag

            def setValue(self, value: float) -> None:
                self._value = value

            def fromScript(self, value: str) -> None:
                self._value = float(value)

        class FakeNuke:
            STARTLINE = 1
            Double_Knob = DoubleKnob

        class Node:
            def __init__(self) -> None:
                self._knobs = {
                    "mask_section": Knob("mask_section"),
                    "mask_grow": Knob("mask_grow", value=23),
                    "next_section": Knob("next_section"),
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
        with mock.patch("kyven_nuke.node._nuke", return_value=FakeNuke()):
            slider = _ensure_double_slider(node, "mask_grow", "Grow", -128, 128, 12)

        self.assertIsInstance(slider, DoubleKnob)
        self.assertEqual(slider.value(), 23)
        self.assertEqual(slider.range, (-128, 128))
        self.assertEqual(list(node.knobs()), ["mask_section", "mask_grow", "next_section"])

    def test_listener_pid_parser_uses_only_exact_listening_port(self) -> None:
        output = """
          TCP    127.0.0.1:18788    0.0.0.0:0    LISTENING    17020
          TCP    127.0.0.1:18777    0.0.0.0:0    LISTENING    999
          TCP    127.0.0.1:18788    127.0.0.1:50000    TIME_WAIT    0
        """

        self.assertEqual(_listener_pids(output), (17020,))

    def test_worker_error_includes_technical_detail(self) -> None:
        message = _job_error_text(
            {
                "status": "failed",
                "error": {
                    "message": "The Kyven worker failed unexpectedly.",
                    "technical_detail": "unexpected keyword argument 'license_url'",
                },
            }
        )

        self.assertIn("unexpected keyword argument", message)

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

    def test_visible_roi_is_clamped_to_canvas_without_negative_values(self) -> None:
        self.assertEqual(
            _clamped_bbox_values((-50.0, -20.0, 2500.0, 1400.0), 1920.0, 1080.0),
            (0.0, 0.0, 1920.0, 1080.0),
        )
        self.assertEqual(
            _clamped_bbox_values((3000.0, 1200.0, 4000.0, 1600.0), 1920.0, 1080.0),
            (1919.0, 1079.0, 1920.0, 1080.0),
        )

    def test_mask_postprocess_discovers_independent_and_tracking_sequences(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "raw_matte.0001.png").write_bytes(b"raw")
            (root / "raw_matte.0002.png").write_bytes(b"raw")
            (root / "raw_tracked_matte.0042.png").write_bytes(b"raw")
            pairs = _cached_postprocess_frames(root)

        self.assertEqual(
            [(source.name, output.name) for source, output in pairs],
            [
                ("raw_matte.0001.png", "matte.0001.png"),
                ("raw_matte.0002.png", "matte.0002.png"),
                ("raw_tracked_matte.0042.png", "tracked_matte.0042.png"),
            ],
        )

    def test_trimap_preview_read_name_ends_with_nuke_frame(self) -> None:
        _input, output = _trimap_preview_paths(Path("D:/cache"), frame=67, revision=23)

        self.assertEqual(output.name, "trimap_preview_r23.0067.png")

    def test_live_invalidation_tracks_prompts_roi_and_refine_controls(self) -> None:
        self.assertTrue(affects_live_result("positive_point_3", "segment"))
        self.assertTrue(affects_live_result("prompt_box", "segment"))
        self.assertTrue(affects_live_result("processing_roi", "refine"))
        self.assertTrue(affects_live_result("in_colorspace", "inpaint"))
        self.assertFalse(affects_live_result("input_color_space", "inpaint"))
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

    def test_refine_auto_crop_payload_uses_context_without_manual_roi(self) -> None:
        payload = refine_payload(
            source="D:/source.png",
            mask="D:/mask.png",
            output="D:/alpha.png",
            trimap_output="D:/trimap.png",
            model_index=0,
            profile="balanced",
            image_width=1920,
            image_height=1080,
            roi_enabled=False,
            roi=(10, 20, 300, 400),
            generate_trimap=True,
            foreground_radius=10,
            background_radius=15,
            crop_mode="auto",
            context_padding=192,
        )

        self.assertEqual(payload["crop_mode"], "auto")
        self.assertEqual(payload["context_padding"], 192)
        self.assertIsNone(payload["roi"])

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

    def test_artist_cache_path_uses_folder_roto_name_and_version_without_uuid(self) -> None:
        class Knob:
            def __init__(self, value: str) -> None:
                self._value = value

            def value(self) -> str:
                return self._value

        class Node:
            def __init__(self, root: str) -> None:
                self._knobs = {
                    "cache_root": Knob(root),
                    "roto_name": Knob("hero_roto"),
                    "cache_version": Knob("v003"),
                    "kyven_uuid": Knob("ignored-uuid"),
                }

            def knobs(self):
                return self._knobs

            def __getitem__(self, name: str):
                return self._knobs[name]

        with tempfile.TemporaryDirectory() as directory:
            path = _cache_root_path(Node(directory))

        self.assertEqual(path.parts[-2:], ("hero_roto", "v003"))
        self.assertNotIn("ignored-uuid", path.parts)

    def test_artist_cache_keeps_only_named_matte_sequence_in_public_root(self) -> None:
        class Knob:
            def __init__(self, value: str) -> None:
                self._value = value

            def value(self) -> str:
                return self._value

        class Node:
            def __init__(self, root: str) -> None:
                self._knobs = {
                    "cache_root": Knob(root),
                    "roto_name": Knob("phone"),
                    "cache_version": Knob("v001"),
                }

            def knobs(self):
                return self._knobs

            def __getitem__(self, name: str):
                return self._knobs[name]

        with tempfile.TemporaryDirectory() as directory:
            node = Node(directory)
            source, raw, matte = _cache_paths(node, 1053)
            source_pattern, _raw_pattern, matte_pattern = _cache_patterns(node)
            frames, _video_source, video_raw, video_matte = _video_cache_paths(
                node, 1001, 1074
            )

            self.assertEqual(matte.name, "phone_v001.1053.png")
            self.assertEqual(matte_pattern.name, "phone_v001.%04d.png")
            self.assertEqual(video_matte, matte_pattern)
            self.assertEqual(source.suffix, ".jpg")
            self.assertEqual(source_pattern, _video_source)
            self.assertEqual(source.parent.name, "tracking_pool")
            self.assertEqual(raw.parent.name, "_kyven_work")
            self.assertEqual(source_pattern.parent.name, "tracking_pool")
            self.assertEqual(video_raw.parent.name, "_kyven_work")
            self.assertEqual(frames.parent.name, "source")
            self.assertEqual(frames.name, "tracking_pool")

    def test_refine_artist_cache_uses_named_public_sequence_and_work_inputs(self) -> None:
        node = self._artist_cache_node("refine_name", "hair")
        with tempfile.TemporaryDirectory() as directory:
            node["cache_root"].setValue(directory)
            source, mask, output, trimap = refine_cache_paths(node, 1005)
            _source_pattern, _mask_pattern, output_pattern, _trimap_pattern = (
                refine_cache_patterns(node)
            )

            self.assertEqual(output.name, "hair_v001.1005.png")
            self.assertEqual(output_pattern.name, "hair_v001.%04d.png")
            self.assertEqual(_refine_public_pattern(node), output_pattern)
            self.assertEqual(source.suffix, ".tif")
            self.assertEqual(mask.suffix, ".tif")
            self.assertEqual(source.parent.name, "source")
            self.assertEqual(mask.parent.name, "_kyven_work")
            self.assertEqual(trimap.parent.name, "_kyven_work")

    def test_inpaint_artist_cache_uses_named_public_sequence_and_work_inputs(self) -> None:
        node = self._artist_cache_node("inpaint_name", "cleanup")
        with tempfile.TemporaryDirectory() as directory:
            node["cache_root"].setValue(directory)
            source, mask, model_mask, output, processed, patch = inpaint_cache_paths(
                node, 1012
            )

            self.assertEqual(output.name, "cleanup_v001.1012.png")
            self.assertEqual(_inpaint_public_pattern(node).name, "cleanup_v001.%04d.png")
            self.assertEqual({source.suffix, mask.suffix, model_mask.suffix}, {".tif"})
            self.assertEqual(source.parent.name, "source")
            for internal in (mask, model_mask, processed, patch):
                self.assertEqual(internal.parent.name, "_kyven_work")

    @staticmethod
    def _artist_cache_node(name_field: str, name_value: str):
        class Knob:
            def __init__(self, value: str) -> None:
                self._value = value

            def value(self) -> str:
                return self._value

            def setValue(self, value: str) -> None:
                self._value = value

        class Node:
            def __init__(self) -> None:
                self._knobs = {
                    "cache_root": Knob(""),
                    name_field: Knob(name_value),
                    "cache_version": Knob("v001"),
                }

            def knobs(self):
                return self._knobs

            def __getitem__(self, name: str):
                return self._knobs[name]

        return Node()

    def test_public_sequence_range_is_discovered_from_named_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pattern = root / "hand_v012.%04d.png"
            (root / "hand_v012.1001.png").write_bytes(b"mask")
            (root / "hand_v012.1048.png").write_bytes(b"mask")
            (root / "different_v012.2000.png").write_bytes(b"other")

            self.assertEqual(_sequence_frame_range(pattern), (1001, 1048))

    def test_artist_cache_components_reject_nested_or_parent_paths(self) -> None:
        for value in ("../shared", "shot/roto", r"shot\roto", ".."):
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                _cache_path_component(value, "Roto Name")

    def test_copied_kyven_node_receives_an_isolated_cache_uuid(self) -> None:
        class Knob:
            def __init__(self, value: str = "") -> None:
                self._value = value

            def value(self) -> str:
                return self._value

            def setValue(self, value: str) -> None:
                self._value = value

        class Node:
            def __init__(self, name: str, node_id: str) -> None:
                self._name = name
                self._knobs = {"kyven_uuid": Knob(node_id)}

            def fullName(self) -> str:
                return self._name

            def knobs(self) -> dict[str, Knob]:
                return self._knobs

            def __getitem__(self, name: str) -> Knob:
                return self._knobs[name]

        original = Node("KyvenInpaint", "shared")
        copied = Node("KyvenInpaint1", "shared")
        fake_nuke = mock.Mock()
        fake_nuke.allNodes.return_value = [original, copied]
        with (
            mock.patch("kyven_nuke.node._nuke", return_value=fake_nuke),
            mock.patch("kyven_nuke.node._disconnect_cached_matte") as disconnect,
        ):
            changed = _ensure_unique_cache_uuid(copied)

        self.assertTrue(changed)
        self.assertEqual(original["kyven_uuid"].value(), "shared")
        self.assertNotEqual(copied["kyven_uuid"].value(), "shared")
        disconnect.assert_called_once_with(copied)

    def test_output_modes_match_internal_switch_inputs(self) -> None:
        self.assertEqual(
            OUTPUT_MODES,
            (
                "Source + Alpha",
                "Alpha",
                "Cutout",
                "Source + Edge Guide",
                "Edge Guide",
                "Bypass",
            ),
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
            (
                "Result",
                "Result + Mask Alpha",
                "Result Premult",
                "Generated Patch",
                "Difference",
                "Source",
            ),
        )

    def test_inpaint_payload_includes_processed_mask_controls(self) -> None:
        payload = inpaint_payload(
            source="D:/source.tif", mask="D:/mask.png", output="D:/result.png",
            model_mask="D:/model-mask.png",
            mask_output="D:/processed.png", patch_output="D:/patch.png",
            model_index=0, profile="balanced",
            image_width=1920, image_height=1080, crop_mode="manual",
            roi=(10, 20, 300, 400), context_padding=96, mask_grow=-2,
            edge_color_match=0.75, edge_softness=9, mask_threshold=0.25,
            invert_mask=True, mask_channel="alpha", processing_size=0,
        )
        self.assertEqual(payload["model_id"], "lama-2025jan-onnx")
        self.assertEqual(payload["mask_output"], "D:/processed.png")
        self.assertEqual(payload["model_mask"], "D:/model-mask.png")
        self.assertEqual(payload["patch_output"], "D:/patch.png")
        self.assertEqual(payload["roi"]["y0"], 680.0)
        self.assertEqual(payload["mask_grow"], -2)
        self.assertNotIn("blend_grow", payload)
        self.assertNotIn("mask_feather", payload)
        self.assertEqual(payload["edge_color_match"], 0.75)
        self.assertEqual(payload["edge_softness"], 9.0)
        self.assertEqual(payload["mask_channel"], "alpha")
        self.assertTrue(payload["invert_mask"])
        self.assertTrue(payload["preprocess_mask"])
        self.assertEqual(payload["quality_mode"], "standard")
        self.assertEqual(payload["refinement_steps"], 15)
        self.assertEqual(payload["refinement_strength"], 1.0)
        self.assertEqual(payload["refinement_scales"], 3)

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
            polish_key_frame=True,
        )

        self.assertEqual(payload["key_frame"], 50)
        self.assertEqual(payload["direction"], "both")
        self.assertTrue(payload["polish_key_frame"])
        self.assertEqual(payload["raw_output_pattern"], "D:/cache/raw_matte.%04d.png")
        self.assertTrue(payload["offload_video_to_cpu"])
        self.assertEqual(payload["points"][0]["y"], 880.0)
        self.assertIsNone(payload["box"])
        self.assertIsNone(payload["roi"])
        self.assertTrue(payload["fill_holes"])
        self.assertEqual(payload["max_hole_area"], 2_048)
        self.assertEqual(payload["rois"], [])

    def test_video_payload_serializes_multiple_correction_frames(self) -> None:
        payload = segment_video_payload(
            frames_dir="D:/cache/frames",
            output_pattern="D:/cache/matte.%04d.png",
            model_index=1,
            profile="balanced",
            image_width=100,
            image_height=80,
            positive_points=[(10, 10)],
            negative_points=[],
            box_enabled=False,
            box=(0, 0, 100, 80),
            first_frame=1,
            last_frame=20,
            key_frame=1,
            direction="both",
            corrections=(
                {"frame": 1, "positive_points": [(10, 10)], "negative_points": []},
                {"frame": 12, "positive_points": [(30, 20)], "negative_points": [(50, 40)]},
            ),
        )

        self.assertEqual([item["frame"] for item in payload["corrections"]], [1, 12])
        self.assertEqual(payload["corrections"][1]["points"][0]["y"], 60.0)
        self.assertEqual(payload["corrections"][1]["points"][1]["label"], "negative")

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

    def test_hq_sam2_runtime_install_is_pinned_and_isolated(self) -> None:
        executable = Path("D:/Kyven/.venv/Scripts/python.exe")
        command = _runtime_install_command("hq-sam2", executable)

        self.assertEqual(command[0], str(executable))
        self.assertIn("--no-deps", command)
        self.assertIn("--force-reinstall", command)
        self.assertEqual(command[-1], HQ_SAM2_RUNTIME_URL)

    def test_missing_catalog_runtime_is_installed_and_verified(self) -> None:
        process = mock.Mock(returncode=0)
        process.communicate.return_value = ("installed", None)
        probe_missing = mock.Mock(returncode=1)
        probe_ready = mock.Mock(returncode=0)
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "python.exe"
            executable.touch()
            with (
                mock.patch(
                    "kyven_nuke.runtime.config.python_executable", return_value=executable
                ),
                mock.patch("kyven_nuke.runtime.config.root", return_value=Path(directory)),
                mock.patch(
                    "kyven_nuke.runtime.subprocess.run",
                    side_effect=(probe_missing, probe_ready),
                ),
                mock.patch("kyven_nuke.runtime.subprocess.Popen", return_value=process) as popen,
            ):
                changed = ensure_model_runtime("hq-sam2")

        self.assertTrue(changed)
        self.assertEqual(popen.call_args.args[0][-1], HQ_SAM2_RUNTIME_URL)

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
