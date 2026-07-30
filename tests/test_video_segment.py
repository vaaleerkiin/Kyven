from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from kyven.cancellation import CancellationToken
from kyven.errors import KyvenError
from kyven.segment.models import BoxPrompt, ExecutionProfile, PointPrompt, ProviderCapabilities
from kyven.segment.output import write_mask_png_atomic
from kyven.segment.providers.base import SegmentationProvider
from kyven.segment.providers.registry import ProviderRegistry
from kyven.segment.video import (
    VideoDirection,
    VideoSegmentRequest,
    VideoSegmentResult,
    VideoSegmentService,
)


class VideoSegmentRequestTests(unittest.TestCase):
    def test_key_frame_points_allow_later_animated_roi_to_move_away(self) -> None:
        class AnimatedRoiProvider(SegmentationProvider):
            @property
            def capabilities(self) -> ProviderCapabilities:
                return ProviderCapabilities(
                    provider_id="animated-roi-test",
                    display_name="Animated ROI Test",
                    provider_version="1",
                    model_family="test",
                    model_variant="test",
                    model_checksum="test",
                    license_name="CC0-1.0",
                    license_url="https://creativecommons.org/publicdomain/zero/1.0/",
                    supports_cpu=True,
                    supports_points=True,
                    supports_boxes=True,
                    minimum_vram_mb=0,
                    supported_profiles=(ExecutionProfile.BALANCED,),
                )

            def predict(self, request, cancellation):
                raise NotImplementedError

            def propagate_video(self, request, cancellation):
                outputs = []
                with Image.open(min(request.frames_dir.glob("*.jpg"))) as image:
                    size = (image.height, image.width)
                for index in range(request.last_frame - request.first_frame + 1):
                    output = request.output_for_index(index)
                    write_mask_png_atomic(output, np.ones(size, dtype=np.bool_))
                    outputs.append(output)
                return VideoSegmentResult(
                    outputs=tuple(outputs),
                    first_frame=request.first_frame,
                    last_frame=request.last_frame,
                    key_frame=request.key_frame,
                    direction=request.direction,
                    metadata={},
                )

            def unload(self) -> None:
                return

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = root / "frames"
            frames.mkdir()
            Image.new("RGB", (8, 6), "white").save(frames / "00001.jpg")
            Image.new("RGB", (8, 6), "white").save(frames / "00002.jpg")
            registry = ProviderRegistry()
            registry.register("animated-roi-test", AnimatedRoiProvider)
            request = VideoSegmentRequest(
                frames_dir=frames,
                output_pattern=root / "matte.%04d.png",
                first_frame=1,
                last_frame=2,
                key_frame=1,
                direction=VideoDirection.FORWARD,
                points=(PointPrompt(2, 2),),
                rois=(
                    (1, BoxPrompt(1, 1, 4, 4)),
                    (2, BoxPrompt(4, 2, 8, 6)),
                ),
                provider_id="animated-roi-test",
                fill_holes=False,
            )

            result = VideoSegmentService(registry).run(request)

            self.assertTrue(result.metadata["animated_processing_roi"])
            with Image.open(root / "matte.0001.png") as first_mask:
                self.assertEqual(first_mask.size, (8, 6))
                self.assertEqual(first_mask.getpixel((1, 1)), 255)
                self.assertEqual(first_mask.getpixel((0, 0)), 0)
            with Image.open(root / "matte.0002.png") as second_mask:
                self.assertEqual(second_mask.getpixel((4, 2)), 255)
                self.assertEqual(second_mask.getpixel((3, 2)), 0)

    def test_video_outputs_receive_hole_postprocess(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "matte.0001.png"
            mask = np.ones((5, 5), dtype=np.bool_)
            mask[2, 2] = False
            write_mask_png_atomic(output, mask)
            request = VideoSegmentRequest(
                frames_dir=root,
                output_pattern=root / "matte.%04d.png",
                first_frame=1,
                last_frame=1,
                key_frame=1,
                direction=VideoDirection.FORWARD,
                points=(PointPrompt(2, 2),),
                max_hole_area=10,
            )
            result = VideoSegmentResult(
                outputs=(output,),
                first_frame=1,
                last_frame=1,
                key_frame=1,
                direction=VideoDirection.FORWARD,
                metadata={},
            )

            processed = VideoSegmentService._postprocess_outputs(
                result,
                request,
                CancellationToken(),
            )

            with Image.open(output) as image:
                self.assertEqual(image.getpixel((2, 2)), 255)
            self.assertEqual(processed.metadata["postprocess"]["filled_holes"], 1)

    def test_key_index_and_output_mapping_preserve_nuke_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (2, 2), "white").save(root / "01001.jpg")
            request = VideoSegmentRequest(
                frames_dir=root,
                output_pattern=root / "matte.%04d.png",
                first_frame=1001,
                last_frame=1010,
                key_frame=1005,
                direction=VideoDirection.BOTH,
                points=(PointPrompt(1, 1),),
            )

            request.validate()
            self.assertEqual(request.key_index, 4)
            self.assertEqual(request.output_for_index(4).name, "matte.1005.png")

    def test_key_frame_must_be_inside_range(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            request = VideoSegmentRequest(
                frames_dir=Path(directory),
                output_pattern=Path(directory) / "matte.%04d.png",
                first_frame=1,
                last_frame=10,
                key_frame=11,
                direction=VideoDirection.FORWARD,
                points=(PointPrompt(1, 1),),
            )

            with self.assertRaises(KyvenError):
                request.validate()

    def test_animated_roi_requires_one_entry_per_frame(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            request = VideoSegmentRequest(
                frames_dir=Path(directory),
                output_pattern=Path(directory) / "matte.%04d.png",
                first_frame=1,
                last_frame=2,
                key_frame=1,
                direction=VideoDirection.FORWARD,
                points=(PointPrompt(1, 1),),
                rois=((1, BoxPrompt(0, 0, 2, 2)),),
            )

            with self.assertRaisesRegex(KyvenError, "exactly one entry"):
                request.validate()


if __name__ == "__main__":
    unittest.main()
