from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from kyven.errors import KyvenError
from kyven.segment.models import PointPrompt
from kyven.segment.video import VideoDirection, VideoSegmentRequest


class VideoSegmentRequestTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
