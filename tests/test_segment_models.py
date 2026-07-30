from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kyven.errors import ErrorCode, KyvenError
from kyven.segment.models import BoxPrompt, PointPrompt, SegmentRequest


class SegmentModelTests(unittest.TestCase):
    def test_cache_key_is_deterministic_and_ignores_output_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.bin"
            source.write_bytes(b"synthetic fixture")
            first = SegmentRequest(source, root / "one.png", points=(PointPrompt(1, 2),))
            second = SegmentRequest(source, root / "two.png", points=(PointPrompt(1, 2),))
            self.assertEqual(first.cache_key("1", "abc"), second.cache_key("1", "abc"))

    def test_processing_roi_changes_cache_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.bin"
            source.write_bytes(b"synthetic fixture")
            first = SegmentRequest(
                source,
                root / "one.png",
                points=(PointPrompt(2, 2),),
                roi=BoxPrompt(0, 0, 10, 10),
            )
            second = SegmentRequest(
                source,
                root / "two.png",
                points=(PointPrompt(2, 2),),
                roi=BoxPrompt(0, 0, 20, 20),
            )

            self.assertNotEqual(first.cache_key("1", "abc"), second.cache_key("1", "abc"))

    def test_invalid_box_is_structured_error(self) -> None:
        with self.assertRaises(KyvenError) as caught:
            BoxPrompt(10, 10, 5, 20)
        self.assertEqual(caught.exception.code, ErrorCode.INVALID_REQUEST)

    def test_request_requires_a_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.bin"
            source.write_bytes(b"fixture")
            request = SegmentRequest(source, Path(directory) / "mask.png")
            with self.assertRaises(KyvenError) as caught:
                request.validate()
            self.assertEqual(caught.exception.code, ErrorCode.INVALID_REQUEST)


if __name__ == "__main__":
    unittest.main()
