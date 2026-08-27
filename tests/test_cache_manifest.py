from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from kyven.atomic_io import replace_with_retry
from kyven.cache_manifest import manifest_path, read_valid_manifest, write_manifest_atomic


class CacheManifestTests(unittest.TestCase):
    def test_atomic_replace_retries_short_windows_read_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            temporary = root / "temporary.png"
            output = root / "output.png"
            temporary.write_bytes(b"new")
            output.write_bytes(b"old")
            real_replace = os.replace
            attempts = 0

            def briefly_locked(source, target):
                nonlocal attempts
                attempts += 1
                if attempts < 3:
                    raise PermissionError(5, "locked")
                real_replace(source, target)

            with mock.patch("kyven.atomic_io.os.replace", side_effect=briefly_locked):
                replace_with_retry(temporary, output)
            self.assertEqual(attempts, 3)
            self.assertEqual(output.read_bytes(), b"new")

    def test_atomic_replace_falls_back_to_in_place_update_for_nuke_read_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            temporary = root / "temporary.png"
            output = root / "output.png"
            temporary.write_bytes(b"new-and-longer")
            output.write_bytes(b"old")

            with mock.patch(
                "kyven.atomic_io.os.replace", side_effect=PermissionError(5, "locked")
            ):
                replace_with_retry(temporary, output, attempts=2)

            self.assertEqual(output.read_bytes(), b"new-and-longer")
            self.assertFalse(temporary.exists())

    def test_manifest_requires_every_valid_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "matte.png"
            logits = root / "logits.npz"
            Image.new("L", (2, 2), 255).save(output)
            np.savez_compressed(logits, logits=np.ones((2, 2), dtype=np.float16))
            write_manifest_atomic(output, "key", {"provider": "test"})

            self.assertIsNotNone(read_valid_manifest(output, "key", (output, logits)))
            self.assertIsNone(read_valid_manifest(output, "different", (output, logits)))
            logits.write_bytes(b"corrupt")
            self.assertIsNone(read_valid_manifest(output, "key", (output, logits)))

    def test_manifest_write_replaces_previous_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.png"
            Image.new("RGB", (1, 1), "black").save(output)
            write_manifest_atomic(output, "first", {"revision": 1})
            write_manifest_atomic(output, "second", {"revision": np.int64(2)})
            payload = read_valid_manifest(output, "second", (output,))
            self.assertEqual(payload["metadata"]["revision"], 2)
            self.assertTrue(manifest_path(output).is_file())


if __name__ == "__main__":
    unittest.main()
