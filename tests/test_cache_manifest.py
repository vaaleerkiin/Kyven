from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from kyven.atomic_io import replace_with_retry
from kyven.cache_layout import publish_cached_output
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

    def test_restores_only_missing_or_changed_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "matte.png"
            auxiliary = Path(directory) / "trimap.png"
            Image.new("L", (3, 3), 255).save(output)
            Image.new("L", (3, 3), 128).save(auxiliary)
            original = auxiliary.read_bytes()
            write_manifest_atomic(output, "key", {}, required_outputs=(output, auxiliary))
            unchanged = output.stat().st_mtime_ns
            auxiliary.unlink()
            self.assertIsNotNone(read_valid_manifest(output, "key", (output, auxiliary)))
            self.assertEqual(auxiliary.read_bytes(), original)
            self.assertEqual(output.stat().st_mtime_ns, unchanged)
            Image.new("L", (3, 3), 0).save(auxiliary)
            self.assertIsNotNone(read_valid_manifest(output, "key", (output, auxiliary)))
            self.assertEqual(auxiliary.read_bytes(), original)

    def test_rejects_corrupt_private_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "matte.png"
            Image.new("L", (3, 3), 255).save(output)
            write_manifest_atomic(output, "key", {})
            for artifact in (output.parent / "_kyven_work" / "artifacts").rglob("*.png"):
                artifact.write_bytes(b"broken")
            output.unlink()
            self.assertIsNone(read_valid_manifest(output, "key", (output,)))

    def test_published_manifest_repairs_public_result_and_retains_staging(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "_kyven_work" / "results" / "result.png"
            source.parent.mkdir(parents=True)
            target = root / "artist.1001.png"
            Image.new("L", (3, 3), 255).save(source)
            expected = source.read_bytes()
            write_manifest_atomic(source, "key", {})
            publish_cached_output(source, target)
            self.assertTrue(source.is_file())
            self.assertIsNotNone(read_valid_manifest(target, "key", (target,)))
            previous = target.stat().st_mtime_ns
            publish_cached_output(source, target)
            self.assertEqual(target.stat().st_mtime_ns, previous)
            target.unlink()
            self.assertIsNotNone(read_valid_manifest(target, "key", (target,)))
            self.assertEqual(target.read_bytes(), expected)

    def test_unrecorded_optional_output_cannot_be_adopted_from_other_request(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "matte.png"
            auxiliary = Path(directory) / "trimap.png"
            Image.new("L", (3, 3), 255).save(output)
            write_manifest_atomic(output, "key", {}, required_outputs=(output,))
            Image.new("L", (3, 3), 128).save(auxiliary)
            self.assertIsNone(read_valid_manifest(output, "key", (output, auxiliary)))

    def test_legacy_sidecar_remains_readable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "matte.png"
            Image.new("L", (3, 3), 255).save(output)
            output.with_suffix(".png.kyven.json").write_text(
                '{"version":1,"cache_key":"key","metadata":{}}', encoding="utf-8")
            self.assertIsNotNone(read_valid_manifest(output, "key", (output,)))

    def test_switching_settings_back_restores_previous_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "matte.png"
            Image.new("L", (3, 3), 255).save(output)
            original = output.read_bytes()
            write_manifest_atomic(output, "A", {})
            Image.new("L", (3, 3), 0).save(output)
            write_manifest_atomic(output, "B", {})
            self.assertIsNotNone(read_valid_manifest(output, "A", (output,)))
            self.assertEqual(output.read_bytes(), original)
            self.assertIsNotNone(read_valid_manifest(output, "B", (output,)))
            manifest_path(output).unlink()
            self.assertIsNotNone(read_valid_manifest(output, "A", (output,)))
            self.assertEqual(output.read_bytes(), original)

    def test_publication_rejects_mismatched_pixels_before_replacing_public_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / "staging.png", root / "public.png"
            Image.new("L", (3, 3), 255).save(source)
            write_manifest_atomic(source, "key", {})
            target.write_bytes(b"previous published pixels")
            Image.new("L", (3, 3), 0).save(source)
            with self.assertRaises(ValueError):
                publish_cached_output(source, target)
            self.assertEqual(target.read_bytes(), b"previous published pixels")

    def test_public_roto_manifest_is_kept_inside_work_folder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "_kyven_work").mkdir()
            output = root / "phone_v001.1053.png"

            path = manifest_path(output)

            self.assertEqual(
                path,
                root / "_kyven_work" / "manifests" / "phone_v001.1053.png.kyven.json",
            )


if __name__ == "__main__":
    unittest.main()
