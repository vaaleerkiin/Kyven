from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from kyven.cancellation import CancellationToken
from kyven.errors import ErrorCode, KyvenError
from kyven.refine.models import (
    RefinementCapabilities,
    RefinePrediction,
    RefineRequest,
)
from kyven.refine.providers.base import RefinementProvider
from kyven.refine.service import RefineService
from kyven.refine.trimap import generate_trimap, normalize_trimap
from kyven.segment.models import BoxPrompt
from kyven.segment.providers.registry import ProviderRegistry
from kyven.server.jobs import JobManager


class SyntheticRefinementProvider(RefinementProvider):
    calls = 0
    @property
    def capabilities(self) -> RefinementCapabilities:
        return RefinementCapabilities(
            provider_id="synthetic-refine",
            display_name="Synthetic Refine",
            provider_version="1",
            model_checksum="fixture",
            license_name="CC0-1.0",
            license_url="https://creativecommons.org/publicdomain/zero/1.0/",
            supports_cpu=True,
            supports_tiling=True,
            minimum_vram_mb=0,
        )

    def predict(
        self,
        request: RefineRequest,
        cancellation: CancellationToken,
    ) -> RefinePrediction:
        type(self).calls += 1
        cancellation.raise_if_cancelled()
        with Image.open(request.mask) as trimap:
            pixels = np.asarray(trimap.convert("L"))
        alpha = np.where(pixels >= 254, 1.0, np.where(pixels <= 1, 0.0, 0.5))
        return RefinePrediction(np.asarray(alpha, dtype=np.float32), {"tiles": 1})

    def unload(self) -> None:
        return


class TrimapTests(unittest.TestCase):
    def test_integral_morphology_matches_pillow_rank_filters(self) -> None:
        pixels = (np.random.default_rng(7).random((23, 31)) > 0.58).astype(np.uint8) * 255
        image = Image.fromarray(pixels, mode="L")
        for foreground_radius, background_radius in ((0, 0), (1, 2), (3, 1)):
            foreground = (
                np.asarray(
                    image.filter(ImageFilter.MinFilter(foreground_radius * 2 + 1))
                    if foreground_radius else image
                )
                >= 128
            )
            possible = (
                np.asarray(
                    image.filter(ImageFilter.MaxFilter(background_radius * 2 + 1))
                    if background_radius else image
                )
                >= 128
            )
            expected = np.where(foreground, 255, np.where(possible, 128, 0)).astype(
                np.uint8
            )

            actual = generate_trimap(pixels, foreground_radius, background_radius)

            np.testing.assert_array_equal(actual, expected)

    def test_generated_trimap_has_three_regions(self) -> None:
        mask = np.zeros((9, 9), dtype=np.uint8)
        mask[2:7, 2:7] = 255
        trimap = generate_trimap(mask, foreground_radius=1, background_radius=1)
        self.assertEqual(trimap[4, 4], 255)
        self.assertEqual(trimap[0, 0], 0)
        self.assertEqual(trimap[2, 2], 128)

    def test_artist_trimap_is_quantized(self) -> None:
        trimap = normalize_trimap(np.asarray([[0, 64, 127, 200, 255]], dtype=np.uint8))
        self.assertEqual(trimap.tolist(), [[0, 0, 128, 255, 255]])


class RefineServiceTests(unittest.TestCase):
    def test_valid_refine_manifest_skips_second_inference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            mask = root / "mask.png"
            Image.new("RGB", (4, 4), "white").save(source)
            Image.new("L", (4, 4), 255).save(mask)
            registry = ProviderRegistry()
            registry.register("synthetic-refine", SyntheticRefinementProvider)
            request = RefineRequest(
                source,
                mask,
                root / "alpha.png",
                provider_id="synthetic-refine",
            )
            SyntheticRefinementProvider.calls = 0
            service = RefineService(registry)
            first = service.run(request)
            second = service.run(request)
            self.assertEqual(SyntheticRefinementProvider.calls, 1)
            self.assertFalse(first.metadata["cache_hit"])
            self.assertTrue(second.metadata["cache_hit"])
            original = request.output.read_bytes()
            request.output.unlink()
            third = service.run(request)
            self.assertTrue(third.metadata["cache_hit"])
            self.assertEqual(SyntheticRefinementProvider.calls, 1)
            self.assertEqual(request.output.read_bytes(), original)

    def test_low_memory_payload_defaults_to_512_pixel_tiles(self) -> None:
        request = JobManager.refine_request_from_payload(
            {
                "source": "D:/source.png",
                "mask": "D:/mask.png",
                "output": "D:/alpha.png",
                "profile": "low_memory",
            }
        )
        self.assertEqual(request.tile_size, 512)
        self.assertEqual(request.tile_overlap, 64)

    def test_payload_accepts_auto_crop_with_context(self) -> None:
        request = JobManager.refine_request_from_payload(
            {
                "source": "D:/source.png",
                "mask": "D:/mask.png",
                "output": "D:/alpha.png",
                "crop_mode": "auto",
                "context_padding": 96,
            }
        )

        self.assertEqual(request.crop_mode, "auto")
        self.assertEqual(request.context_padding, 96)
        self.assertIsNone(request.roi)

    def test_auto_crop_bounds_full_trimap_and_adds_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            mask = root / "mask.png"
            output = root / "refined.png"
            Image.new("RGB", (12, 10), "white").save(source)
            mask_pixels = np.zeros((10, 12), dtype=np.uint8)
            mask_pixels[4:6, 5:7] = 255
            Image.fromarray(mask_pixels, mode="L").save(mask)
            registry = ProviderRegistry()
            registry.register("synthetic-refine", SyntheticRefinementProvider)

            result = RefineService(registry).run(
                RefineRequest(
                    source=source,
                    mask=mask,
                    output=output,
                    provider_id="synthetic-refine",
                    crop_mode="auto",
                    context_padding=2,
                    foreground_radius=0,
                    background_radius=0,
                )
            )

            self.assertEqual(
                result.metadata["processing_roi"],
                {
                    "x0": 3,
                    "y0": 2,
                    "x1": 9,
                    "y1": 8,
                    "width": 6,
                    "height": 6,
                    "source_width": 12,
                    "source_height": 10,
                },
            )
            self.assertEqual(result.metadata["crop_mode"], "auto")

    def test_roi_refinement_restores_full_frame_and_preserves_outside_mask(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            mask = root / "mask.png"
            output = root / "refined.png"
            trimap_output = root / "trimap.png"
            Image.new("RGB", (8, 8), "white").save(source)
            mask_pixels = np.zeros((8, 8), dtype=np.uint8)
            mask_pixels[1:7, 1:7] = 255
            Image.fromarray(mask_pixels, mode="L").save(mask)
            registry = ProviderRegistry()
            registry.register("synthetic-refine", SyntheticRefinementProvider)
            result = RefineService(registry).run(
                RefineRequest(
                    source=source,
                    mask=mask,
                    output=output,
                    trimap_output=trimap_output,
                    provider_id="synthetic-refine",
                    roi=BoxPrompt(0, 0, 4, 4),
                    foreground_radius=1,
                    background_radius=1,
                )
            )
            with Image.open(output) as matte:
                self.assertEqual(matte.size, (8, 8))
                self.assertEqual(matte.getpixel((6, 6)), 255)
                self.assertIn(matte.getpixel((1, 1)), {127, 128})
            with Image.open(trimap_output) as trimap:
                self.assertEqual(trimap.size, (8, 8))
                self.assertEqual(trimap.getpixel((6, 6)), 0)
                self.assertEqual(trimap.getpixel((1, 1)), 128)
            self.assertEqual(result.trimap_output, trimap_output)
            self.assertEqual(result.metadata["processing_roi"]["width"], 4)
            self.assertEqual(len(result.cache_key), 64)

    def test_trimap_settings_change_cache_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            mask = root / "mask.png"
            Image.new("RGB", (2, 2), "white").save(source)
            Image.new("L", (2, 2), 255).save(mask)
            first = RefineRequest(source, mask, root / "a.png", foreground_radius=2)
            second = RefineRequest(source, mask, root / "b.png", foreground_radius=3)
            self.assertNotEqual(first.cache_key("1", "x"), second.cache_key("1", "x"))

    def test_cancelled_refine_does_not_write_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            mask = root / "mask.png"
            output = root / "alpha.png"
            Image.new("RGB", (4, 4), "white").save(source)
            Image.new("L", (4, 4), 255).save(mask)
            registry = ProviderRegistry()
            registry.register("synthetic-refine", SyntheticRefinementProvider)
            token = CancellationToken()
            token.cancel()
            with self.assertRaises(KyvenError) as caught:
                RefineService(registry).run(
                    RefineRequest(
                        source=source,
                        mask=mask,
                        output=output,
                        provider_id="synthetic-refine",
                    ),
                    token,
                )
            self.assertEqual(caught.exception.code, ErrorCode.CANCELLED)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
