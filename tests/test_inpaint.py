from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from kyven.inpaint.models import InpaintCapabilities, InpaintPrediction, InpaintRequest
from kyven.inpaint.providers.sdxl import _model_size
from kyven.inpaint.service import InpaintService
from kyven.segment.models import BoxPrompt
from kyven.segment.providers.registry import ProviderRegistry
from kyven.server.jobs import JobManager


class FakeInpaintProvider:
    def __init__(self) -> None:
        self.requests: list[InpaintRequest] = []
        self.mask_pixels: list[np.ndarray] = []

    @property
    def capabilities(self) -> InpaintCapabilities:
        return InpaintCapabilities("fake", "Fake", "1", "checksum", "MIT", "", True, None)

    def predict(self, request: InpaintRequest, cancellation) -> InpaintPrediction:
        self.requests.append(request)
        with Image.open(request.mask) as mask:
            self.mask_pixels.append(np.asarray(mask.convert("L")).copy())
        with Image.open(request.source) as image:
            shape = np.asarray(image.convert("RGB")).shape
        return InpaintPrediction(np.full(shape, (255, 0, 0), dtype=np.uint8))

    def unload(self) -> None: pass


class InpaintServiceTests(unittest.TestCase):
    def test_sdxl_model_size_preserves_aspect_and_uses_quality_resolution(self) -> None:
        self.assertEqual(_model_size((400, 200), 1024), (1024, 512))
        self.assertEqual(_model_size((1920, 1080), 768), (768, 432))

    def test_generative_parameters_are_validated_and_cached(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            mask = root / "mask.png"
            output = root / "output.png"
            Image.new("RGB", (8, 8), "gray").save(source)
            Image.new("L", (8, 8), 255).save(mask)
            request = InpaintRequest(
                source, mask, output, provider_id="sdxl-inpainting-1.0",
                prompt="remove the sign", negative_prompt="text", seed=42,
                steps=18, guidance_scale=5.5, strength=0.95,
                render_quality="preview",
            )
            request.validate()
            canonical = request.canonical()
            self.assertEqual(canonical["prompt"], "remove the sign")
            self.assertEqual(canonical["seed"], 42)
            self.assertEqual(canonical["render_quality"], "preview")

            parsed = JobManager.inpaint_request_from_payload(
                {
                    "source": str(source.resolve()),
                    "mask": str(mask.resolve()),
                    "output": str(output.resolve()),
                    "prompt": "new wall",
                    "seed": 17,
                },
                default_model_id="sdxl-inpainting-1.0",
            )
            self.assertEqual(parsed.provider_id, "sdxl-inpainting-1.0")
            self.assertEqual(parsed.prompt, "new wall")
            self.assertEqual(parsed.seed, 17)

    def _registry(self, provider: FakeInpaintProvider) -> ProviderRegistry:
        registry = ProviderRegistry(); registry.register("fake", lambda: provider); return registry

    def test_auto_roi_and_masked_merge_preserve_outside(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "source.png"; mask = root / "mask.png"; output = root / "output.png"; patch = root / "patch.png"
            Image.fromarray(np.full((20, 30, 3), 50, dtype=np.uint8)).save(source)
            pixels = np.zeros((20, 30), dtype=np.uint8); pixels[8:12, 13:17] = 255; Image.fromarray(pixels).save(mask)
            provider = FakeInpaintProvider()
            result = InpaintService(self._registry(provider)).run(InpaintRequest(source, mask, output, patch_output=patch, provider_id="fake", context_padding=2, mask_grow=0, edge_color_match=0))
            rendered = np.asarray(Image.open(output).convert("RGB"))
            rendered_patch = np.asarray(Image.open(patch).convert("RGB"))
            self.assertTrue(np.all(rendered[0, 0] == 50)); self.assertTrue(np.all(rendered[9, 14] == (255, 0, 0)))
            self.assertTrue(np.all(rendered_patch[0, 0] == 50)); self.assertTrue(np.all(rendered_patch[7, 12] == (255, 0, 0)))
            self.assertEqual(result.metadata["processing_roi"]["width"], 8)
            self.assertEqual(Image.open(provider.requests[0].source).size if provider.requests[0].source.exists() else (8, 8), (8, 8))

    def test_empty_mask_does_not_activate_provider(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "source.png"; mask = root / "mask.png"; output = root / "output.png"
            original = np.arange(300, dtype=np.uint8).reshape(10, 10, 3); Image.fromarray(original).save(source); Image.fromarray(np.zeros((10, 10), dtype=np.uint8)).save(mask)
            registry = ProviderRegistry()
            result = InpaintService(registry).run(InpaintRequest(source, mask, output))
            self.assertTrue(result.metadata["empty_mask"]); np.testing.assert_array_equal(np.asarray(Image.open(output)), original)

    def test_manual_roi_ignores_mask_outside_roi(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "source.png"; mask = root / "mask.png"; output = root / "output.png"
            Image.fromarray(np.full((10, 10, 3), 25, dtype=np.uint8)).save(source)
            pixels = np.zeros((10, 10), dtype=np.uint8); pixels[8, 8] = 255; Image.fromarray(pixels).save(mask)
            provider = FakeInpaintProvider()
            result = InpaintService(self._registry(provider)).run(InpaintRequest(source, mask, output, provider_id="fake", crop_mode="manual", roi=BoxPrompt(0, 0, 5, 5)))
            self.assertTrue(result.metadata["empty_mask"]); self.assertFalse(provider.requests)

    def test_invert_and_grow_drive_effective_composite_mask(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "source.png"; mask = root / "mask.png"
            output = root / "output.png"; processed = root / "processed.png"
            Image.fromarray(np.full((12, 12, 3), 25, dtype=np.uint8)).save(source)
            pixels = np.full((12, 12), 255, dtype=np.uint8); pixels[2:10, 2:10] = 0
            Image.fromarray(pixels).save(mask); provider = FakeInpaintProvider()
            result = InpaintService(self._registry(provider)).run(InpaintRequest(
                source, mask, output, mask_output=processed, provider_id="fake",
                invert_mask=True, mask_grow=-1, context_padding=0,
            ))
            processed_pixels = np.asarray(Image.open(processed))
            self.assertEqual(result.mask_output, processed)
            self.assertEqual(int(np.count_nonzero(processed_pixels)), 36)

    def test_rgba_source_can_supply_mask_alpha_in_one_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); combined = root / "combined.tif"
            output = root / "output.png"; processed = root / "processed.png"
            pixels = np.full((10, 14, 4), 40, dtype=np.uint8)
            pixels[..., 3] = 0; pixels[3:7, 5:9, 3] = 255
            Image.fromarray(pixels, mode="RGBA").save(combined)
            provider = FakeInpaintProvider()
            InpaintService(self._registry(provider)).run(InpaintRequest(
                combined, combined, output, mask_output=processed, provider_id="fake",
                mask_channel="alpha", context_padding=0, mask_grow=0,
            ))
            self.assertEqual(int(np.count_nonzero(np.asarray(Image.open(processed)))), 16)
            self.assertEqual(len(provider.requests), 1)

    def test_edge_color_match_corrects_patch_offset_without_touching_outside(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "source.png"; mask = root / "mask.png"
            output = root / "output.png"
            Image.fromarray(np.full((40, 40, 3), 50, dtype=np.uint8)).save(source)
            pixels = np.zeros((40, 40), dtype=np.uint8); pixels[16:24, 16:24] = 255
            Image.fromarray(pixels).save(mask)
            provider = FakeInpaintProvider()
            result = InpaintService(self._registry(provider)).run(InpaintRequest(
                source, mask, output, provider_id="fake", context_padding=8,
                mask_grow=0, edge_color_match=1,
            ))
            rendered = np.asarray(Image.open(output).convert("RGB"))
            np.testing.assert_array_equal(rendered[0, 0], (50, 50, 50))
            np.testing.assert_array_equal(rendered[20, 20], (223, 32, 32))
            self.assertEqual(result.metadata["edge_color_offset"], [-32.0, 32.0, 32.0])

    def test_disabled_preprocess_preserves_clean_soft_blend_mask(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            mask = root / "mask.png"
            output = root / "output.png"
            processed = root / "processed.png"
            model_mask = root / "model-mask.png"
            Image.fromarray(np.full((8, 8, 3), 25, dtype=np.uint8)).save(source)
            pixels = np.zeros((8, 8), dtype=np.uint8)
            pixels[2:6, 2:6] = 192
            pixels[3:5, 3:5] = 255
            Image.fromarray(pixels).save(mask)
            Image.fromarray(np.where(pixels >= 128, 255, 0).astype(np.uint8)).save(model_mask)
            provider = FakeInpaintProvider()

            InpaintService(self._registry(provider)).run(
                InpaintRequest(
                    source,
                    mask,
                    output,
                    model_mask=model_mask,
                    mask_output=processed,
                    provider_id="fake",
                    crop_mode="full",
                    preprocess_mask=False,
                    invert_mask=True,
                    mask_grow=40,
                    edge_color_match=0,
                )
            )

            np.testing.assert_array_equal(np.asarray(Image.open(processed)), pixels)
            self.assertEqual(len(provider.requests), 1)

    def test_exported_model_mask_drives_inference_and_composite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            mask = root / "mask.png"
            model_mask = root / "model-mask.png"
            output = root / "output.png"
            patch = root / "patch.png"
            processed = root / "processed.png"
            Image.fromarray(np.full((20, 20, 3), 25, dtype=np.uint8)).save(source)
            clean = np.zeros((20, 20), dtype=np.uint8)
            clean[9:11, 9:11] = 255
            exported = np.zeros((20, 20), dtype=np.uint8)
            exported[6:14, 6:14] = 255
            Image.fromarray(clean).save(mask)
            Image.fromarray(exported).save(model_mask)
            provider = FakeInpaintProvider()

            InpaintService(self._registry(provider)).run(
                InpaintRequest(
                    source,
                    mask,
                    output,
                    model_mask=model_mask,
                    mask_output=processed,
                    patch_output=patch,
                    provider_id="fake",
                    crop_mode="full",
                    mask_grow=-128,
                    mask_threshold=1.0,
                    edge_color_match=0,
                )
            )

            self.assertEqual(int(np.count_nonzero(provider.mask_pixels[0])), 64)
            np.testing.assert_array_equal(np.asarray(Image.open(processed)), exported)
            rendered = np.asarray(Image.open(output).convert("RGB"))
            rendered_patch = np.asarray(Image.open(patch).convert("RGB"))
            self.assertTrue(np.all(rendered[6, 6] == (255, 0, 0)))
            self.assertTrue(np.all(rendered[9, 9] == (255, 0, 0)))
            self.assertTrue(np.all(rendered_patch[6, 6] == (255, 0, 0)))


if __name__ == "__main__":
    unittest.main()
