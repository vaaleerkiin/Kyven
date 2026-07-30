from __future__ import annotations

import unittest

from kyven.cancellation import CancellationToken
from kyven.segment.models import ProviderCapabilities, SegmentPrediction, SegmentRequest
from kyven.segment.providers.base import SegmentationProvider
from kyven.segment.providers.registry import ProviderRegistry


class FakeProvider(SegmentationProvider):
    unloaded = False

    @property
    def capabilities(self) -> ProviderCapabilities:
        raise NotImplementedError

    def predict(self, request: SegmentRequest, cancellation: CancellationToken) -> SegmentPrediction:
        raise NotImplementedError

    def unload(self) -> None:
        self.unloaded = True


class ProviderRegistryTests(unittest.TestCase):
    def test_factory_is_lazy_and_instance_is_reused(self) -> None:
        created: list[FakeProvider] = []

        def factory() -> FakeProvider:
            provider = FakeProvider()
            created.append(provider)
            return provider

        registry = ProviderRegistry()
        registry.register("fake", factory)
        self.assertEqual(created, [])
        first = registry.get("fake")
        second = registry.get("fake")
        self.assertIs(first, second)
        self.assertEqual(len(created), 1)

    def test_unload_releases_and_forgets_provider(self) -> None:
        provider = FakeProvider()
        registry = ProviderRegistry()
        registry.register("fake", lambda: provider)
        registry.get("fake")
        registry.unload("fake")
        self.assertTrue(provider.unloaded)

    def test_default_sam2_provider_declares_license_without_loading_model(self) -> None:
        from kyven.segment.providers.registry import default_registry

        registry = default_registry(
            checkpoint="missing-checkpoint.pt",
            model_config="configs/sam2.1/sam2.1_hiera_s.yaml",
        )
        capabilities = registry.get("sam2").capabilities
        self.assertEqual(capabilities.license_name, "Apache-2.0")
        self.assertTrue(capabilities.supports_cpu)

    def test_activate_unloads_a_different_resident_provider(self) -> None:
        first = FakeProvider()
        second = FakeProvider()
        registry = ProviderRegistry()
        registry.register("first", lambda: first)
        registry.register("second", lambda: second)
        registry.activate("first")
        registry.activate("second")
        self.assertTrue(first.unloaded)
        self.assertEqual(registry.loaded_ids, ("second",))


if __name__ == "__main__":
    unittest.main()
