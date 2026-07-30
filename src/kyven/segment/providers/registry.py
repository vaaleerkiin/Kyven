"""Lazy provider registry that avoids loading disabled models."""

from __future__ import annotations

from collections.abc import Callable

from kyven.errors import ErrorCode, KyvenError
from kyven.segment.providers.base import SegmentationProvider

ProviderFactory = Callable[[], SegmentationProvider]


class ProviderRegistry:
    """Register factories without constructing model providers."""

    def __init__(self) -> None:
        self._factories: dict[str, ProviderFactory] = {}
        self._instances: dict[str, SegmentationProvider] = {}

    def register(self, provider_id: str, factory: ProviderFactory) -> None:
        """Register a provider factory under a stable identifier."""

        self._factories[provider_id] = factory

    def get(self, provider_id: str) -> SegmentationProvider:
        """Construct a provider only when it is first selected."""

        if provider_id not in self._factories:
            raise KyvenError(
                code=ErrorCode.PROVIDER_UNAVAILABLE,
                message=f"Segmentation provider is not available: {provider_id}",
                recoverable=True,
                suggested_action="List installed providers and select an available one.",
            )
        if provider_id not in self._instances:
            self._instances[provider_id] = self._factories[provider_id]()
        return self._instances[provider_id]

    def activate(self, provider_id: str) -> SegmentationProvider:
        """Keep only the selected provider resident before returning it."""

        for loaded_id in tuple(self._instances):
            if loaded_id != provider_id:
                self.unload(loaded_id)
        return self.get(provider_id)

    def unload(self, provider_id: str) -> None:
        """Unload and forget one constructed provider."""

        provider = self._instances.pop(provider_id, None)
        if provider is not None:
            provider.unload()

    def unload_all(self) -> None:
        """Unload every constructed provider."""

        for provider_id in tuple(self._instances):
            self.unload(provider_id)

    @property
    def registered_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))

    @property
    def loaded_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._instances))


def default_registry(
    *,
    checkpoint: str,
    model_config: str,
    device: str = "auto",
    expected_checksum: str | None = None,
) -> ProviderRegistry:
    """Create the built-in registry without importing or loading SAM 2."""

    from kyven.segment.providers.sam2 import Sam2Provider

    registry = ProviderRegistry()
    registry.register(
        "sam2",
        lambda: Sam2Provider(
            checkpoint=checkpoint,
            model_config=model_config,
            device=device,
            expected_checksum=expected_checksum,
        ),
    )
    return registry
