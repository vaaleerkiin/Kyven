"""Provider boundary for replaceable segmentation models."""

from __future__ import annotations

from abc import ABC, abstractmethod

from kyven.cancellation import CancellationToken
from kyven.segment.models import ProviderCapabilities, SegmentPrediction, SegmentRequest


class SegmentationProvider(ABC):
    """Model-neutral segmentation interface implemented outside host adapters."""

    @property
    @abstractmethod
    def capabilities(self) -> ProviderCapabilities:
        """Describe hardware, prompts, model identity, and licensing."""

    @abstractmethod
    def predict(
        self,
        request: SegmentRequest,
        cancellation: CancellationToken,
    ) -> SegmentPrediction:
        """Run segmentation for one source image."""

    @abstractmethod
    def unload(self) -> None:
        """Release model resources, including VRAM where applicable."""

