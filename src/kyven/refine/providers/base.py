"""Provider boundary for replaceable refinement models."""

from __future__ import annotations

from abc import ABC, abstractmethod

from kyven.cancellation import CancellationToken
from kyven.refine.models import (
    RefinementCapabilities,
    RefinePrediction,
    RefineRequest,
)


class RefinementProvider(ABC):
    @property
    @abstractmethod
    def capabilities(self) -> RefinementCapabilities:
        """Describe model, hardware, and licensing."""

    @abstractmethod
    def predict(
        self,
        request: RefineRequest,
        cancellation: CancellationToken,
    ) -> RefinePrediction:
        """Refine one image using its prepared trimap."""

    @abstractmethod
    def unload(self) -> None:
        """Release model resources."""
