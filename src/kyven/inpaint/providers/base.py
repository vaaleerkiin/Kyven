from __future__ import annotations

from abc import ABC, abstractmethod

from kyven.cancellation import CancellationToken
from kyven.inpaint.models import InpaintCapabilities, InpaintPrediction, InpaintRequest


class InpaintProvider(ABC):
    @property
    @abstractmethod
    def capabilities(self) -> InpaintCapabilities: ...

    @abstractmethod
    def predict(self, request: InpaintRequest, cancellation: CancellationToken) -> InpaintPrediction: ...

    @abstractmethod
    def unload(self) -> None: ...
