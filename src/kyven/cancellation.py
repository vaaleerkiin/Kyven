"""Cooperative cancellation primitives for long-running jobs."""

from __future__ import annotations

from threading import Event

from kyven.errors import ErrorCode, KyvenError


class CancellationToken:
    """Thread-safe cancellation flag passed across engine boundaries."""

    def __init__(self) -> None:
        self._event = Event()

    def cancel(self) -> None:
        """Request cancellation."""

        self._event.set()

    @property
    def is_cancelled(self) -> bool:
        """Return whether cancellation has been requested."""

        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        """Raise a structured cancellation error when requested."""

        if self.is_cancelled:
            raise KyvenError(
                code=ErrorCode.CANCELLED,
                message="The segmentation job was cancelled.",
                recoverable=True,
                suggested_action="Run the job again when ready.",
            )

