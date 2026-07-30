"""Cooperative cancellation primitives for long-running jobs."""

from __future__ import annotations

from threading import Event, RLock

from kyven.errors import ErrorCode, KyvenError


class CancellationToken:
    """Thread-safe cancellation flag passed across engine boundaries."""

    def __init__(self) -> None:
        self._event = Event()
        self._progress_lock = RLock()
        self._progress = 0.0
        self._progress_message = "Queued"

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

    def report_progress(self, progress: float, message: str) -> None:
        """Publish bounded progress for host polling without coupling to a UI."""

        with self._progress_lock:
            self._progress = max(self._progress, min(1.0, max(0.0, float(progress))))
            self._progress_message = str(message)

    def progress_snapshot(self) -> tuple[float, str]:
        """Return the latest progress fraction and user-facing stage."""

        with self._progress_lock:
            return self._progress, self._progress_message
