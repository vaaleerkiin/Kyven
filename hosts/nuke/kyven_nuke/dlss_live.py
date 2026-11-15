"""Host-independent latest-request-wins state machine for the DLSS UI callback."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Event


class DlssCancellation(Event):
    """Distinguish a user stop from replacement by a newer preview request."""

    def __init__(self) -> None:
        super().__init__()
        self.reason = ""

    def cancel(self, reason: str) -> None:
        if not self.is_set():
            self.reason = reason
        self.set()


@dataclass
class DlssLiveState:
    desired: str = ""
    changed_at: float = 0.0
    attempted: str = ""
    active: str = ""
    live_active: bool = True
    cancel: DlssCancellation = field(default_factory=DlssCancellation)
    settings: str = ""

    def observe(self, signature: str, now: float, enabled: bool = True) -> None:
        if not enabled:
            if self.live_active:
                self.cancel.cancel("live_disabled")
            self.desired = self.attempted = ""
            return
        if signature != self.desired:
            self.desired, self.changed_at = signature, now
            if self.active and self.live_active and signature != self.active:
                self.cancel.cancel("superseded")

    def ready(self, now: float, delay: float) -> bool:
        return bool(self.desired and not self.active and self.desired != self.attempted
                    and now - self.changed_at >= delay)

    def begin(self, signature: str, live: bool = True) -> DlssCancellation:
        self.active = self.attempted = signature
        self.live_active = live
        self.cancel = DlssCancellation()
        return self.cancel

    def finish(self, signature: str) -> None:
        if self.active == signature:
            if self.cancel.is_set() and self.cancel.reason in {"superseded", "settings_changed"}:
                self.attempted = ""
            self.active = ""

    def accepts(self, signature: str) -> bool:
        return signature == self.desired and not self.cancel.is_set()
