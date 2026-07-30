"""Structured errors shared by the engine, CLI, and host adapters."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ErrorCode(str, Enum):
    """Stable machine-readable Kyven error codes."""

    INVALID_REQUEST = "invalid_request"
    DEPENDENCY_MISSING = "dependency_missing"
    MODEL_NOT_FOUND = "model_not_found"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    CANCELLED = "cancelled"
    INFERENCE_FAILED = "inference_failed"
    OUTPUT_FAILED = "output_failed"


@dataclass(slots=True)
class KyvenError(Exception):
    """An error that can be rendered safely in a host UI."""

    code: ErrorCode
    message: str
    technical_detail: str = ""
    recoverable: bool = False
    suggested_action: str = ""

    def __str__(self) -> str:
        return self.message

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable representation for IPC and host adapters."""

        return {
            "code": self.code.value,
            "message": self.message,
            "technical_detail": self.technical_detail,
            "recoverable": self.recoverable,
            "suggested_action": self.suggested_action,
        }

