"""Safe user-knob edits without exposing intermediate interfaces to callbacks."""
from __future__ import annotations

from functools import wraps
from typing import Any


def suspend_callbacks(function: Any) -> Any:
    @wraps(function)
    def guarded(node: Any, *args: Any, **kwargs: Any) -> Any:
        knob = node.knobs().get("knobChanged")
        if knob is None:
            return function(node, *args, **kwargs)
        callback = knob.value()
        knob.setValue("")
        try:
            return function(node, *args, **kwargs)
        finally:
            knob.setValue(callback)
    return guarded
