"""One unambiguous artist mask input: alpha, including legacy Groups."""
from __future__ import annotations

from typing import Any


def force_alpha_mask(node: Any, selector: Any) -> None:
    """Disconnect legacy channel expressions before hiding the serialized control."""
    selector["which"].clearAnimated()
    selector["which"].setValue(0)
    if "mask_channel" in node.knobs():
        legacy = node.knobs()["mask_channel"]
        legacy.clearAnimated()
        legacy.setValue(0)
        legacy.setVisible(False)
