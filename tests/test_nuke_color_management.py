from __future__ import annotations

import sys
import unittest
from pathlib import Path

NUKE_ROOT = Path(__file__).parents[1] / "hosts" / "nuke"
sys.path.insert(0, str(NUKE_ROOT))

from kyven_nuke.color_management import set_data_io


class _Knob:
    def __init__(self, value: object = None, values: list[str] | None = None) -> None:
        self.value = value
        self._values = values or []

    def setValue(self, value: object) -> None:
        self.value = value

    def values(self) -> list[str]:
        return self._values


class _Node:
    def __init__(self, **knobs: _Knob) -> None:
        self._knobs = knobs

    def knobs(self) -> dict[str, _Knob]:
        return self._knobs

    def __getitem__(self, name: str) -> _Knob:
        return self._knobs[name]


class NukeColorManagementTests(unittest.TestCase):
    def test_raw_knob_bypasses_ocio(self) -> None:
        raw = _Knob(False)
        set_data_io(_Node(raw=raw))
        self.assertIs(raw.value, True)

    def test_colorspace_fallback_selects_utility_raw(self) -> None:
        colorspace = _Knob("ACES - ACEScg", ["ACES - ACEScg", "Utility - Raw"])
        set_data_io(_Node(colorspace=colorspace))
        self.assertEqual(colorspace.value, "Utility - Raw")

    def test_missing_color_controls_is_supported(self) -> None:
        set_data_io(_Node())


if __name__ == "__main__":
    unittest.main()
