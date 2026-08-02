from __future__ import annotations

import sys
import unittest
from pathlib import Path

NUKE_ROOT = Path(__file__).parents[1] / "hosts" / "nuke"
sys.path.insert(0, str(NUKE_ROOT))

from kyven_nuke.color_management import match_color_io, set_data_io, set_interchange_color_io


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
        colorspace = _Knob("ACES - ACEScg", ["ACES - ACEScg", "Utility - Raw"])
        set_data_io(_Node(raw=raw, colorspace=colorspace))
        self.assertIs(raw.value, True)
        self.assertEqual(colorspace.value, "Utility - Raw")

    def test_colorspace_fallback_selects_utility_raw(self) -> None:
        colorspace = _Knob("ACES - ACEScg", ["ACES - ACEScg", "Utility - Raw"])
        set_data_io(_Node(colorspace=colorspace))
        self.assertEqual(colorspace.value, "Utility - Raw")

    def test_missing_color_controls_is_supported(self) -> None:
        set_data_io(_Node())

    def test_interchange_prefers_shared_srgb_texture_space(self) -> None:
        raw = _Knob(True)
        colorspace = _Knob(
            "default",
            ["Output - sRGB", "ACES - ACEScg", "Utility - sRGB - Texture"],
        )
        set_interchange_color_io(_Node(raw=raw, colorspace=colorspace))
        self.assertIs(raw.value, False)
        self.assertEqual(colorspace.value, "Utility - sRGB - Texture")

    def test_color_read_matches_source_writer(self) -> None:
        source = _Node(colorspace=_Knob("Utility - sRGB - Texture"))
        target_raw = _Knob(True)
        target_space = _Knob("default")
        match_color_io(source, _Node(raw=target_raw, colorspace=target_space))
        self.assertIs(target_raw.value, False)
        self.assertEqual(target_space.value, "Utility - sRGB - Texture")


if __name__ == "__main__":
    unittest.main()
