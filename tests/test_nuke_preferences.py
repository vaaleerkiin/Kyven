from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

NUKE_ROOT = Path(__file__).parents[1] / "hosts" / "nuke"
sys.path.insert(0, str(NUKE_ROOT))

from kyven_nuke.preferences import default_model_index, load_defaults, set_default_model


class NukePreferencesTests(unittest.TestCase):
    def test_defaults_fall_back_when_file_is_missing_or_corrupt(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch("kyven_nuke.preferences.config.runtime_dir", return_value=Path(directory)),
        ):
            self.assertEqual(load_defaults()["refine"], "vitmatte-small-composition-1k")
            self.assertEqual(load_defaults()["generative_fill"], "mage-flow-edit-turbo-q6")
            (Path(directory) / "preferences.json").write_text("broken", encoding="utf-8")
            self.assertEqual(load_defaults()["segment"], "sam2.1-small")

    def test_selected_default_is_persistent_and_resolves_to_index(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch("kyven_nuke.preferences.config.runtime_dir", return_value=Path(directory)),
        ):
            set_default_model("refine", "mematte-vits-aim500")
            self.assertEqual(load_defaults()["refine"], "mematte-vits-aim500")
            self.assertEqual(
                default_model_index(
                    "refine",
                    (
                        "vitmatte-small-composition-1k",
                        "vitmatte-base-distinctions-646",
                        "mematte-vits-aim500",
                        "mematte-vitb-composition-1k",
                    ),
                ),
                2,
            )


if __name__ == "__main__":
    unittest.main()
