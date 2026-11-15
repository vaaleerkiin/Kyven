"""Register self-contained Nuke plugins before loading the Kyven menu."""

from pathlib import Path

import nuke


_waft = Path(__file__).resolve().parents[2] / "plugins" / "Cattery" / "WAFT"
if (_waft / "WAFT.gizmo").is_file():
    nuke.pluginAddPath(_waft.as_posix())
