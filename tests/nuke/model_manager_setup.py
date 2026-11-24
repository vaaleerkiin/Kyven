"""Native manager construction before .venv exists; optional external runtime probes."""
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "hosts/nuke"), str(ROOT / "src")]
import nuke
from kyven_nuke import model_manager, runtime

destination = Path(os.environ["KYVEN_MANAGER_REPORT"])
report = {"checks": []}
original_root = os.environ.get("KYVEN_ROOT")
native_panel = nuke.Panel


class Panel:
    def __init__(self, title):
        self.native = native_panel(title)

    def addEnumerationPulldown(self, name, values):
        self.native.addEnumerationPulldown(name, values)
        if name == "Action":
            assert "Install Required Components" in values
        if name == "Model":
            assert "Kyven Base Runtime" in values and "Klein" in values
        report["checks"].append(name)

    def addNotepad(self, *args):
        self.native.addNotepad(*args)

    def show(self):
        return False


try:
    assert "torch" not in sys.modules
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        catalog = root / "src/kyven/resources/models/catalog.json"
        catalog.parent.mkdir(parents=True)
        catalog.write_bytes((ROOT / "src/kyven/resources/models/catalog.json").read_bytes())
        os.environ["KYVEN_ROOT"] = directory
        nuke.Panel = Panel
        model_manager.show_model_manager()
        report["checks"].append("manager opens without .venv or server")
    ready_root = os.environ.get("KYVEN_MANAGER_READY_ROOT")
    if ready_root:
        os.environ["KYVEN_ROOT"] = ready_root
        assert runtime.ensure_model_runtime("core") is False
        assert runtime.ensure_model_runtime("generative-fill") is False
        report["checks"].append("core and fill dependencies verified externally")
    assert "torch" not in sys.modules
    report["checks"].append("Nuke never imports Torch")
    report["ok"] = True
finally:
    nuke.Panel = native_panel
    if original_root is None:
        os.environ.pop("KYVEN_ROOT", None)
    else:
        os.environ["KYVEN_ROOT"] = original_root
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))
    nuke.scriptExit()
