"""Native Nuke model switching and saved-group migration, without inference."""
import json
import os
import sys
import traceback
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "hosts/nuke"), str(ROOT / "src")]
import kyven_nuke  # noqa: F401 - callback namespace in Nuke
import nuke
from kyven_nuke import generative_fill_node as fill

report = {"checks": []}
destination = Path(os.environ["KYVEN_MOSS_NUKE_REPORT"])


def check(name, value):
    assert value, name
    report["checks"].append(name)


try:
    node = fill.create_generative_fill_node()
    check("Klein remains default", fill._model_id(node) == fill.PROVIDER_ID)
    check("both model choices available", len(node["model"].values()) == 2)
    node["model"].setValue(1)
    check("MOSS selection", fill._model_id(node) == fill.MODEL_IDS[1])
    check("MOSS step preset", node["steps"].value() == 20)
    node["steps"].setValue(24)
    node["moss_guidance"].setValue(3.5)
    check("fractional MOSS guidance", node["moss_guidance"].value() == 3.5)
    fill._ensure_fill_graph(node)
    check("refresh preserves artist settings", node["steps"].value() == 24 and fill._model_id(node) == fill.MODEL_IDS[1])
    node["prompt"].setValue("Remove the person")
    with patch.object(fill.inpaint_node, "_payload", return_value={}):
        payload = fill._payload(node)
    check("request routes selected backend", payload["model_id"] == fill.MODEL_IDS[1])
    check("advanced guidance exported", payload["fill"]["moss_guidance"] == 3.5)
    node["model"].setValue(0)
    check("switch back to Klein", fill._model_id(node) == fill.PROVIDER_ID and node["steps"].value() == 4)
    callback = node["knobChanged"].value()
    node["knobChanged"].setValue("")
    node["model"].setValues([fill.MODEL_LABELS[0]])
    node["steps"].setValue(7)
    node.removeKnob(node["moss_guidance"])
    node["knobChanged"].setValue(callback)
    fill._ensure_fill_graph(node)
    check("old saved selector upgrades", len(node["model"].values()) == 2)
    check("migration preserves Klein settings", node["steps"].value() == 7)
    check("migration adds MOSS guidance", "moss_guidance" in node.knobs())
    check("Torch stays outside Nuke", "torch" not in sys.modules)
    report["ok"] = True
except Exception:
    report["error"] = traceback.format_exc()
    raise
finally:
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))
    nuke.scriptExit()
