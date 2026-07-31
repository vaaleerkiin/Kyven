"""Shared Kyven visual identity for the Nuke host adapter."""

from __future__ import annotations

import webbrowser
from pathlib import Path
from typing import Any

PROJECT_URL = "https://github.com/vaaleerkiin/Kyven"
LOGO_PATH = Path(__file__).resolve().parents[1] / "icons" / "kyven_logo.png"


def open_project_page() -> None:
    """Open the Kyven project page in the user's default browser."""

    webbrowser.open(PROJECT_URL)


def logo_markup() -> str:
    """Return compact rich text used at the top of every Kyven node."""

    # Nuke's Qt rich-text renderer accepts a native absolute path here, but displays
    # a broken-document icon for a file:// URI on Windows.
    logo_path = LOGO_PATH.resolve().as_posix()
    return (
        f'<a href="{PROJECT_URL}"><img src="{logo_path}" width="72" height="72"></a>'
        '<br><font color="#91a3b0">Click the logo to open the Kyven GitHub project.</font>'
    )


def add_node_branding(node: Any, nuke: Any) -> None:
    """Add the linked logo and reliable browser-button fallback to a Kyven node."""

    if "kyven_logo" not in node.knobs():
        node.addKnob(nuke.Text_Knob("kyven_logo", "", logo_markup()))
    else:
        node["kyven_logo"].setValue(logo_markup())

    if "open_kyven_github" not in node.knobs():
        button = nuke.PyScript_Knob(
            "open_kyven_github",
            "Open Kyven on GitHub",
            "kyven_nuke.branding.open_project_page()",
        )
        button.setTooltip("Open https://github.com/vaaleerkiin/Kyven")
        button.setFlag(nuke.STARTLINE)
        node.addKnob(button)
    else:
        node["open_kyven_github"].setFlag(nuke.STARTLINE)

    # Existing nodes receive these knobs during upgrade, so explicitly keep them at the top.
    names = list(node.knobs())
    if "kyven" not in names or "kyven_logo" not in names:
        return
    anchor = names.index("kyven")
    tail = [node[name] for name in names[anchor + 1 :]]
    for knob in tail:
        node.removeKnob(knob)
    preferred = ("kyven_logo", "kyven_title", "open_kyven_github")
    for name in preferred:
        for knob in tail:
            if knob.name() == name:
                node.addKnob(knob)
    for knob in tail:
        if knob.name() not in preferred:
            node.addKnob(knob)
