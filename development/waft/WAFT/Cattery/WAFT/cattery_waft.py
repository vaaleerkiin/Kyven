"""Cattery callbacks. No external Python packages are needed in Nuke."""
from pathlib import Path
import nuke

_updating = False


def _select_model(node):
    global _updating
    if _updating:
        return
    _updating = True
    try:
        half = node['useGPUIfAvailable'].value() and node['precision'].getValue() == 0
        filename = 'WAFT.cat' if half else 'WAFT_FP32.cat'
        path = (Path(__file__).resolve().parent / filename).as_posix()
        for name in ('Inference1', 'Inference2'):
            inference = node.node(name)
            if inference is not None and inference['modelFile'].value() != path:
                inference['modelFile'].setValue(path)
    finally:
        _updating = False


def on_create():
    node = nuke.thisNode()
    if 'kyven_logo' in node.knobs():
        logo = (Path(__file__).resolve().parent / 'kyven_logo.png').as_posix()
        node['kyven_logo'].setValue(
            '<a href="https://github.com/vaaleerkiin/Kyven" title="Open Kyven on GitHub">'
            f'<img src="{logo}" width="48" height="48"></a>'
        )
    _select_model(node)
    inference = node.node('Inference1')
    node['channelsIn'].setValue('RGB current frame + RGB adjacent frame (sRGB)')
    if inference is not None and 'gpuName' in inference.knobs():
        node['gpuName'].setValue(inference['gpuName'].value())


def knob_changed():
    node = nuke.thisNode()
    knob = nuke.thisKnob()
    if knob is not None and knob.name() in ('precision', 'useGPUIfAvailable'):
        _select_model(node)
