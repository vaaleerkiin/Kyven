"""Cattery callbacks. No external Python packages are needed in Nuke."""
from pathlib import Path
import nuke

_updating = False


def _replace_inference(node, inference, path, gpu):
    """Create a fresh native operator; changing modelFile retains loaded state."""
    links = [(consumer, index)
             for consumer in node.nodes()
             for index in range(consumer.inputs())
             if consumer.input(index) == inference]
    name = inference.name()
    with node:
        replacement = nuke.nodes.Inference(
            name=name + '_Reload',
            inputs=[inference.input(0)], modelFile=path,
            useGPUIfAvailable=gpu, halfPrecision=False)
        replacement.setXYpos(inference.xpos(), inference.ypos())
        nuke.delete(inference)
        for consumer, index in links:
            consumer.setInput(index, replacement)
        replacement.setName(name)


def _select_model(node, rebuild=True):
    global _updating
    if _updating:
        return
    _updating = True
    try:
        gpu = bool(node['useGPUIfAvailable'].value())
        half = gpu and node['precision'].getValue() == 0
        filename = 'WAFT.cat' if half else 'WAFT_FP32.cat'
        path = (Path(__file__).resolve().parent / filename).as_posix()
        for name in ('Inference1', 'Inference2'):
            inference = node.node(name)
            if inference is not None and not rebuild:
                inference['modelFile'].setValue(path)
                inference['useGPUIfAvailable'].clearAnimated()
                inference['useGPUIfAvailable'].setValue(gpu)
                continue
            if inference is not None and (
                inference['modelFile'].value() != path
                or inference['useGPUIfAvailable'].hasExpression()
                or inference['useGPUIfAvailable'].value() != gpu
            ):
                _replace_inference(node, inference, path, gpu)
    finally:
        _updating = False


def style_node(node):
    """Keep the DAG tile plain and place panel sections on their own rows."""
    node['icon'].setValue('')
    if node.name() == 'WAFT' or (node.name().startswith('WAFT') and node.name()[4:].isdigit()):
        node.setName('KyvenWAFT')
    if 'WAFT' in node.knobs():
        node['WAFT'].setLabel('Kyven WAFT')
    if 'kyven_title' in node.knobs():
        node['kyven_title'].setValue('<font size="5" color="#dce9f2"><b>KYVEN WAFT</b></font>')
        node['kyven_title'].clearFlag(nuke.STARTLINE)
    for name in (
        'kyven_logo', 'model_section', 'model_info', 'precision', 'useGPUIfAvailable',
        'localGPU', 'processing', 'input_section', 'in_colorspace', 'channelsIn',
        'output_section', 'output_mode', 'cleanup_section', 'grid_cleanup', 'cleanup_range',
    ):
        if name in node.knobs():
            node[name].setFlag(nuke.STARTLINE)


def on_create():
    node = nuke.thisNode()
    style_node(node)
    if 'kyven_logo' in node.knobs():
        logo = (Path(__file__).resolve().parent / 'kyven_logo.png').as_posix()
        node['kyven_logo'].setValue(
            '<a href="https://github.com/vaaleerkiin/Kyven" title="Open Kyven on GitHub">'
            f'<img src="{logo}" width="48" height="48"></a>'
        )
    _select_model(node, rebuild=False)
    inference = node.node('Inference1')
    node['channelsIn'].setValue('RGB current frame + RGB adjacent frame (sRGB)')
    if inference is not None and 'gpuName' in inference.knobs():
        node['gpuName'].setValue(inference['gpuName'].value())


def knob_changed():
    node = nuke.thisNode()
    knob = nuke.thisKnob()
    if knob is not None and knob.name() in ('precision', 'useGPUIfAvailable'):
        _select_model(node)
