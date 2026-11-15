"""Check Kyven startup registration without adding the WAFT path manually."""
from pathlib import Path
import nuke

root = Path(__file__).resolve().parents[3]
package = root / 'plugins/Cattery/WAFT'
assert package.resolve() in [Path(p).resolve() for p in nuke.pluginPath()], nuke.pluginPath()
node = nuke.nodes.WAFT()
assert node.name().startswith('KyvenWAFT'), node.name()
assert node['icon'].value() == ''
for name in ('model_section', 'input_section', 'output_section', 'cleanup_section'):
    assert node[name].getFlag(nuke.STARTLINE), name
custom = nuke.nodes.WAFT(name='ShotFlow')
import cattery_waft
cattery_waft.style_node(custom)
assert custom.name() == 'ShotFlow'
for name in ('Inference1', 'Inference2'):
    assert Path(node.node(name)['modelFile'].value()).resolve() == (package / 'WAFT.cat').resolve()
node['useGPUIfAvailable'].setValue(False)
# Terminal mode does not always dispatch knobChanged callbacks.
cattery_waft._select_model(node)
for name in ('Inference1', 'Inference2'):
    assert Path(node.node(name)['modelFile'].value()).resolve() == (package / 'WAFT_FP32.cat').resolve()
print('KYVEN_WAFT_STARTUP_AND_MODEL_SELECTION_PASSED', flush=True)
