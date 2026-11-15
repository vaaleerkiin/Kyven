"""Render both outputs using Nuke's native Inference node."""
import json
import os
from pathlib import Path
import nuke

root = Path(__file__).resolve().parents[1]
package = root / 'WAFT/Cattery/WAFT'
if os.environ.get('WAFT_TEST_INSTALLED') == '1':
    package = root.parents[1] / 'plugins/Cattery/WAFT'
if os.environ.get('WAFT_TEST_PACKAGE'):
    package = Path(os.environ['WAFT_TEST_PACKAGE'])
nuke.pluginAddPath(package.as_posix())
fmt = nuke.addFormat('3840 2160 1 WAFT_project_test')
nuke.root()['format'].setValue(fmt)
nuke.root()['first_frame'].setValue(1)
nuke.root()['last_frame'].setValue(3)
nuke.frame(2)
read = nuke.nodes.Read(file=(root / 'build/test_input.%04d.png').as_posix(), first=1, last=3)
read['raw'].setValue(True)
node = nuke.nodes.WAFT(inputs=[read])
node['in_colorspace'].setValue('sRGB')
node['useGPUIfAvailable'].setValue(True)
assert Path(node.node('Inference1')['modelFile'].value()).resolve() == (package / 'WAFT.cat').resolve()
assert (node.width(), node.height()) == (240, 160)
def bounds(n):
    b = n.bbox()
    return [b.x(), b.y(), b.x()+b.w(), b.y()+b.h()]
print('GRAPH', {key: [i.name() if i else None for i in node.node(key).dependencies()]
                for key in ['OutputMode', 'NativeMotion', 'Switch1']}, flush=True)
report = {}
for mode, label in [(0, 'motion'), (1, 'smart')]:
    node['output_mode'].setValue(mode)
    node.forceValidate()
    channels = node.channels()
    assert bounds(node) == [0.0, 0.0, 240.0, 160.0], (label, bounds(node))
    print('OUTPUT', label, channels, flush=True)
    expected = ['forward.u', 'forward.v', 'backward.u', 'backward.v'] if mode == 0 else [
        'smartvector_f01_v01.n_u', 'smartvector_f01_v01.n_v',
        'smartvector_f01_v01.p_u', 'smartvector_f01_v01.p_v']
    assert all(c in channels for c in expected), (label, channels)
    write = nuke.nodes.Write(inputs=[node], file=(root / ('build/test_' + label + '.exr')).as_posix())
    write['file_type'].setValue('exr')
    write['channels'].setValue('all')
    write['datatype'].setValue('32 bit float')
    nuke.execute(write, 2, 2)
    result = nuke.nodes.Read(file=write['file'].value())
    result['raw'].setValue(True)
    samples = {c: result.sample(c, 120, 80) for c in expected}
    for channel, reference in zip(expected, [4.0, -3.0, -4.0, 3.0]):
        assert abs(samples[channel] - reference) < 0.6, (label, channel, samples[channel])
    print('SAMPLES', label, samples, flush=True)
    report[label] = {'channels': channels, 'samples': samples}
    nuke.delete(result)
    nuke.delete(write)
node['output_mode'].setValue(0)
nuke.scriptSave((root / 'build/WAFT_test.nk').as_posix())
(root / 'build/nuke_validation.json').write_text(json.dumps(report, indent=2))
print('NATIVE_NUKE_RENDER_PASSED', flush=True)
