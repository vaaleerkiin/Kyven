"""Render installed WAFT across precision/device changes in one Nuke session."""
import json
from pathlib import Path

import nuke

root = Path(__file__).resolve().parents[1]
package = root.parents[1] / 'plugins/Cattery/WAFT'
nuke.pluginAddPath(package.as_posix())
import cattery_waft

nuke.root()['format'].setValue(nuke.addFormat('3840 2160 1 WAFT_device_test'))
read = nuke.nodes.Read(file=(root / 'build/test_input.%04d.png').as_posix(), first=1, last=3)
read['raw'].setValue(True)
node = nuke.nodes.WAFT(inputs=[read])
node['in_colorspace'].setValue('sRGB')
node['output_mode'].setValue(1)
write = nuke.nodes.Write(inputs=[node], channels='all', file_type='exr')
write['datatype'].setValue('32 bit float')
report = []
for precision, gpu in [(0, True), (1, True), (1, False), (0, False), (0, True)]:
    node['precision'].setValue(precision)
    node['useGPUIfAvailable'].setValue(gpu)
    cattery_waft._select_model(node)
    expected_model = 'WAFT.cat' if gpu and precision == 0 else 'WAFT_FP32.cat'
    for index in (1, 2):
        inference = node.node(f'Inference{index}')
        assert Path(inference['modelFile'].value()).name == expected_model
        assert inference['useGPUIfAvailable'].value() == gpu
        assert node.node(f'FlowPlanes{index}').input(0) == inference
    label = f'{precision}_{gpu}_{len(report)}'
    write['file'].setValue((root / f'build/devices_{label}.exr').as_posix())
    print('DEVICE_CASE', label, node.node('Inference2')['modelFile'].value(), flush=True)
    try:
        nuke.clearRAMCache()
        nuke.execute(write, 2, 2)
        result = nuke.nodes.Read(file=write['file'].value(), raw=True)
        values = [result.sample('smartvector_f01_v01.' + c, 120, 80)
                  for c in ('n_u', 'n_v', 'p_u', 'p_v')]
        for actual, expected in zip(values, (4, -3, -4, 3)):
            assert abs(actual - expected) < 0.6, values
        nuke.delete(result)
        report.append({'case': label, 'samples': values})
    except Exception as error:
        report.append({'case': label, 'error': str(error)})
    print('DEVICE_RESULT', report[-1], flush=True)
(root / 'build/device_validation.json').write_text(json.dumps(report, indent=2))
assert all('error' not in case for case in report), report
print('WAFT_DEVICE_MATRIX_PASSED', flush=True)
