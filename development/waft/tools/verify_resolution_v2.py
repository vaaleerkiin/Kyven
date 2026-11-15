"""Exercise reduced-resolution inference with a mismatched project format."""
import json
from pathlib import Path
import time
import nuke

root = Path(__file__).resolve().parents[1]
nuke.pluginAddPath((root / 'WAFT/Cattery/WAFT').as_posix())
nuke.root()['format'].setValue(nuke.addFormat('3840 2160 1 TestProject4K'))
nuke.root()['first_frame'].setValue(1)
nuke.root()['last_frame'].setValue(3)
nuke.frame(2)
read = nuke.nodes.Read(file=(root / 'build/test_input.%04d.png').as_posix(), first=1, last=3)
read['raw'].setValue(True)
# Source is 1920x1080; project is 3840x2160. Deliberately stretch both axes
# differently so a single scalar correction cannot accidentally pass the test.
source = nuke.nodes.Reformat(inputs=[read])
source['type'].setValue('to box')
source['box_width'].setValue(1920)
source['box_height'].setValue(1080)
source['box_fixed'].setValue(True)
source['resize'].setValue('distort')
node = nuke.nodes.WAFT(inputs=[source])
node['in_colorspace'].setValue('sRGB')
node['useGPUIfAvailable'].setValue(True)
assert node['processing'].getValue() == 0
report = {'project': [3840, 2160], 'source': [source.width(), source.height()]}
def bounds(n):
    b = n.bbox()
    return [b.x(), b.y(), b.x()+b.w(), b.y()+b.h()]
for mode, size in [(0, (512, 288)), (1, (768, 432)), (2, (1024, 576)), (3, (1920, 1080))]:
    node['processing'].setValue(mode)
    work = node.node('WorkingResolution')
    assert (work.width(), work.height()) == size, (mode, work.width(), work.height())
    assert (node.width(), node.height()) == (1920, 1080)
    print('FORMAT_PASS', mode, size, 'output', node.width(), node.height(), flush=True)
node['processing'].setValue(0)
assert node['precision'].getValue() == 0
assert node.node('Inference1')['modelFile'].value().endswith('/WAFT.cat')
node['output_mode'].setValue(1)
write = nuke.nodes.Write(inputs=[node])
write['file'].setValue((root / 'build/test_v2_1080_smart.exr').as_posix())
write['file_type'].setValue('exr')
write['channels'].setValue('all')
write['datatype'].setValue('32 bit float')
started = time.perf_counter()
nuke.execute(write, 2, 2)
report['seconds'] = time.perf_counter() - started
result = nuke.nodes.Read(file=write['file'].value())
result['raw'].setValue(True)
assert (result.width(), result.height()) == (1920, 1080)
channels = ['smartvector_f01_v01.n_u', 'smartvector_f01_v01.n_v',
            'smartvector_f01_v01.p_u', 'smartvector_f01_v01.p_v']
samples = {c: result.sample(c, 960, 540) for c in channels}
report['smart_samples'] = samples
for channel, expected in zip(channels, [32.0, -20.25, -32.0, 20.25]):
    assert abs(samples[channel] - expected) < 2.0, (channel, samples[channel], expected)
print('SCALED_VECTORS_PASS', samples, flush=True)
# The other output should keep the source format as well, including a
# non-square-pixel source and an oversized bounding box.
node['output_mode'].setValue(0)
assert (node.width(), node.height()) == (1920, 1080)
assert bounds(node) == [0.0, 0.0, 1920.0, 1080.0], bounds(node)
write['file'].setValue((root / 'build/test_v2_1080_motion.exr').as_posix())
nuke.execute(write, 2, 2)
report['motion_bbox'] = bounds(node)
# The two modes must report source bounds even when VectorToMotion internally
# advertises the project rectangle.
assert report['motion_bbox'] == [0.0, 0.0, 1920.0, 1080.0]
node['precision'].setValue(1)
assert node.node('Inference1')['modelFile'].value().endswith('/WAFT_FP32.cat')
node['precision'].setValue(0)
assert node.node('Inference1')['modelFile'].value().endswith('/WAFT.cat')
node['useGPUIfAvailable'].setValue(False)
assert node.node('Inference1')['modelFile'].value().endswith('/WAFT_FP32.cat')
node['useGPUIfAvailable'].setValue(True)
assert node.node('Inference1')['modelFile'].value().endswith('/WAFT.cat')
source['box_pixel_aspect'].setValue(2.0)
assert abs(node.format().pixelAspect() - 2.0) < 0.001
source['box_pixel_aspect'].setValue(1.0)
nuke.scriptSave((root / 'build/WAFT_v2_1080_test.nk').as_posix())
(root / 'build/resolution_v3_validation.json').write_text(json.dumps(report, indent=2))
print('RESOLUTION_V2_PASSED', report, flush=True)
