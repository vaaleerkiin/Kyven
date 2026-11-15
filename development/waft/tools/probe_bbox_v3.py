import nuke
from pathlib import Path
root = Path(__file__).resolve().parents[1]
nuke.pluginAddPath((root / 'WAFT/Cattery/WAFT').as_posix())
nuke.root()['format'].setValue(nuke.addFormat('2048 1556 1 Project2K'))
nuke.frame(2)
read = nuke.nodes.Read(file=(root / 'build/test_input.0002.png').as_posix())
source = nuke.nodes.Reformat(inputs=[read])
source['type'].setValue('to box')
source['box_width'].setValue(1024)
source['box_height'].setValue(540)
source['box_fixed'].setValue(True)
source['resize'].setValue('distort')
node = nuke.nodes.WAFT(inputs=[source])
node['useGPUIfAvailable'].setValue(False)
node.begin()
for direction in [1, 2]:
    original = nuke.toNode('Inference' + str(direction))
    replacement = nuke.nodes.Expression(inputs=[original.input(0)])
    replacement['expr0'].setValue('2')
    replacement['expr1'].setValue('3')
    for consumer in original.dependent():
        for index in range(consumer.inputs()):
            if consumer.input(index) == original:
                consumer.setInput(index, replacement)
node.end()
def box(n):
    b = n.bbox()
    return [b.x(), b.y(), b.x()+b.w(), b.y()+b.h()]
for proxy in [False, True]:
    nuke.root()['proxy'].setValue(proxy)
    print('PROXY', proxy, flush=True)
    for name in ['Input1', 'SourceBounds', 'WorkingResolution', 'Shuffle3', 'SourceResolution', 'ScaleVectorY', 'Remove1', 'NativeMotion', 'Output1']:
        n = node.node(name)
        print('BOUNDS', name, n.width(), n.height(), box(n), flush=True)
    node['output_mode'].setValue(1)
    for xy in [(5,5), (512,270), (1018,534)]:
        print('SAMPLE', xy, node.sample('smartvector_f01_v01.n_u', *xy), flush=True)
print('REFORMAT_KNOBS', source.writeKnobs(0), flush=True)
