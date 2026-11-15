import nuke
from pathlib import Path
root = Path(__file__).resolve().parents[1]
nuke.pluginAddPath((root / 'WAFT/Cattery/WAFT').as_posix())
nuke.root()['format'].setValue(nuke.addFormat('3840 2160 1 Project4K'))
r = nuke.nodes.Read(file=(root / 'build/test_input.0002.png').as_posix())
r['raw'].setValue(True)
n = nuke.nodes.WAFT(inputs=[r])
n['useGPUIfAvailable'].setValue(False)
print('ROOT', nuke.root().format().width(), nuke.root().format().height())
for item in [r, n] + n.nodes():
    try:
        print('FORMAT', item.name(), item.Class(), item.width(), item.height(), item.bbox())
    except Exception as exc:
        print(item.name(), str(exc))
for cls in ['Reformat', 'Crop', 'Inference']:
    node = getattr(nuke.nodes, cls)()
    print('KNOBS', cls)
    for key, knob in node.knobs().items():
        if key not in ['help', 'knobChanged', 'onCreate']:
            print(key, knob.Class(), repr(knob.value()))
