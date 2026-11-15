import nuke
print('NUKE_VERSION', nuke.NUKE_VERSION_STRING)
node = nuke.nodes.VectorToMotion()
for key, knob in node.knobs().items():
    print(key, knob.Class(), repr(knob.value()))
