import nuke
for name in ['Median', 'Bilateral', 'Blur']:
    try:
        node = getattr(nuke.nodes, name)()
        print('NODE', name, flush=True)
        for key in node.knobs():
            if key in ['help', 'onCreate', 'knobChanged', 'onDestroy']:
                continue
            knob = node[key]
            print(key, knob.Class(), repr(knob.value()), knob.tooltip(), flush=True)
    except Exception as exc:
        print('ERROR', name, repr(exc), flush=True)
