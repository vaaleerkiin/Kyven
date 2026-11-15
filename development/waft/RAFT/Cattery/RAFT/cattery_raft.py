import nuke 
import sys

def on_create():
    this = nuke.thisNode()
    inference = nuke.toNode(f"{this.name()}.Inference1")
    this["gpuName"].setValue(inference["gpuName"].value())
    this["channelsIn"].setValue("rgba.red, rgba.green, rgba.blue")
    inference.forceValidate()
    is_enabled = inference["modelFile"].enabled()
    if (sys.platform.lower() == "darwin") and (not inference["useGPUIfAvailable"].enabled()): this["useGPUIfAvailable"].setValue(False), this["useGPUIfAvailable"].setEnabled(False)
    if not is_enabled:
        for k in this.knobs(): this[k].setEnabled(False)

def knob_changed():
    this = nuke.thisNode()
    if 'halfPrecision' in this.knobs():
        this["halfPrecision"].setVisible(this["useGPUIfAvailable"].value())

def on_delete():
    nuke.thisNode()['knobChanged'].setValue("")