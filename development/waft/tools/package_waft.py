"""Build the WAFT Cattery shell from the supplied RAFT node graph."""
import json
import re
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
target = ROOT / 'WAFT/Cattery/WAFT'
target.mkdir(parents=True, exist_ok=True)
source = ROOT / 'RAFT/Cattery/RAFT'
text = (source / 'RAFT.gizmo').read_text()
text = text.replace('RAFT', 'WAFT').replace('cattery_raft', 'cattery_waft')
lines = text.splitlines()
for i, line in enumerate(lines):
    if line.startswith(' onCreate '):
        lines[i] = ' onCreate "import cattery_waft; cattery_waft.on_create()"'
    if line.startswith(' knobChanged '):
        lines[i] = ' knobChanged "import cattery_waft; cattery_waft.knob_changed()"'
text = '\n'.join(lines) + '\n'
text = text.replace('version 13.1 v1', 'version 16.0 v4')
text = text.replace('T "NVIDIA RTX A5000"', 'T ""')
text = text.replace(' addUserKnob {20 WAFT}', ''' addUserKnob {20 WAFT}
 addUserKnob {4 processing l "Processing Resolution" t "Limits the longest edge used by WAFT. Output always returns to the input source format, with vector lengths scaled accordingly. Lower settings use less VRAM and can lose fine detail." M {"512 (Low VRAM)" "768 (Balanced)" "1024 (High)" "Full Source"}}
 addUserKnob {4 precision l "Precision" t "GPU FP16 uses half precision weights and attention, with float32 flow accumulation. FP32 supports CPU. Turning off GPU automatically selects the FP32 model." M {"GPU FP16 (Fast)" "FP32 (Compatible)"}}
 addUserKnob {4 output_mode l "Output Mode" t "Motion Vectors: native Nuke motion channels via VectorToMotion. Smart Vectors: the same smartvector layer as RAFT Cattery." M {"Motion Vectors" "Smart Vectors"}}
 addUserKnob {7 grid_cleanup l "Grid Suppression" t "Edge-aware smoothing of fine grid/ripple in the vector field before resizing. 0 preserves the raw model output, 1 applies full cleanup. May remove small real motion variations." R 0 1}
 grid_cleanup 1
 addUserKnob {7 cleanup_range l "Edge Threshold" t "Vector difference, in working-resolution pixels, over which smoothing is reduced. Lower values preserve smaller motion boundaries; higher values suppress stronger ripple." R 0.01 2}
 cleanup_range 0.25
 addUserKnob {26 model_info l "Model" T "WAFT-a1 / 5 iterations / v5"}''')
text = text.replace('  serialiseKnob {iters:12;}\n', '')
# Both directions must honor the same precision setting.
text = text.replace('  name Inference1', '  halfPrecision {{parent.halfPrecision}}\n  name Inference1')
text = '\n'.join(line for line in text.splitlines() if 'addUserKnob {6 halfPrecision' not in line) + '\n'
text = text.replace('halfPrecision {{parent.halfPrecision}}', 'halfPrecision false')
# Filter each 2D flow independently, so forward/backward boundaries do not
# influence one another. Remove the original blue image plane first: bilateral
# range weights must depend on displacement, not image texture.
def add_vector_cleanup(match):
    index = re.search(r'name Inference(\d)', match.group(0))[1]
    return match.group(0) + '''
 Remove {
  operation keep
  channels {rgba.red rgba.green}
  name FlowPlanes%s
 }
 Bilateral {
  Size 2
  PositionalSigma 0.7
  ColorSigma {{parent.cleanup_range}}
  mix {{parent.grid_cleanup}}
  disable {{parent.grid_cleanup==0}}
  name VectorCleanup%s
 }''' % (index, index)
text = re.sub(r' Inference \{[\s\S]*?\n \}', add_vector_cleanup, text)
# Crop oversized bounds before inference; explicitly use the connected source.
text = text.replace('  name Colorspace1\n', '  name Colorspace1\n')
needle = ' Dot {\n  name Dot9'
text = text.replace(needle, ''' Crop {
  box {0 0 {input.width} {input.height}}
  reformat false
  name SourceBounds
 }
 Reformat {
  type "to box"
  box_width {{max(1,floor(input.width*(parent.processing==3?1:min(1,(parent.processing==0?512:parent.processing==1?768:1024)/max(input.width,input.height)))))}}
  box_height {{max(1,floor(input.height*(parent.processing==3?1:min(1,(parent.processing==0?512:parent.processing==1?768:1024)/max(input.width,input.height)))))}}
  box_fixed true
  resize distort
  filter Lanczos4
  black_outside false
  name WorkingResolution
 }
 Dot {
  name Dot9''', 1)
# Restore the vector field's spatial resolution, then its displacement units.
text = text.replace('add_layer {smartvector', ''' Reformat {
  type "to box"
  box_width {{parent.Input1.width}}
  box_height {{parent.Input1.height}}
  box_fixed true
  box_pixel_aspect {{parent.Input1.pixel_aspect}}
  resize distort
  filter Cubic
  black_outside false
  name SourceResolution
 }
 Multiply {
  channels {forward.u backward.u}
  value {{parent.Input1.width/parent.WorkingResolution.width}}
  name ScaleVectorX
 }
 Multiply {
  channels {forward.v backward.v}
  value {{parent.Input1.height/parent.WorkingResolution.height}}
  name ScaleVectorY
 }
set waft_vector_bounds [stack 0]
add_layer {smartvector''', 1)
text = text.replace(' Switch {\n  inputs 2\n  which', '''set waft_smart [stack 0]
push $waft_smart
 VectorToMotion {
  name NativeMotion
  xpos 1040
  ypos 505
 }
 Remove {
  operation keep
  channels motion
  name MotionOnly
 }
 Switch {
  inputs 2
  which {{parent.output_mode}}
  name OutputMode
  xpos 1010
  ypos 560
 }
 Switch {
  inputs 2
  which''')
# VectorToMotion's internal graph advertises the project BBox. Constrain the
# final output after both branches, rather than relying on its input format.
text = text.replace(' Output {\n  name Output1', '''set waft_output [stack 0]
push $waft_vector_bounds
push $waft_output
 CopyBBox {
  inputs 2
  name OutputBounds
 }
 Output {
  name Output1''', 1)
# Keep the native graph and existing control names while matching Kyven's layout.
ui_start = text.index(' addUserKnob {20 WAFT}')
ui_end = text.index('\n}', ui_start)
controls = {}
for line in text[ui_start:ui_end].splitlines():
    match = re.match(r' addUserKnob \{\d+ (\w+)\b', line)
    if match:
        controls[match[1]] = line


def section(name, title):
    return f' addUserKnob {{26 {name} l "" +STARTLINE T "<br><font color=\'#9fc7e8\'><b>{title}</b></font>"}}'


layout = [
    ' addUserKnob {20 WAFT l "Kyven WAFT"}',
    ' addUserKnob {26 kyven_logo l "" +STARTLINE}',
    ' addUserKnob {26 kyven_title l "" -STARTLINE T "<font size=\'5\' color=\'#dce9f2\'><b>KYVEN WAFT</b></font>"}',
    section('model_section', 'MODEL AND PERFORMANCE'),
    controls['model_info'], controls['precision'], controls['useGPUIfAvailable'],
    ' useGPUIfAvailable true', controls['localGPU'], controls['gpuName'],
    controls['processing'],
    section('input_section', 'INPUT'),
    controls['in_colorspace'], controls['channelsIn'],
    section('output_section', 'OUTPUT'),
    controls['output_mode'],
    section('cleanup_section', 'VECTOR CLEANUP'),
    controls['grid_cleanup'], ' grid_cleanup 1',
    controls['cleanup_range'], ' cleanup_range 0.25',
]
text = text[:ui_start] + '\n'.join(layout) + text[ui_end:]
(target / 'WAFT.gizmo').write_text(text, encoding='utf-8')
(target / 'cat.json').write_text(json.dumps({
    'minimum_nuke_version_required': 16.0,
    'version': 5,
    'category': 'Optical Flow',
    'description': 'Warping-Alone Field Transforms. WAFT-a1, motion and SmartVector outputs.',
    'cats': [{'filepath': 'WAFT.gizmo', 'name': 'WAFT', 'icon': 'WAFT.png'}]
}, indent=4))
icons = ROOT.parents[1] / 'hosts/nuke/icons'
for suffix in ['.png', '@2x.png']:
    shutil.copyfile(icons / 'kyven_logo_icon.png', target / ('WAFT' + suffix))
shutil.copyfile(icons / 'kyven_logo.png', target / 'kyven_logo.png')
shutil.copyfile(ROOT / 'WAFT-source/LICENSE', target / 'LICENSE-WAFT.txt')
print(target)
