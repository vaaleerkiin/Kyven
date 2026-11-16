"""Executed by real Nuke, never imported by ordinary pytest collection."""
import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'hosts/nuke'), str(ROOT/'src')]
import kyven_nuke  # noqa: F401 - supplies the name used by serialized Nuke callbacks.
import nuke
from kyven_nuke import dlss_node, inpaint_node, node, refine_node
from kyven_nuke.dlss_interface import restyle_dlss_node

report = {'nuke': nuke.NUKE_VERSION_STRING, 'checks': []}


def check(name, condition):
    assert condition, name
    report['checks'].append(name)


def run():
    fmt = nuke.addFormat('12 10 0 0 12 10 1 KyvenSmoke')
    source = nuke.nodes.Constant(format=fmt.name())
    mask = nuke.nodes.Expression(inputs=[source], expr0='0', expr1='0', expr2='0',
        expr3='x>=3 && x<9 && y>=2 && y<8 ? 0.000001 : 0')
    refine = refine_node.create_refine_node()
    refine.setInput(refine_node.SOURCE_INPUT, source)
    refine.setInput(refine_node.MASK_INPUT, mask)
    refine['output_mode'].setValue('Trimap')
    refine['crop_mode'].setValue('full')
    check('refine has no channel menu', 'mask_channel' not in refine.knobs())
    for foreground, background in ((0, 0), (1, 1), (1, 2), (2, 1), (1.9, 2.9)):
        refine['foreground_radius'].setValue(foreground)
        refine['background_radius'].setValue(background)
        for y in range(10):
            for x in range(12):
                def support(radius, x=x, y=y):
                    return [3 <= min(11, max(0, x+dx)) < 9 and 2 <= min(9, max(0, y+dy)) < 8
                        for dx in range(-int(radius), int(radius)+1)
                        for dy in range(-int(radius), int(radius)+1)]
                expected = 1 if all(support(foreground)) else 128/255 if any(support(background)) else 0
                actual = refine.sample('rgba.red', x, y)
                assert abs(actual-expected) < 1e-6, (foreground, background, x, y, expected, actual)
        check(f'native morphology {foreground}/{background} matches binary reference', True)
    for alpha, expected in (('0', 0), ('1', 1)):
        mask['expr3'].setValue(alpha)
        refine['foreground_radius'].setValue(2)
        refine['background_radius'].setValue(2)
        for x, y in ((0, 0), (11, 9), (6, 5)):
            check(f'constant alpha {alpha} at image edge {x}/{y}',
                  abs(refine.sample('rgba.red', x, y)-expected) < 1e-6)
    mask['expr3'].setValue('x>=3 && x<9 && y>=2 && y<8 ? 0.000001 : 0')
    refine['foreground_radius'].setValue(0)
    refine['background_radius'].setValue(0)
    check('snapshot keeps faint alpha separately',
          abs(refine.node('KyvenRefineMaskExport').sample('rgba.alpha', 5, 5) - 0.000001) < 1e-10)
    check('snapshot RGB keeps faint alpha support before quantization',
          refine.node('KyvenRefineMaskExport').sample('rgba.red', 5, 5) == 1)
    refine['crop_mode'].setValue('manual')
    refine['processing_roi'].setValue([4, 3, 8, 7])
    check('native guide is black outside ROI', refine.sample('rgba.red', 3, 2) == 0)
    check('native guide retains alpha inside ROI', refine.sample('rgba.red', 5, 5) == 1)
    refine['processing_roi'].setValue([4.2, 3.2, 7.8, 6.8])
    check('fractional ROI uses backend floor/ceil bounds', refine.sample('rgba.red', 4, 3) == 1)
    refine['generate_trimap'].setValue(False)
    for value, expected in ((0, 0), (0.1, 0), (0.5, 128/255), (0.9, 1), (1, 1)):
        mask['expr3'].setValue(str(value))
        check(f'manual guide {value}', abs(refine.sample('rgba.red', 5, 5)-expected) < 1e-6)
    legacy = nuke.Enumeration_Knob('mask_channel', 'Old Mask Channel', ['Alpha', 'Red'])
    refine.addKnob(legacy)
    legacy.setValue(1)
    refine.node('KyvenMaskChannelSwitch')['which'].setExpression('parent.mask_channel')
    refine_node._ensure_refine_output_controls(refine)
    check('legacy refine selector hidden', not legacy.visible())
    check('legacy refine reads alpha', refine.node('KyvenMaskChannelSwitch').sample('rgba.red', 5, 5) == 1)

    inpaint = inpaint_node.create_inpaint_node()
    inpaint.setInput(inpaint_node.SOURCE_INPUT, source)
    inpaint.setInput(inpaint_node.MASK_INPUT, mask)
    check('inpaint has no channel menu', 'mask_channel' not in inpaint.knobs())
    for _ in range(2):
        inpaint_node._ensure_input_color_control(inpaint)
        node._place_knob_after(inpaint, 'in_colorspace', 'model')
    check('colorspace link survives interface edits',
          inpaint.knobs()['in_colorspace'].Class() == 'Link_Knob')
    check('inpaint reads alpha rather than RGB',
          inpaint.node('KyvenInpaintMaskChannel').sample('rgba.red', 5, 5) == 1)

    segment = node.create_segment_node()
    check('segment edge guide labels', 'Edge Guide' in segment['output_mode'].values())
    segment['output_mode'].setValues(['Source + Alpha', 'Alpha', 'Cutout', 'Source + Trimap', 'Trimap', 'Bypass'])
    segment['output_mode'].setValue('Trimap')
    node._ensure_output_controls(segment)
    check('legacy segment output retains selection', segment['output_mode'].value() == 'Edge Guide')

    dlss = dlss_node.create_dlss_node()
    dlss['live_mode'].setValue(False)
    check('DLSS starts with backward channels', dlss['motion_x'].value() == 'backward.u')
    dlss['motion_x'].setValue('R')
    dlss['motion_y'].setValue('G')
    dlss['dlss_ui_version'].setValue(8)
    restyle_dlss_node(dlss, upgrade=True)
    check('legacy DLSS preserves RGB vector mapping', dlss['motion_x'].value() == 'R')

    identities = {current.name(): current['kyven_uuid'].value()
                  for current in (refine, inpaint, segment, dlss)}
    inpaint_name = inpaint.name()
    script = Path(os.environ['KYVEN_NUKE_SMOKE_REPORT']).with_suffix('.nk')
    nuke.scriptSaveAs(str(script), overwrite=1)
    nuke.scriptClear()
    nuke.scriptOpen(str(script))
    node.repair_segment_nodes()
    dlss_node.repair_dlss_nodes()
    for name, identity in identities.items():
        check(f'{name} survives reopen', nuke.toNode(name)['kyven_uuid'].value() == identity)
    check('reopened inpaint colorspace link',
          nuke.toNode(inpaint_name).knobs()['in_colorspace'].Class() == 'Link_Knob')


try:
    run()
    report['ok'] = True
except Exception:  # noqa: BLE001 - report failures to the external pytest runner.
    report['ok'] = False
    report['error'] = traceback.format_exc()
Path(os.environ['KYVEN_NUKE_SMOKE_REPORT']).write_text(json.dumps(report, indent=2))
print('KYVEN_NUKE_SMOKE', json.dumps(report))
