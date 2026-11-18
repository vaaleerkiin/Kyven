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
        expr3='x>=3 && x<9 && y>=2 && y<8 ? 1 : 0')
    refine = refine_node.create_refine_node()
    for control in ('tile_size', 'tile_overlap', 'context_padding'):
        check(f'{control} is a full-width slider',
              isinstance(refine[control], nuke.Double_Knob)
              and bool(refine[control].getFlag(nuke.STARTLINE)))
    check('Context Padding keeps its original default', refine['context_padding'].value() == 128)
    check('tile display defaults off', not refine['show_tiles'].value())
    # A legacy integer knob may contain animated padding values.
    callback = refine['knobChanged'].value()
    refine['knobChanged'].setValue('')
    refine.removeKnob(refine['context_padding'])
    legacy_padding = nuke.Int_Knob('context_padding', 'Context Padding')
    legacy_padding.setAnimated()
    legacy_padding.setValueAt(16, 1)
    legacy_padding.setValueAt(96, 2)
    refine.addKnob(legacy_padding)
    refine['knobChanged'].setValue(callback)
    refine_node._ensure_refine_crop_controls(refine)
    check('padding slider upgrade preserves animation',
          isinstance(refine['context_padding'], nuke.Double_Knob)
          and refine['context_padding'].valueAt(1) == 16
          and refine['context_padding'].valueAt(2) == 96)
    refine['context_padding'].clearAnimated()
    refine['context_padding'].setValue(128)
    check('tiling has only Tiles and Full Crop',
          tuple(refine['tile_mode'].values()) == ('Tiles', 'Full Crop'))
    check('tiling defaults to explicit Tiles', refine['tile_mode'].value() == 'Tiles')
    check('summary displays the chosen tile size', '1024 x 1024' in refine['tiling_help'].value())
    check('tile-size slider is always visible', refine['tile_size'].visible())
    check('known-tile skipping defaults on', refine['skip_known_tiles'].value())
    refine['profile'].setValue('low_memory')
    check('profile changes do not change explicit tile size', refine_node._tile_size(refine) == 1024)
    refine['tile_size'].setValue(150)
    refine['tile_overlap'].setValue(13)
    refine['tile_mode'].setValue('Full Crop')
    check('Full Crop explicitly disables tiling', refine_node._tile_size(refine) == 0)
    check('Full Crop disables remembered sliders', refine['tile_overlap'].visible()
          and not refine['tile_overlap'].enabled() and not refine['tile_size'].enabled())
    refine['tile_mode'].setValue('Tiles')
    check('returning to Tiles preserves 150/13', refine['tile_size'].value() == 150
          and refine['tile_overlap'].value() == 13)
    refine['tile_size'].setValue(128)
    refine['tile_overlap'].setValue(64)
    check('invalid overlap explains the valid range', '0 to 63' in refine['tiling_help'].value())
    refine['tile_overlap'].setValue(32)
    check('Tiles explains tile stride', 'Step 96 px' in refine['tiling_help'].value())
    refine['profile'].setValue('quality')
    check('Quality does not silently select Full Crop', refine_node._tile_size(refine) == 128)
    # Upgrade the previous three-mode selector without changing a chosen 150/13.
    callback = refine['knobChanged'].value()
    refine['knobChanged'].setValue('')
    refine['tile_mode'].setValues(['Auto (Profile)', 'Custom', 'Full Crop'])
    refine['tile_mode'].setValue('Custom')
    refine['tile_size'].setValue(150)
    refine['tile_overlap'].setValue(13)
    refine['knobChanged'].setValue(callback)
    from kyven_nuke.refine_tiling import ensure_controls as ensure_tiling_controls
    ensure_tiling_controls(refine)
    check('legacy custom values survive removal of Auto', refine['tile_mode'].value() == 'Tiles'
          and refine_node._tile_size(refine) == 150 and refine['tile_overlap'].value() == 13)
    # Resolve old Auto once, then keep the resulting size independent of profiles.
    refine['knobChanged'].setValue('')
    refine['tile_mode'].setValues(['Auto (Profile)', 'Custom', 'Full Crop'])
    refine['tile_mode'].setValue('Auto (Profile)')
    refine['profile'].setValue('low_memory')
    refine['knobChanged'].setValue(callback)
    ensure_tiling_controls(refine)
    check('legacy Auto becomes explicit 512 px Tiles', refine['tile_mode'].value() == 'Tiles'
          and refine_node._tile_size(refine) == 512)
    refine['knobChanged'].setValue('')
    refine['tile_mode'].setValues(['Auto (Profile)', 'Custom', 'Full Crop'])
    refine['tile_mode'].setValue('Auto (Profile)')
    refine['profile'].setValue('quality')
    refine['knobChanged'].setValue(callback)
    ensure_tiling_controls(refine)
    check('legacy Quality Auto remains a Full Crop pass', refine['tile_mode'].value() == 'Full Crop')
    refine['tile_mode'].setValue('Tiles')
    refine['tile_size'].setValue(1024)
    refine['profile'].setValue('balanced')
    refine['tile_overlap'].setValue(64)
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
    check('snapshot RGB preserves original alpha without binarization',
          abs(refine.node('KyvenRefineMaskExport').sample('rgba.red', 5, 5) - 0.000001) < 1e-10)
    check('faint alpha is background with Cattery cutoffs', refine.sample('rgba.red', 5, 5) == 0)
    mask['expr3'].setValue('x>=3 && x<9 && y>=2 && y<8 ? 1 : 0')
    refine['crop_mode'].setValue('manual')
    refine['processing_roi'].setValue([4, 3, 8, 7])
    check('native guide is black outside ROI', refine.sample('rgba.red', 3, 2) == 0)
    check('native guide retains alpha inside ROI', refine.sample('rgba.red', 5, 5) == 1)
    refine['processing_roi'].setValue([4.2, 3.2, 7.8, 6.8])
    check('fractional ROI uses backend floor/ceil bounds', refine.sample('rgba.red', 4, 3) == 1)
    refine['generate_trimap'].setValue(False)
    for value, expected in ((0, 0), (0.001, 0), (0.1, 128/255), (0.5, 128/255),
                            (0.9, 128/255), (0.999, 1), (1, 1)):
        mask['expr3'].setValue(str(value))
        check(f'manual guide {value}', abs(refine.sample('rgba.red', 5, 5)-expected) < 1e-6)

    # Equivalent to the trimap-only branch of the installed Cattery ViTMatte gizmo:
    # grayscale morphology -> Dissolve 0.5 -> 0.01/0.99 thresholds.
    erode = nuke.nodes.Dilate(inputs=[mask])
    erode['channels'].setValue('alpha')
    dilate = nuke.nodes.Dilate(inputs=[mask])
    dilate['channels'].setValue('alpha')
    average = nuke.nodes.Dissolve(inputs=[erode, dilate])
    average['which'].setValue(0.5)
    reference = nuke.nodes.Expression(inputs=[average])
    reference['expr3'].setValue('a < 0.01 ? 0 : (a > 0.99 ? 1 : 128.0/255)')
    refine['generate_trimap'].setValue(True)
    refine['crop_mode'].setValue('full')
    mask['expr3'].setValue('x<2 ? 0.001 : x<4 ? 0.1 : x<6 ? 0.5 : x<8 ? 0.9 : 1')
    report['refine_exports'] = []
    export_root = Path(os.environ['KYVEN_NUKE_SMOKE_REPORT']).parent
    for foreground, background in ((0, 0), (1, 2), (2, 1)):
        refine['foreground_radius'].setValue(foreground)
        refine['background_radius'].setValue(background)
        erode['size'].setValue(-foreground)
        dilate['size'].setValue(background)
        samples = []
        for y in range(10):
            row = []
            for x in range(12):
                expected = reference.sample('rgba.alpha', x, y)
                actual = refine.sample('rgba.red', x, y)
                assert abs(actual-expected) < 1e-6, (foreground, background, x, y, expected, actual)
                row.append(round(expected*255))
            samples.append(row)
        check(f'Cattery soft-mask morphology {foreground}/{background}', True)
        writer = refine.node('KyvenRefineMaskWrite')
        refine_node._configure_export_writers(None, writer)
        path = export_root / f'refine-mask-{foreground}-{background}.tif'
        writer['file'].setValue(str(path).replace('\\', '/'))
        nuke.execute(writer, 1, 1)
        report['refine_exports'].append({'mask': str(path), 'foreground_radius': foreground,
                                       'background_radius': background, 'expected': samples[::-1]})
    mask['expr3'].setValue('1')
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
