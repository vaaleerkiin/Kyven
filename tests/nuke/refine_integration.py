"""Real Nuke output integration checks; no inference or server is used."""
import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'hosts/nuke'), str(ROOT / 'src')]
import kyven_nuke  # noqa: F401 - serialized callbacks need the package name.
import nuke
from kyven_nuke import live, refine_node
from kyven_nuke.node import _set_matte_read

report = {'checks': []}


def check(name, condition):
    assert condition, name
    report['checks'].append(name)


def fail_inference(*args, **kwargs):
    raise AssertionError('Output integration must not request inference')


def run():
    fmt = nuke.addFormat('96 64 0 0 96 64 1 RefineIntegration')
    source = nuke.nodes.Constant(format=fmt.name())
    source['color'].setValue([0.11, 0.22, 0.33, 1])
    mask = nuke.nodes.Constant(format=fmt.name())
    original = 0.25123456
    mask['color'].setValue([0.9, 0.7, 0.2, original])
    node = refine_node.create_refine_node()
    integration_controls = ('integration_section', 'roi_blend', 'roi_feather', 'integration_help')
    check('integration is hidden in the default Auto mode',
          all(not node[name].visible() for name in integration_controls))
    node.setInput(refine_node.SOURCE_INPUT, source)
    node.setInput(refine_node.MASK_INPUT, mask)
    node['crop_mode'].setValue('manual')
    check('Manual mode reveals integration controls',
          all(node[name].visible() for name in integration_controls))
    node['processing_roi'].setValue([24, 16, 72, 48])
    _set_matte_read(node, os.environ['KYVEN_NUKE_INTEGRATION_MATTE'].replace('\\', '/'), 1, 1)
    refine_node.ensure_server = fail_inference
    identity = node['kyven_uuid'].value()
    check('integration is opt-in', not node['roi_blend'].value())
    check('integration has a default feather', node['roi_feather'].value() == 64)
    check('disabled integration preserves cached result', abs(node.sample('rgba.alpha', 0, 0)-0.8) < 1e-6)
    paths = [Path(os.environ['KYVEN_NUKE_INTEGRATION_REPORT']).parent / f'{i}.png'
             for i in range(4)]
    before = refine_node._payload(node, source, *paths, frame=1)
    old_live_request = live.request_live_update
    live.request_live_update = fail_inference
    try:
        node['live_mode'].setValue(True)
        node['roi_blend'].setValue(True)
        node['roi_feather'].setValue(8)
    finally:
        node['live_mode'].setValue(False)
        live.request_live_update = old_live_request
    check('output controls leave inference request unchanged',
          before == refine_node._payload(node, source, *paths, frame=1))
    for y in range(64):
        for x in range(96):
            if 24 <= x < 72 and 16 <= y < 48:
                t = max(0, min(1, (x-24)/8, (71-x)/8, (y-16)/8, (47-y)/8))
                weight = t*t*(3-2*t)
            else:
                weight = 0
            expected = original*(1-weight) + 0.8*weight
            actual = node.sample('rgba.alpha', x, y)
            assert abs(actual-expected) < 2e-6, (x, y, actual, expected)
    check('full canvas matches smooth inward integration', True)
    check('outside ROI keeps original float alpha',
          abs(node.sample('rgba.alpha', 23, 30)-original) < 1e-7)
    check('source RGB is preserved', abs(node.sample('rgba.red', 30, 30)-0.11) < 1e-7)
    node['output_mode'].setValue('Refined Matte')
    check('matte RGB uses integrated alpha', abs(node.sample('rgba.red', 23, 30)-original) < 1e-7)
    node['output_mode'].setValue('Refined Cutout')
    check('cutout uses integrated alpha', abs(node.sample('rgba.red', 23, 30)-0.11*original) < 1e-7)
    node['output_mode'].setValue('Trimap')
    trimap = node.sample('rgba.alpha', 30, 30)
    node['roi_blend'].setValue(False)
    check('trimap is independent of integration', node.sample('rgba.alpha', 30, 30) == trimap)
    node['roi_blend'].setValue(True)
    node['output_mode'].setValue('Source + Refined Alpha')
    node['roi_feather'].setValue(0)
    check('zero feather gives an explicit hard paste', abs(node.sample('rgba.alpha', 24, 30)-0.8) < 1e-6)
    check('hard paste still keeps original outside', abs(node.sample('rgba.alpha', 23, 30)-original) < 1e-7)
    node['roi_feather'].setValue(8)
    node['processing_roi'].setValue([0, 0, 72, 48])
    check('image edges are not faded', abs(node.sample('rgba.alpha', 0, 0)-0.8) < 1e-6)
    node['processing_roi'].setValue([24, 16, 32, 24])
    node['roi_feather'].setValue(128)
    check('narrow ROI retains a full strength center', abs(node.sample('rgba.alpha', 28, 20)-0.8) < 1e-6)
    for mode in ('auto', 'full'):
        node['crop_mode'].setValue(mode)
        check(f'{mode} mode hides the entire integration section',
              all(not node[name].visible() for name in integration_controls))
        check(f'{mode} mode bypasses optional ROI integration', abs(node.sample('rgba.alpha', 0, 0)-0.8) < 1e-6)
    node['crop_mode'].setValue('manual')
    check('returning to Manual restores controls and settings',
          all(node[name].visible() for name in integration_controls)
          and node['roi_blend'].value() and node['roi_feather'].value() == 128)
    node['processing_roi'].setValue([24, 16, 72, 48])
    node['roi_feather'].setValue(8)
    refine_node._ensure_refine_output_controls(node)
    check('repair preserves integration settings and identity', node['roi_blend'].value()
          and node['roi_feather'].value() == 8 and node['kyven_uuid'].value() == identity)
    check('repair preserves cached matte path',
          node.node('KyvenMatteRead')['file'].value() == os.environ['KYVEN_NUKE_INTEGRATION_MATTE'].replace('\\', '/'))
    path = Path(os.environ['KYVEN_NUKE_INTEGRATION_REPORT']).with_suffix('.nk')
    name = node.name()
    nuke.scriptSaveAs(str(path), overwrite=1)
    nuke.scriptClear()
    nuke.scriptOpen(str(path))
    node = nuke.toNode(name)
    check('integration survives script reopen', node['roi_blend'].value()
          and abs(node.sample('rgba.alpha', 23, 30)-original) < 1e-7)


try:
    run()
    report['ok'] = True
except Exception:  # noqa: BLE001 - report failures to the external pytest runner.
    report['error'] = traceback.format_exc()
Path(os.environ['KYVEN_NUKE_INTEGRATION_REPORT']).write_text(json.dumps(report, indent=2))
