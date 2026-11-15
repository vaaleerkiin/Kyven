"""Guard hover-help coverage as public node controls are added."""
import ast
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / 'hosts' / 'nuke'))
from kyven_nuke.tooltips import apply_tooltips, tooltip_for

# Layout text and serialized internal state are not adjustable public controls.
INTERNAL = {
    'kyven', 'kyven_title', 'kyven_uuid', 'kyven_job_id', 'kyven_trimap_preview_frame',
    'positive_point_count', 'negative_point_count', 'correction_frame_count',
    'correction_frames', 'key_frame_previous', 'roi_enabled', 'processing_size',
    'dlss_ui_version', 'legacy_mask_source', 'worker', 'max_dimension', 'reset_epoch',
    'mask_channel',  # DLSS hides this; other nodes are checked separately below.
}


@pytest.mark.parametrize('kind,filename', [
    ('segment', 'node.py'), ('refine', 'refine_node.py'),
    ('inpaint', 'inpaint_node.py'), ('dlss', 'dlss_node.py'),
    ('dlss', 'dlss_interface.py'),
])
def test_public_controls_have_help(kind, filename):
    tree = ast.parse((ROOT / 'hosts' / 'nuke' / 'kyven_nuke' / filename).read_text(encoding='utf-8-sig'))
    controls = set()
    for call in ast.walk(tree):
        if not isinstance(call, ast.Call):
            continue
        if isinstance(call.func, ast.Attribute) and call.func.attr.endswith('_Knob'):
            if call.func.attr in {'Text_Knob', 'Tab_Knob'}:
                continue
            arg = call.args[0] if call.args else None
        elif isinstance(call.func, ast.Name) and call.func.id in {'add', 'ensure'}:
            if len(call.args) < 2 or not isinstance(call.args[0], ast.Constant):
                continue
            knob_type = call.args[0].value
            if not isinstance(knob_type, str) or not knob_type.endswith('_Knob') or knob_type in {'Text_Knob', 'Tab_Knob'}:
                continue
            arg = call.args[1]
        else:
            continue
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            controls.add(arg.value)
    assert not {name for name in controls - INTERNAL if not tooltip_for(name, kind)}
    if kind in {'refine', 'inpaint'}:
        assert tooltip_for('mask_channel', kind)


@pytest.mark.parametrize('kind', ['segment', 'refine', 'inpaint', 'dlss'])
def test_repair_sets_contextual_help_without_changing_values(kind):
    model, output, builtin = Mock(), Mock(), Mock()
    node = SimpleNamespace(knobs=lambda: {'model': model, 'output_mode': output, 'label': builtin})
    apply_tooltips(node, kind)
    model.setTooltip.assert_called_once_with(tooltip_for('model', kind))
    output.setTooltip.assert_called_once_with(tooltip_for('output_mode', kind))
    model.setValue.assert_not_called()
    builtin.setTooltip.assert_not_called()


@pytest.mark.parametrize('name', ['positive_point', 'positive_point_32', 'negative_point_12',
                                  'remove_positive_point_32', 'correction_frame_32', 'remove_correction_32'])
def test_dynamic_prompt_help(name):
    assert tooltip_for(name, 'segment')
