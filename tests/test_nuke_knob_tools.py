"""Real Nuke follows links in node[name]; UI edits must use the owned knobs."""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'hosts/nuke'))

from kyven_nuke.knob_tools import suspend_callbacks
from kyven_nuke.mask_input import force_alpha_mask
from kyven_nuke.node import _place_knob_after


@pytest.mark.parametrize('raises', [False, True])
def test_callbacks_are_suspended_and_restored_even_after_failure(raises):
    knob = Mock()
    knob.value.return_value = 'existing.callback()'
    node = SimpleNamespace(knobs=lambda: {'knobChanged': knob})

    @suspend_callbacks
    def operation(current):
        knob.setValue.assert_called_with('')
        if raises:
            raise RuntimeError('edit failed')
        return 'done'
    if raises:
        with pytest.raises(RuntimeError, match='edit failed'):
            operation(node)
    else:
        assert operation(node) == 'done'
    knob.setValue.assert_called_with('existing.callback()')


@pytest.mark.parametrize('legacy', [False, True])
def test_alpha_mask_selector_has_no_legacy_expression(legacy):
    selector, legacy_control = Mock(), Mock()
    node = SimpleNamespace(knobs=lambda: {'mask_channel': legacy_control} if legacy else {})
    force_alpha_mask(node, {'which': selector})
    selector.clearAnimated.assert_called_once()
    selector.setValue.assert_called_once_with(0)
    if legacy:
        legacy_control.clearAnimated.assert_called_once()
        legacy_control.setValue.assert_called_once_with(0)
        legacy_control.setVisible.assert_called_once_with(False)


def test_reordering_moves_link_itself_not_its_target():
    class Node:
        def __init__(self):
            self.owned = {name: SimpleNamespace(name=lambda value=name: value)
                          for name in ('anchor', 'link', 'move')}
            self.target = SimpleNamespace(name=lambda: 'internal.colorspace')

        def knobs(self):
            return self.owned

        def __getitem__(self, name):
            return self.target if name == 'link' else self.owned[name]

        def removeKnob(self, knob):
            assert knob is not self.target
            self.owned.pop(knob.name())

        def addKnob(self, knob):
            self.owned[knob.name()] = knob

    node = Node()
    _place_knob_after(node, 'move', 'anchor')
    assert list(node.knobs()) == ['anchor', 'move', 'link']
