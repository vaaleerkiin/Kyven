"""Interactive preview dispatch must not pile up work during a slider drag."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).parents[1] / 'hosts' / 'nuke'))
from kyven_nuke import refine_tile_preview as preview


def test_slider_burst_dispatches_the_latest_revision_once(monkeypatch):
    timers = []

    class Timer:
        def __init__(self, delay, callback):
            self.callback = callback
            timers.append(self)

        def start(self):
            pass

    monkeypatch.setattr(preview, 'threading', SimpleNamespace(Timer=Timer))
    monkeypatch.setattr(preview, '_scheduled', set())
    monkeypatch.setattr(preview, '_running', set())
    monkeypatch.setattr(preview, '_revisions', {})
    start = Mock()
    monkeypatch.setattr(preview, '_start', start)
    monkeypatch.setattr(preview, '_nuke', lambda: SimpleNamespace(
        executeInMainThread=lambda callback, args: callback(*args)))

    for revision in range(1, 201):
        preview._revisions['Refine'] = revision
        preview._schedule('Refine', revision, 0.03)
    assert len(timers) == 1
    assert not start.called
    timers[0].callback()
    start.assert_called_once_with('Refine', 200)
    assert not preview._scheduled

    preview._running.add('Refine')
    for revision in range(201, 401):
        preview._revisions['Refine'] = revision
        preview._schedule('Refine', revision, 0.03)
    assert len(timers) == 1  # Completion dispatches the latest pending edit.
