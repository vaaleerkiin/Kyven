from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "hosts" / "nuke"))

from kyven_nuke.dlss_live import DlssLiveState


def test_scrubbing_is_debounced_and_latest_frame_wins():
    state = DlssLiveState()
    state.observe("frame1", 0)
    state.observe("frame2", 0.1)
    assert not state.ready(0.3, 0.25)
    assert state.ready(0.4, 0.25)
    cancellation = state.begin("frame2")
    assert not state.ready(1, 0.25)
    state.observe("frame3", 0.5)
    assert cancellation.is_set()
    assert not state.accepts("frame2")
    state.finish("frame2")
    assert state.ready(1, 0.25)
    state.begin("frame3")
    assert state.accepts("frame3")


def test_same_frame_own_setting_edit_invalidates_result_and_no_failed_retry_loop():
    state = DlssLiveState()
    state.observe("frame1-settingsA", 0)
    state.begin("frame1-settingsA")
    state.finish("frame1-settingsA")
    assert not state.ready(1, 0.25)
    state.observe("frame1-settingsB", 2)
    assert state.ready(3, 0.25)
    state.begin("frame1-settingsB")
    assert not state.accepts("frame1-settingsA")


def test_disabling_live_cancels_active_live_work_but_preserves_manual_range():
    state = DlssLiveState()
    state.observe("frame1", 0)
    live = state.begin("frame1")
    state.observe("", 1, enabled=False)
    assert live.is_set()
    state.finish("frame1")
    manual = state.begin("range1", live=False)
    state.observe("", 2, enabled=False)
    assert not manual.is_set()


def test_returning_to_same_frame_while_old_request_cancels_is_reprocessed():
    state = DlssLiveState()
    state.observe("a", 0)
    state.begin("a")
    state.observe("b", 1)
    state.observe("a", 2)
    assert not state.accepts("a")
    state.finish("a")
    # The cancelled attempt must be eligible again even if the artist returns to it.
    assert state.ready(3, 0.25)


def test_unchanged_completed_intent_never_retries_on_internal_cancellation():
    state = DlssLiveState()
    state.observe("frame400-settingsA", 0)
    cancellation = state.begin("frame400-settingsA")
    cancellation.cancel("internal_update")
    state.finish("frame400-settingsA")
    assert not state.ready(1, 0.25)
    state.observe("frame400-settingsA", 2)
    assert not state.ready(3, 0.25)
