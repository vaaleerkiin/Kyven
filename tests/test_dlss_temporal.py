from __future__ import annotations

import numpy as np
import pytest

from kyven.cancellation import CancellationToken
from kyven.dlss.history import TemporalBuffer, TemporalFrame
from kyven.dlss.temporal import reproject
from kyven.errors import KyvenError


def sample(delta=0.2, *, source=None, mask=None, motion=None, forward=None, depth=None):
    source = np.full((4, 8, 3), 0.2, np.float32) if source is None else source
    return TemporalFrame(source, np.full_like(source, delta),
                         np.ones(source.shape[:2], np.float32) if mask is None else mask,
                         np.zeros((*source.shape[:2], 2), np.float32) if motion is None else motion,
                         forward, depth)


def accumulate(buffer, current, **kwargs):
    return buffer.accumulate(current.residual, current.source, current.mask, current.motion,
                             strength=0.75, rejection=0.1, depth=current.depth, **kwargs)


def test_finite_raw_buffer_reduces_alternating_correction_flicker():
    buffer = TemporalBuffer()
    raw, generated = [], []
    for index in range(24):
        current = sample(0.2 if index % 2 else -0.2)
        result, _ = accumulate(buffer, current)
        raw.append(float(current.residual.mean()))
        generated.append(float(result.mean()))
        buffer.append(current, 3, 1024**2)
    assert np.std(generated[4:]) < np.std(raw[4:]) * 0.2
    assert len(buffer.frames) == 3


def test_history_expires_without_recursively_recycling_blended_outputs():
    buffer = TemporalBuffer()
    buffer.append(sample(1), 2, 1024**2)
    results = []
    for _ in range(4):
        current = sample(0)
        result, _ = accumulate(buffer, current, clamp=0)
        results.append(float(result.mean()))
        buffer.append(current, 2, 1024**2)
    assert results[0] > results[1] > 0
    assert results[2:] == [0, 0]


def test_buffer_is_bounded_by_bytes_and_evicts_oldest_before_append():
    buffer = TemporalBuffer()
    size = sample().nbytes
    for index in range(10):
        buffer.append(sample(index), 4, size * 2)
        assert buffer.nbytes <= size * 2
    assert len(buffer.frames) == 2
    assert [frame.residual[0, 0, 0] for frame in buffer.frames] == [8, 9]
    with pytest.raises(ValueError, match="budget"):
        buffer.append(sample(), 3, size - 1)


def test_chained_vectors_align_older_samples_without_blending_original_rgb():
    buffer = TemporalBuffer()
    motion = np.zeros((4, 8, 2), np.float32)
    motion[..., 0] = -1
    oldest = sample(0, motion=motion)
    oldest.residual[:, 1] = 0.2
    previous = sample(0, motion=motion)
    previous.residual[:, 2] = 0.2
    buffer.append(oldest, 3, 1024**2)
    buffer.append(previous, 3, 1024**2)
    current = sample(0, motion=motion)
    original_source = current.source.copy()
    result, stats = accumulate(buffer, current, clamp=0)
    np.testing.assert_allclose(result[:, 3], 0.15)
    np.testing.assert_array_equal(result[:, [0, 1, 2, 4, 5, 6, 7]], 0)
    assert stats["temporal_samples_used"] == 2
    np.testing.assert_array_equal(current.source, original_source)


def test_forward_backward_check_rejects_wrong_vectors_in_repeating_flat_regions():
    buffer = TemporalBuffer()
    forward = np.zeros((4, 8, 2), np.float32)
    buffer.append(sample(0.2, forward=forward), 3, 1024**2)
    current = sample(0)
    current.motion[..., 0] = -2
    result, stats = accumulate(buffer, current, consistency=True, consistency_threshold=1)
    np.testing.assert_array_equal(result, 0)
    assert stats["temporal_reused_fraction"] == 0
    forward[..., 0] = 2
    result, stats = accumulate(buffer, current, consistency=True, consistency_threshold=1, clamp=0)
    np.testing.assert_array_equal(result[:, :2], 0)
    np.testing.assert_allclose(result[:, 2:], 0.15)
    assert stats["temporal_reused_fraction"] == 0.75


def test_depth_disocclusion_and_intermediate_mask_change_reject_old_history():
    buffer = TemporalBuffer()
    depth = np.ones((4, 8), np.float32)
    buffer.append(sample(0.2, depth=depth), 3, 1024**2)
    changed_mask = np.ones((4, 8), np.float32)
    changed_mask[0] = 0
    buffer.append(sample(0, mask=changed_mask, depth=depth), 3, 1024**2)
    current_depth = depth.copy()
    current_depth[1] = 2
    result, _ = accumulate(buffer, sample(0, depth=current_depth), clamp=0)
    np.testing.assert_array_equal(result[:2], 0)
    assert result[2:].mean() > 0


def test_clamp_limits_large_old_corrections_when_current_detail_disappears():
    buffer = TemporalBuffer()
    buffer.append(sample(10), 3, 1024**2)
    result, _ = accumulate(buffer, sample(0))
    assert result.max() <= 0.0150001


def test_buffer_reprojection_is_cooperatively_cancellable():
    buffer = TemporalBuffer()
    buffer.append(sample(), 3, 1024**2)
    token = CancellationToken()
    token.cancel()
    with pytest.raises(KyvenError):
        accumulate(buffer, sample(), cancellation=token)


def test_reprojection_checks_cancellation_between_row_strips():
    class CancelAfterFirstStrip(CancellationToken):
        checks = 0

        def raise_if_cancelled(self):
            self.checks += 1
            if self.checks == 2:
                self.cancel()
            super().raise_if_cancelled()

    token = CancelAfterFirstStrip()
    with pytest.raises(KyvenError):
        reproject(np.zeros((130, 8, 3), np.float32), np.zeros((130, 8, 2), np.float32), token)
    assert token.checks == 2
