from __future__ import annotations

import numpy as np

from kyven.dlss.color import prepare, preserve_source_color, restore_residual


def test_rolloff_bounds_hdr_and_negative_rgb_and_preserves_positive_chroma():
    rgb = np.array([[[0.18, 0.09, 0.045], [16, 8, 4], [1e6, 2e5, 1e5],
                     [-4, 8, 2], [-10, -2, -6]]], dtype=np.float32)
    working, inverse = prepare(rgb)
    assert np.isfinite(working).all()
    assert 0 <= working.min() <= working.max() <= 1
    np.testing.assert_allclose(working[0, :3, 1] / working[0, :3, 0],
                               rgb[0, :3, 1] / rgb[0, :3, 0], rtol=1e-3)
    assert inverse[0, 0] == 1
    assert inverse[0, 1] > 1
    np.testing.assert_array_equal(restore_residual(working, working, inverse), 0)


def test_rolloff_is_independent_of_other_pixels_and_preceding_frames():
    original = np.array([[[8, 4, 2], [-0.2, 0.1, 2]]], dtype=np.float32)
    before, gain = prepare(original)
    mixed = np.concatenate((original, np.full_like(original, 1e8)), axis=1)
    after, new_gain = prepare(mixed)
    np.testing.assert_array_equal(before, after[:, :2])
    np.testing.assert_array_equal(gain, new_gain[:, :2])


def test_inverse_restores_scene_linear_change_instead_of_leaving_rolled_rgb():
    source = np.array([[[8, 4, -0.5]]], dtype=np.float32)
    working, inverse = prepare(source)
    desired_delta = np.array([[[0.4, -0.2, 0.1]]], dtype=np.float32)
    predicted = working + desired_delta / inverse[..., None]
    result = source + restore_residual(predicted, working, inverse)
    np.testing.assert_allclose(result, source + desired_delta, atol=2e-6)


def test_knee_changes_highlights_but_does_not_change_low_positive_rgb():
    source = np.array([[[0.1, 0.05, 0], [8, 4, 2]]], dtype=np.float32)
    a, _ = prepare(source, knee=0.5)
    b, _ = prepare(source, knee=0.8)
    np.testing.assert_array_equal(a[:, :1], b[:, :1])
    assert not np.array_equal(a[:, 1:], b[:, 1:])


def test_rolloff_handles_full_float32_range_without_invalid_model_input():
    maximum = np.finfo(np.float32).max
    source = np.array([[[maximum, -maximum, 0], [-maximum, -maximum, -maximum]]],
                      dtype=np.float32)
    working, inverse = prepare(source)
    assert np.isfinite(working).all()
    assert np.isfinite(inverse).all()
    assert working.min() >= 0 and working.max() <= 1
    np.testing.assert_array_equal(restore_residual(working, working, inverse), 0)


def test_color_preservation_removes_magenta_and_retains_brightness_detail():
    source = np.array([[[0.18, 0.18, 0.18], [0.04, 0.08, 0.6], [-0.2, 2, 8], [0, 0, 0]]],
                      dtype=np.float32)
    generated = np.array([[[0.4, 0.1, 0.4], [0.12, 0.04, 0.7], [2, 1, 12], [0.1, 0, 0.1]]],
                         dtype=np.float32)
    preserve_source_color(source, generated)
    np.testing.assert_array_equal(generated[0, 0, :], np.repeat(generated[0, 0, 0], 3))
    gain = generated[0, :3, 2] / source[0, :3, 2]
    np.testing.assert_allclose(generated[0, :3], source[0, :3] * gain[:, None], rtol=1e-6)
    assert not np.array_equal(generated, source)
    assert generated[0, 2, 0] < 0
    np.testing.assert_array_equal(generated[0, 3], 0)


def test_color_preservation_is_exact_for_identity_hdr_negatives_and_subnormals():
    maximum = np.finfo(np.float32).max
    tiny = np.nextafter(np.float32(0), np.float32(1))
    source = np.array([[[maximum, -maximum, maximum / 2], [tiny, tiny * 2, 0],
                        [0, 0, 0], [-3, -2, -1]]], dtype=np.float32)
    generated = source.copy()
    preserve_source_color(source, generated)
    np.testing.assert_array_equal(generated, source)
