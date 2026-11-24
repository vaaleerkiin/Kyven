"""Color regressions: exposure drift, removed-subject leakage and new colors."""
import numpy as np

from kyven.generative_fill.color_match import match_context


def test_removal_matches_flat_background_without_sampling_old_subject():
    scene = np.empty((160, 160, 3), np.uint8)
    scene[:60] = [80, 85, 90]
    scene[60:110] = [140, 145, 150]
    scene[110:] = [195, 190, 175]
    mask = np.zeros((160, 160), np.uint8)
    mask[10:150, 55:105] = 255
    original = scene.copy()
    original[mask > 0] = [220, 20, 30]  # must never color the reconstructed background
    generated = scene.astype(np.int16) + 6
    generated[mask > 0] += 14
    fixed, info = match_context(generated.astype(np.uint8), original, mask, background=True)
    assert info["applied"] and info["background"]
    for row in (30, 85, 130):
        np.testing.assert_allclose(fixed[row, 80], scene[row, 80], atol=2)


def test_context_calibration_preserves_new_blue_object_in_an_addition():
    yy, xx = np.indices((120, 160))
    original = np.stack([60 + xx // 2, 80 + yy // 2, 70 + xx // 3], -1).astype(np.uint8)
    expected = original.copy()
    expected[35:85, 60:100] = [35, 75, 180]
    mask = np.zeros(original.shape[:2], np.uint8)
    mask[30:90, 55:105] = 255
    generated = np.rint(expected * 1.04 + 7).astype(np.uint8)
    fixed, info = match_context(generated, original, mask, background=False)
    assert info["applied"] and not info["background"]
    np.testing.assert_allclose(fixed[50, 80], expected[50, 80], atol=2)


def test_no_clean_context_does_not_use_the_masked_subject_as_a_reference():
    generated = np.full((32, 32, 3), 180, np.uint8)
    original = np.full_like(generated, 20)
    fixed, info = match_context(generated, original, np.full((32, 32), 255, np.uint8),
                               background=True)
    np.testing.assert_array_equal(fixed, generated)
    assert not info["applied"]
