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


def test_local_matching_corrects_spatial_cast_without_sampling_removed_subject():
    gradient = np.linspace(100, 150, 240).astype(np.uint8)
    scene = np.broadcast_to(gradient[None, :, None], (180, 240, 3)).copy()
    mask = np.zeros(scene.shape[:2], np.uint8)
    mask[30:150, 65:175] = 255
    generated = scene.copy()
    generated[mask > 0] = (generated[mask > 0].astype(np.int16) + [20, -15, 16]).astype(np.uint8)
    source = scene.copy()
    source[mask > 0] = [220, 10, 15]
    ordinary, _ = match_context(generated, source, mask, background=True)
    fixed, info = match_context(generated, source, mask, background=True,
                               permission_mask=mask, local_radius=64)
    assert info["local_background"]
    before = np.abs(ordinary[mask > 0].astype(float) - scene[mask > 0]).mean()
    after = np.abs(fixed[mask > 0].astype(float) - scene[mask > 0]).mean()
    assert after < before * 0.5
    source[mask > 0] = [5, 220, 30]
    without_old_subject, _ = match_context(generated, source, mask, background=True,
                                         permission_mask=mask, local_radius=64)
    np.testing.assert_array_equal(fixed, without_old_subject)


def test_local_background_step_is_never_used_for_addition_context_matching():
    source = np.full((100, 100, 3), 125, np.uint8)
    mask = np.zeros((100, 100), np.uint8)
    mask[30:70, 30:70] = 255
    generated = source.copy()
    generated[mask > 0] = [20, 50, 210]
    fixed, _ = match_context(generated, source, mask, background=False,
                             permission_mask=mask, local_radius=64)
    np.testing.assert_array_equal(fixed[mask > 0], generated[mask > 0])


def test_paired_tone_curve_corrects_gamma_cast_and_keeps_new_object_color():
    yy, xx = np.indices((256, 256))
    source = np.stack((xx, yy, (xx+yy)//2), axis=-1).astype(np.uint8)
    mask = np.zeros((256, 256), np.uint8)
    mask[90:160, 100:170] = 255
    expected = source.copy()
    expected[mask > 0] = [30, 75, 180]
    generated = np.rint(255*(expected.astype(float)/255)**0.65).astype(np.uint8)
    fixed, info = match_context(generated, source, mask, permission_mask=mask)
    assert info["applied"] and info["method"] == "tone_curve"
    assert np.mean(abs(fixed[mask == 0].astype(float)-source[mask == 0])) < 3
    np.testing.assert_allclose(fixed[120, 130], expected[120, 130], atol=3)
