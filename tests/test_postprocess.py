from __future__ import annotations

import unittest

import numpy as np

from kyven.segment.postprocess import fill_enclosed_holes


class MaskPostprocessTests(unittest.TestCase):
    def test_small_enclosed_hole_is_filled_without_changing_outer_edge(self) -> None:
        mask = np.zeros((10, 10), dtype=np.bool_)
        mask[1:9, 1:9] = True
        mask[4:6, 4:6] = False

        result = fill_enclosed_holes(mask, max_area=10)

        self.assertEqual(result.filled_holes, 1)
        self.assertEqual(result.filled_pixels, 4)
        self.assertTrue(result.mask[4:6, 4:6].all())
        self.assertFalse(result.mask[0, 0])

    def test_hole_larger_than_limit_is_preserved(self) -> None:
        mask = np.ones((10, 10), dtype=np.bool_)
        mask[3:7, 3:7] = False

        result = fill_enclosed_holes(mask, max_area=8)

        self.assertEqual(result.filled_holes, 0)
        self.assertFalse(result.mask[3:7, 3:7].any())

    def test_background_connected_to_frame_border_is_not_filled(self) -> None:
        mask = np.ones((10, 10), dtype=np.bool_)
        mask[0:6, 5] = False

        result = fill_enclosed_holes(mask, max_area=0)

        self.assertEqual(result.filled_holes, 0)
        self.assertFalse(result.mask[0:6, 5].any())
