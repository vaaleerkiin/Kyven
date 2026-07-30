from __future__ import annotations

import unittest

import numpy as np

from kyven.errors import KyvenError
from kyven.segment.models import BoxPrompt, PointLabel, PointPrompt
from kyven.segment.roi import expand_mask, resolve_region, translate_points


class ProcessingRoiTests(unittest.TestCase):
    def test_crop_coordinates_translate_and_mask_expands_to_full_frame(self) -> None:
        region = resolve_region(BoxPrompt(2.2, 1.1, 6.2, 4.8), width=8, height=6)
        points = translate_points((PointPrompt(4, 3),), region)
        expanded = expand_mask(np.ones((4, 5), dtype=np.bool_), region)

        self.assertEqual((region.x0, region.y0, region.x1, region.y1), (2, 1, 7, 5))
        self.assertEqual((points[0].x, points[0].y), (2, 2))
        self.assertEqual(expanded.shape, (6, 8))
        self.assertFalse(expanded[0, 0])
        self.assertTrue(expanded[1, 2])

    def test_positive_point_outside_roi_is_rejected(self) -> None:
        region = resolve_region(BoxPrompt(2, 2, 6, 6), width=8, height=8)

        with self.assertRaises(KyvenError):
            translate_points((PointPrompt(1, 1, PointLabel.POSITIVE),), region)

    def test_negative_point_outside_roi_is_ignored(self) -> None:
        region = resolve_region(BoxPrompt(2, 2, 6, 6), width=8, height=8)

        points = translate_points(
            (
                PointPrompt(3, 3, PointLabel.POSITIVE),
                PointPrompt(1, 1, PointLabel.NEGATIVE),
            ),
            region,
        )

        self.assertEqual(len(points), 1)
        self.assertEqual((points[0].x, points[0].y), (1, 1))
