"""Source-anchored correction confined to an inward boundary band."""

from __future__ import annotations

import numpy as np


def repair_boundary(generated, source, mask, width=24):
    import cv2
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import splu

    selected = np.asarray(mask) > 0
    result = np.asarray(generated).copy()
    distance = cv2.distanceTransform(selected.astype(np.uint8), cv2.DIST_L2, 5)
    band = selected & (distance <= width)
    if not band.any() or selected.all():
        result[~selected] = source[~selected]
        return result, {"applied": False}
    index = np.full(selected.shape, -1, dtype=np.int32)
    index[band] = np.arange(int(band.sum()))
    delta = np.asarray(source, dtype=np.float64) - result.astype(np.float64)
    outer = selected & ~cv2.erode(selected.astype(np.uint8), np.ones((3, 3), np.uint8),
                                 borderType=cv2.BORDER_CONSTANT, borderValue=1).astype(bool)
    clean = cv2.inpaint(np.asarray(source), selected.astype(np.uint8) * 255, 3,
                       cv2.INPAINT_NS).astype(np.float64)
    fixed = outer[band]
    rhs = np.zeros((int(band.sum()), 3))
    diagonal = np.full(len(rhs), 0.015)
    rows, cols, values = [], [], []
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        ys, xs = np.nonzero(band)
        yy, xx = ys + dy, xs + dx
        valid = (yy >= 0) & (yy < band.shape[0]) & (xx >= 0) & (xx < band.shape[1])
        valid &= ~outer[ys, xs]
        ys, xs, yy, xx = ys[valid], xs[valid], yy[valid], xx[valid]
        ids, neighbors = index[ys, xs], index[yy, xx]
        diagonal[ids] += 1
        inner = neighbors >= 0
        rows.extend(ids[inner])
        cols.extend(neighbors[inner])
        values.extend(-np.ones(inner.sum()))
        anchor = ~selected[yy, xx]
        rhs[ids[anchor]] += delta[yy[anchor], xx[anchor]]
    diagonal[fixed] = 1
    rhs[fixed] = (clean - result.astype(np.float64))[outer]
    rows.extend(range(len(rhs)))
    cols.extend(range(len(rhs)))
    values.extend(diagonal)
    matrix = coo_matrix((values, (rows, cols)), shape=(len(rhs), len(rhs))).tocsc()
    correction = splu(matrix, permc_spec="MMD_AT_PLUS_A").solve(rhs)
    result[band] = np.clip(np.rint(result[band].astype(float) + correction), 0, 255)
    result[~selected] = source[~selected]
    return result, {
        "applied": True,
        "method": "screened-boundary-band",
        "band_pixels": int(band.sum()),
        "width": width,
        "interior_preserved": True,
    }
