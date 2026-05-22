"""Anchor-to-ground-truth matching strategy.

This module implements nearest-centre matching: each ground-truth IC is
assigned to the grid cell whose centre is closest in Euclidean distance.
This differs from the standard SSD IoU-based matching and is the strategy
described in Section IV.A of the paper.
"""

import numpy as np
from numpy.typing import NDArray


def match_by_nearest_centre(
    gt_centers: NDArray[np.float64],
    anchor_centers: NDArray[np.float64],
) -> list[int]:
    """Match each ground-truth IC to its nearest grid-cell anchor.

    Each ground-truth centre is greedily matched to the closest unoccupied
    anchor. If the nearest anchor is already taken, the next-closest is used.

    Args:
        gt_centers: Array of shape ``(N, 2)`` containing the ``(cx, cy)``
            coordinates of ``N`` ground-truth ICs.
        anchor_centers: Array of shape ``(M, 2)`` containing the ``(cx, cy)``
            coordinates of all ``M`` grid-cell anchors.

    Returns:
        List of length ``N`` where entry ``i`` is the index into
        ``anchor_centers`` matched to ``gt_centers[i]``.

    Raises:
        AssertionError: If the number of returned matches differs from ``N``.
    """
    matches: list[int] = []
    for gt_center in gt_centers:
        distances = np.sqrt(np.sum(np.abs(anchor_centers - gt_center) ** 2, axis=-1))
        candidate = int(np.argmin(distances))
        while candidate in matches:
            distances[candidate] = np.inf
            candidate = int(np.argmin(distances))
        matches.append(candidate)

    assert len(matches) == gt_centers.shape[0], (
        f"Match count mismatch: expected {gt_centers.shape[0]}, got {len(matches)}"
    )
    return matches
