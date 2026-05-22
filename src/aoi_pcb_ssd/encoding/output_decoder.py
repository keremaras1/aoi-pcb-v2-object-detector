# Portions of this file are derived from
# https://github.com/pierluigiferrari/ssd_keras
# Copyright 2018 Pierluigi Ferrari, licensed under the Apache License, Version 2.0.
# Modifications Copyright 2024 Kerem Aras, also licensed under the Apache License, Version 2.0.
"""SSD output decoder: converts raw model predictions to absolute corner coordinates."""

import numpy as np
from numpy.typing import NDArray


def decode_detections(
    y_pred: NDArray,
    normalize_coords: bool = True,
    img_height: int | None = None,
    img_width: int | None = None,
) -> list[NDArray]:
    """Decode raw SSD output into absolute IC corner coordinates.

    The model outputs a ``(batch, 64, 12)`` tensor where each row contains:
    ``[bg_score, ic_score, tl_x_off, tl_y_off, tr_x_off, tr_y_off,
       bl_x_off, bl_y_off, br_x_off, br_y_off, cx, cy]``

    This function:
    1. Reads the predicted class ID and confidence from the softmax scores.
    2. Reconstructs absolute corner coordinates by adding cell-centre offsets
       back to the predicted corner offsets (reversing the encoding step).
    3. Optionally denormalises coordinates from [0, 1] to pixel space.
    4. Filters out background predictions (class_id == 0).

    Args:
        y_pred: Raw model output of shape ``(batch, n_boxes, 12)``.
        normalize_coords: If ``True``, multiply coordinates by image dimensions.
            Requires ``img_height`` and ``img_width`` to be set.
        img_height: Image height in pixels. Required when ``normalize_coords=True``.
        img_width: Image width in pixels. Required when ``normalize_coords=True``.

    Returns:
        List of ``batch_size`` arrays. Each array has shape ``(n_detections, 10)``
        with columns ``[class_id, confidence, tl_x, tl_y, tr_x, tr_y,
        bl_x, bl_y, br_x, br_y]``. Rows with ``class_id == 0`` (background)
        are removed. An image with no detections returns an empty array.
    """
    # Allocate output: [class_id, confidence, tl_x, tl_y, tr_x, tr_y, bl_x, bl_y, br_x, br_y]
    decoded = np.copy(y_pred[:, :, -12:-2])

    decoded[:, :, 0] = np.argmax(y_pred[:, :, :-10], axis=-1)   # class id
    decoded[:, :, 1] = np.amax(y_pred[:, :, :-10], axis=-1)     # confidence

    # Add cell-centre back to corner offsets to recover absolute coordinates
    decoded[:, :, [2, 4, 6, 8]] = (
        y_pred[:, :, [-10, -8, -6, -4]] + np.expand_dims(y_pred[:, :, -2], axis=-1)
    )
    decoded[:, :, [3, 5, 7, 9]] = (
        y_pred[:, :, [-9, -7, -5, -3]] + np.expand_dims(y_pred[:, :, -1], axis=-1)
    )

    if normalize_coords:
        decoded[:, :, [2, 4, 6, 8]] *= img_width
        decoded[:, :, [3, 5, 7, 9]] *= img_height

    return [
        batch_item[np.nonzero(batch_item[:, 0])]
        for batch_item in decoded
    ]
