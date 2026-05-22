# Portions of this file are derived from
# https://github.com/pierluigiferrari/ssd_keras
# Copyright 2018 Pierluigi Ferrari, licensed under the Apache License, Version 2.0.
# Modifications Copyright 2024 Kerem Aras, also licensed under the Apache License, Version 2.0.
"""SSD input encoder for four-corner IC keypoint detection.

Encodes a batch of ground-truth IC annotations into the fixed-size target
tensor consumed by ``model.fit()``. Each of the 64 grid cells is assigned
either an IC (positive) or background. Positive cells store normalised
corner-point offsets relative to the cell centre; background cells store zeros.

This differs from the upstream SSD encoder in two ways:
  - Matching uses nearest-centre distance instead of IoU (see :mod:`matching`).
  - Regression targets are four corner points (8 values) instead of
    centre+width+height (4 values).
"""

import numpy as np
from numpy.typing import NDArray

from aoi_pcb_ssd.encoding.matching import match_by_nearest_centre

# Column indices of the ground-truth label array produced by DataGenerator.
# Shape per image: (n_ics, 11) — [class_id, tl_x, tl_y, tr_x, tr_y, bl_x, bl_y, br_x, br_y, cx, cy]
_CLASS_ID = 0
_TL_X, _TL_Y = 1, 2
_TR_X, _TR_Y = 3, 4
_BL_X, _BL_Y = 5, 6
_BR_X, _BR_Y = 7, 8
_CX, _CY = 9, 10


class SSDInputEncoder:
    """Encode ground-truth IC labels into the SSD target tensor.

    For each image the encoder:
    1. Normalises absolute pixel coordinates to [0, 1].
    2. Matches each IC to its nearest grid-cell anchor
       (see :func:`~aoi_pcb_ssd.encoding.matching.match_by_nearest_centre`).
    3. Writes the IC's one-hot class vector and eight corner coordinates into
       the matched cell's row of the target tensor.
    4. Converts absolute corner coordinates to offsets relative to the matched
       cell's centre (positive cells only; background cells keep zeros).

    The returned tensor has shape ``(batch, n_boxes, n_classes + 8 + 2)``
    where the last two columns hold the anchor centre coordinates used by the
    loss and decoder.

    Args:
        img_height: Input image height in pixels.
        img_width: Input image width in pixels.
        n_classes: Number of foreground classes (background is added internally).
        predictor_sizes: List of ``(height, width)`` tuples, one per feature
            map level. For the paper's architecture this is ``[(8, 8)]``.
        normalize_coords: Whether to normalise coordinates to [0, 1].
        background_id: Class index reserved for background (default 0).
    """

    def __init__(
        self,
        img_height: int,
        img_width: int,
        n_classes: int,
        predictor_sizes: list[tuple[int, int]],
        normalize_coords: bool = True,
        background_id: int = 0,
    ) -> None:
        self.img_height = img_height
        self.img_width = img_width
        self.n_classes = n_classes + 1
        self.normalize_coords = normalize_coords
        self.background_id = background_id
        self.predictor_sizes = np.array(predictor_sizes)

        self.boxes_list: list[NDArray] = []
        self.steps_diag: list[tuple[float, float]] = []
        self.offsets_diag: list[tuple[float, float]] = []

        for size in self.predictor_sizes:
            boxes, _, step, offset = self._generate_anchor_grid(size)
            self.boxes_list.append(boxes)
            self.steps_diag.append(step)
            self.offsets_diag.append(offset)

    def __call__(self, ground_truth_labels: list[NDArray]) -> NDArray:
        """Encode a batch of ground-truth label arrays.

        Args:
            ground_truth_labels: List of ``batch_size`` arrays, each of shape
                ``(n_ics, 11)`` with columns
                ``[class_id, tl_x, tl_y, tr_x, tr_y, bl_x, bl_y, br_x, br_y, cx, cy]``.

        Returns:
            Encoded target tensor of shape ``(batch_size, n_boxes, n_classes + 10)``.
        """
        batch_size = len(ground_truth_labels)
        y_encoded = self._make_encoding_template(batch_size)
        y_encoded[:, :, self.background_id] = 1.0
        class_vectors = np.eye(self.n_classes)

        for i in range(batch_size):
            if ground_truth_labels[i].size == 0:
                continue

            labels = ground_truth_labels[i].astype(float)

            if self.normalize_coords:
                labels[:, [_TL_Y, _TR_Y, _BL_Y, _BR_Y, _CY]] /= self.img_height
                labels[:, [_TL_X, _TR_X, _BL_X, _BR_X, _CX]] /= self.img_width

            classes_one_hot = class_vectors[labels[:, _CLASS_ID].astype(int)]
            labels_one_hot = np.concatenate(
                [classes_one_hot,
                 labels[:, [_TL_X, _TL_Y, _TR_X, _TR_Y, _BL_X, _BL_Y, _BR_X, _BR_Y]]],
                axis=-1,
            )

            matches = match_by_nearest_centre(
                labels[:, [_CX, _CY]], y_encoded[i, :, -2:]
            )
            y_encoded[i, matches, :-2] = labels_one_hot

        # Convert absolute corner coords to cell-centre-relative offsets for
        # positive cells only; background cells keep their zero corner values.
        is_positive = (y_encoded[:, :, self.background_id] == 0)
        cx_grid = y_encoded[:, :, -2]
        cy_grid = y_encoded[:, :, -1]

        x_cols = y_encoded[:, :, [-10, -8, -6, -4]]
        y_cols = y_encoded[:, :, [-9, -7, -5, -3]]

        y_encoded[:, :, [-10, -8, -6, -4]] = np.where(
            is_positive[:, :, np.newaxis],
            x_cols - cx_grid[:, :, np.newaxis],
            x_cols,
        )
        y_encoded[:, :, [-9, -7, -5, -3]] = np.where(
            is_positive[:, :, np.newaxis],
            y_cols - cy_grid[:, :, np.newaxis],
            y_cols,
        )

        return y_encoded

    def _generate_anchor_grid(
        self, feature_map_size: NDArray
    ) -> tuple[NDArray, tuple, tuple[float, float], tuple[float, float]]:
        """Compute (cx, cy) grid-centre coordinates for one feature map level."""
        step_h = self.img_height / feature_map_size[0]
        step_w = self.img_width / feature_map_size[1]
        offset_h, offset_w = 0.5, 0.5

        cy_vals = np.linspace(
            step_h * offset_h,
            (offset_h + feature_map_size[0] - 1) * step_h,
            feature_map_size[0],
        )
        cx_vals = np.linspace(
            step_w * offset_w,
            (offset_w + feature_map_size[1] - 1) * step_w,
            feature_map_size[1],
        )

        cx_grid, cy_grid = np.meshgrid(cx_vals, cy_vals)
        boxes = np.zeros((feature_map_size[0], feature_map_size[1], 1, 2))
        boxes[:, :, 0, 0] = cx_grid
        boxes[:, :, 0, 1] = cy_grid

        if self.normalize_coords:
            boxes[:, :, :, 0] /= self.img_width
            boxes[:, :, :, 1] /= self.img_height
            step_h /= self.img_height
            step_w /= self.img_width

        return boxes, (cy_vals, cx_vals), (step_h, step_w), (offset_h, offset_w)

    def _make_encoding_template(self, batch_size: int) -> NDArray:
        """Build the zero-filled target tensor for one batch."""
        boxes_batch = []
        for boxes in self.boxes_list:
            tiled = np.tile(np.expand_dims(boxes, axis=0), (batch_size, 1, 1, 1, 1))
            boxes_batch.append(np.reshape(tiled, (batch_size, -1, 2)))

        anchors = np.concatenate(boxes_batch, axis=1)
        n_boxes = anchors.shape[1]
        classes = np.zeros((batch_size, n_boxes, self.n_classes))
        corners = np.zeros((batch_size, n_boxes, 8))
        return np.concatenate((classes, corners, anchors), axis=2)
