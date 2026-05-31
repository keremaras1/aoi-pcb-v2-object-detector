"""Tests for SSD label encoding, prediction decoding, and anchor matching."""

import numpy as np

from aoi_pcb_ssd.encoding import (
    SSDInputEncoder,
    decode_detections,
    match_by_nearest_centre,
)

_IMG = 256
_PREDICTOR_SIZES = [(8, 8)]
_N_BOXES = 64  # 8 x 8 grid


def _encoder(normalize_coords: bool = True) -> SSDInputEncoder:
    return SSDInputEncoder(
        img_height=_IMG,
        img_width=_IMG,
        n_classes=1,
        predictor_sizes=_PREDICTOR_SIZES,
        normalize_coords=normalize_coords,
    )


def _one_ic_label() -> np.ndarray:
    # [class_id, tl_x, tl_y, tr_x, tr_y, bl_x, bl_y, br_x, br_y, cx, cy]
    return np.array([[1, 40, 50, 80, 50, 40, 90, 80, 90, 60, 70]], dtype=float)


class TestMatching:
    def test_matches_nearest_anchor(self) -> None:
        anchors = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        matches = match_by_nearest_centre(np.array([[0.1, 0.1]]), anchors)
        assert matches == [0]

    def test_collision_falls_back_to_next_nearest(self) -> None:
        anchors = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        # Both ground truths are closest to anchor 0; the second must shift.
        matches = match_by_nearest_centre(np.array([[0.1, 0.1], [0.0, 0.0]]), anchors)
        assert matches[0] == 0
        assert matches[1] != 0
        assert len(set(matches)) == 2

    def test_one_match_per_ground_truth(self) -> None:
        anchors = np.random.rand(64, 2)
        gt = np.random.rand(5, 2)
        assert len(match_by_nearest_centre(gt, anchors)) == 5


class TestEncoder:
    def test_output_shape(self) -> None:
        y = _encoder()([_one_ic_label()])
        # (batch, n_boxes, n_classes + 1 background + 8 corners + 2 anchor centre)
        assert y.shape == (1, _N_BOXES, 12)

    def test_background_cells_marked_by_default(self) -> None:
        y = _encoder()([_one_ic_label()])
        # Exactly one positive cell (background score 0); the rest are background.
        positive = y[0, :, 0] == 0
        assert positive.sum() == 1

    def test_positive_cell_is_one_hot_foreground(self) -> None:
        y = _encoder()([_one_ic_label()])
        cell = np.argmax(y[0, :, 0] == 0)
        assert list(y[0, cell, :2]) == [0.0, 1.0]

    def test_empty_labels_leave_all_background(self) -> None:
        y = _encoder()([np.empty((0, 11))])
        assert np.all(y[0, :, 0] == 1.0)

    def test_unnormalized_encoding_shape(self) -> None:
        y = _encoder(normalize_coords=False)([_one_ic_label()])
        assert y.shape == (1, _N_BOXES, 12)


class TestDecoder:
    def test_roundtrip_recovers_corner_pixels(self) -> None:
        label = _one_ic_label()
        y = _encoder()([label])
        detections = decode_detections(y, normalize_coords=True, img_height=_IMG, img_width=_IMG)
        assert len(detections) == 1
        det = detections[0]
        assert det.shape == (1, 10)
        assert det[0, 0] == 1  # foreground class id
        # Columns 2..10 are the eight corner coordinates, recovered exactly.
        np.testing.assert_allclose(det[0, 2:], label[0, 1:9], atol=1e-3)

    def test_background_only_yields_no_detections(self) -> None:
        y_pred = np.zeros((1, 4, 12))
        y_pred[:, :, 0] = 1.0  # all cells score background
        detections = decode_detections(
            y_pred, normalize_coords=True, img_height=_IMG, img_width=_IMG
        )
        assert detections[0].size == 0

    def test_empty_label_decodes_to_no_detections(self) -> None:
        y = _encoder()([np.empty((0, 11))])
        detections = decode_detections(y, normalize_coords=True, img_height=_IMG, img_width=_IMG)
        assert detections[0].size == 0
