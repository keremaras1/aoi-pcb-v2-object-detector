"""Tests for SSD label encoding, prediction decoding, and anchor matching."""

import numpy as np

from aoi_pcb_ssd.encoding import (
    SSDInputEncoder,
    decode_detections,
    match_by_nearest_centre,
)

_IMG = 256
_PREDICTOR_SIZES = [(8, 8)]
_N_BOXES = 64  # 8 × 8 grid


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
        # Given: four anchors at the corners of a unit square.
        anchors = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        # When: a ground-truth close to anchor 0 is matched.
        matches = match_by_nearest_centre(np.array([[0.1, 0.1]]), anchors)
        # Then: the nearest anchor (0) is selected.
        assert matches == [0]

    def test_collision_falls_back_to_next_nearest(self) -> None:
        # Given: four anchors and two ground truths both closest to anchor 0.
        anchors = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        # When: greedy matching resolves the conflict.
        matches = match_by_nearest_centre(np.array([[0.1, 0.1], [0.0, 0.0]]), anchors)
        # Then: no two ground truths share the same anchor.
        assert matches[0] == 0
        assert matches[1] != 0
        assert len(set(matches)) == 2

    def test_assertion_ensures_one_match_per_ground_truth(self) -> None:
        # Given: 64 anchors (full 8×8 grid) and 5 random ground truths.
        rng = np.random.default_rng(0)
        anchors = rng.random((64, 2))
        gt = rng.random((5, 2))
        # When: matching runs to completion.
        matches = match_by_nearest_centre(gt, anchors)
        # Then: exactly 5 matches are returned — the post-loop assertion in
        # matching.py passes, guaranteeing one match per ground truth.
        assert len(matches) == 5


class TestEncoder:
    def test_output_shape(self) -> None:
        # Given/When: one IC label encoded by the default encoder.
        y = _encoder()([_one_ic_label()])
        # Then: shape is (batch=1, n_boxes=64, n_classes+1+8+2=12).
        assert y.shape == (1, _N_BOXES, 12)

    def test_background_cells_marked_by_default(self) -> None:
        # Given/When: one IC in a 64-cell grid.
        y = _encoder()([_one_ic_label()])
        # Then: exactly one cell is positive (background score 0).
        positive = y[0, :, 0] == 0
        assert positive.sum() == 1

    def test_positive_cell_is_one_hot_foreground(self) -> None:
        # Given/When: one IC encoded.
        y = _encoder()([_one_ic_label()])
        # When: the positive cell is located.
        cell = np.argmax(y[0, :, 0] == 0)
        # Then: its class scores are [bg=0, ic=1].
        assert list(y[0, cell, :2]) == [0.0, 1.0]

    def test_empty_labels_leave_all_background(self) -> None:
        # Given/When: an image with no ICs.
        y = _encoder()([np.empty((0, 11))])
        # Then: all 64 cells are background.
        assert np.all(y[0, :, 0] == 1.0)

    def test_unnormalized_anchor_centres_are_pixel_space(self) -> None:
        # Given: an encoder with normalize_coords=False.
        enc = _encoder(normalize_coords=False)
        # When: one IC is encoded.
        y = enc([_one_ic_label()])
        anchor_coords = y[0, :, -2:]
        # Then: anchor centres are in pixel space — values exceed [0,1] but
        # stay within the image dimensions.
        assert float(anchor_coords.max()) > 1.0
        assert float(anchor_coords.max()) <= _IMG


class TestDecoder:
    def test_roundtrip_recovers_corner_pixels(self) -> None:
        # Given: normalised encoder output for one IC.
        label = _one_ic_label()
        y = _encoder()([label])
        # When: decoded with coordinate denormalisation.
        detections = decode_detections(y, normalize_coords=True, img_height=_IMG, img_width=_IMG)
        assert len(detections) == 1
        det = detections[0]
        # Then: decoded corners match the original pixel labels within float tolerance.
        assert det.shape == (1, 10)
        assert det[0, 0] == 1  # foreground class id
        np.testing.assert_allclose(det[0, 2:], label[0, 1:9], atol=1e-3)

    def test_decoder_without_normalisation_recovers_pixel_corners(self) -> None:
        # Given: pixel-space encoder output (normalize_coords=False throughout).
        label = _one_ic_label()
        enc = _encoder(normalize_coords=False)
        y = enc([label])
        # When: decoded without the img_height/img_width scale-up step
        # (the else-branch of ``if normalize_coords`` in decode_detections).
        detections = decode_detections(y, normalize_coords=False)
        det = detections[0]
        # Then: corners are recovered without any coordinate transformation.
        assert det[0, 0] == 1
        np.testing.assert_allclose(det[0, 2:], label[0, 1:9], atol=1.0)

    def test_background_only_yields_no_detections(self) -> None:
        # Given: a prediction tensor where every cell scores background.
        y_pred = np.zeros((1, 4, 12))
        y_pred[:, :, 0] = 1.0
        # When: decoded.
        detections = decode_detections(
            y_pred, normalize_coords=True, img_height=_IMG, img_width=_IMG
        )
        # Then: no detections are returned.
        assert detections[0].size == 0

    def test_empty_label_decodes_to_no_detections(self) -> None:
        # Given: an image with no ICs → all-background encoded output.
        y = _encoder()([np.empty((0, 11))])
        # When: decoded.
        detections = decode_detections(y, normalize_coords=True, img_height=_IMG, img_width=_IMG)
        # Then: no detections.
        assert detections[0].size == 0
