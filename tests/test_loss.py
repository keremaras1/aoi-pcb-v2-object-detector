"""Tests for the custom SSD multibox loss (Section V of the paper)."""

import pytest
import tensorflow as tf

from aoi_pcb_ssd.model.loss import AOILoss

# Encoded target row: [bg, ic, 8 corner offsets, cx, cy].


def _cell(is_foreground: bool, offset: float = 0.0, cx: float = 0.5, cy: float = 0.5) -> list:
    cls = [0.0, 1.0] if is_foreground else [1.0, 0.0]
    return cls + [offset] * 8 + [cx, cy]


def _batch(cells: list[list]) -> tf.Tensor:
    return tf.constant([cells], dtype=tf.float32)  # (1, n_boxes, 12)


class TestLossMagnitude:
    def test_zero_loss_on_perfect_prediction(self) -> None:
        # Identical class one-hots and offsets: both loss terms vanish.
        y = _batch([_cell(True), _cell(False), _cell(False), _cell(False)])
        loss = AOILoss(neg_pos_ratio=3, alpha=3.0).compute_loss(y, y)
        assert float(loss[0]) == pytest.approx(0.0, abs=1e-5)

    def test_misclassified_positive_raises_loss(self) -> None:
        y_true = _batch([_cell(True), _cell(False), _cell(False)])
        y_pred = _batch([_cell(False), _cell(False), _cell(False)])
        loss = AOILoss(alpha=3.0).compute_loss(y_true, y_pred)
        assert float(loss[0]) > 0.0

    def test_localisation_loss_grows_with_corner_error(self) -> None:
        y_true = _batch([_cell(True, 0.0), _cell(False)])
        small = AOILoss(alpha=3.0).compute_loss(y_true, _batch([_cell(True, 0.2), _cell(False)]))
        large = AOILoss(alpha=3.0).compute_loss(y_true, _batch([_cell(True, 0.5), _cell(False)]))
        assert float(large[0]) > float(small[0]) > 0.0


class TestAlphaWeighting:
    def test_alpha_scales_localisation_term(self) -> None:
        # Classes are correct, so only the localisation term (weighted by alpha)
        # contributes. Tripling alpha must triple the loss.
        y_true = _batch([_cell(True, 0.0), _cell(False), _cell(False)])
        y_pred = _batch([_cell(True, 0.2), _cell(False), _cell(False)])
        l1 = float(AOILoss(alpha=1.0).compute_loss(y_true, y_pred)[0])
        l3 = float(AOILoss(alpha=3.0).compute_loss(y_true, y_pred)[0])
        assert l3 == pytest.approx(3.0 * l1)


class TestHardNegativeMining:
    def test_negative_count_capped_by_ratio(self) -> None:
        # One positive (predicted correctly) and nine background cells each
        # mispredicted as foreground with identical loss L. With alpha=0 the
        # total is purely the kept negatives: ratio 3 -> 3L, ratio 6 -> 6L.
        true_cells = [_cell(True)] + [_cell(False)] * 9
        pred_cells = [_cell(True)] + [_cell(True)] * 9
        y_true, y_pred = _batch(true_cells), _batch(pred_cells)
        l3 = float(AOILoss(neg_pos_ratio=3, alpha=0.0).compute_loss(y_true, y_pred)[0])
        l6 = float(AOILoss(neg_pos_ratio=6, alpha=0.0).compute_loss(y_true, y_pred)[0])
        assert l3 > 0.0
        assert l6 == pytest.approx(2.0 * l3)
