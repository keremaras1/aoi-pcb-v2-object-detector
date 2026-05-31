"""Tests for the custom SSD multibox loss (Section V of the paper)."""

import pytest
import tensorflow as tf

from aoi_pcb_ssd.model.loss import AOILoss

from .conftest import _batch, _cell

# Encoded target row: [bg, ic, 8 corner offsets, cx, cy].


class TestLossMagnitude:
    def test_zero_loss_on_perfect_prediction(self) -> None:
        # Given: identical class one-hots and offsets — a perfect prediction.
        y = _batch([_cell(True), _cell(False), _cell(False), _cell(False)])
        # When: loss is computed.
        loss = AOILoss(neg_pos_ratio=3, alpha=3.0).compute_loss(y, y)
        # Then: both classification and localisation terms vanish.
        assert float(loss[0]) == pytest.approx(0.0, abs=1e-5)

    def test_misclassified_positive_raises_loss(self) -> None:
        # Given: one positive cell in the ground truth.
        y_true = _batch([_cell(True), _cell(False), _cell(False)])
        # When: the model predicts background for that positive cell.
        y_pred = _batch([_cell(False), _cell(False), _cell(False)])
        loss = AOILoss(alpha=3.0).compute_loss(y_true, y_pred)
        # Then: classification loss is nonzero.
        assert float(loss[0]) > 0.0

    def test_localisation_loss_grows_with_corner_error(self) -> None:
        # Given: one positive cell; classifications are correct throughout.
        y_true = _batch([_cell(True, 0.0), _cell(False)])
        # When: two predictions with different corner errors are evaluated.
        small = AOILoss(alpha=3.0).compute_loss(y_true, _batch([_cell(True, 0.2), _cell(False)]))
        large = AOILoss(alpha=3.0).compute_loss(y_true, _batch([_cell(True, 0.5), _cell(False)]))
        # Then: larger offset error produces strictly higher loss.
        assert float(large[0]) > float(small[0]) > 0.0


class TestAlphaWeighting:
    def test_alpha_scales_localisation_term(self) -> None:
        # Given: correct class predictions and a nonzero corner offset (so only
        # the localisation term, weighted by alpha, contributes to the total).
        y_true = _batch([_cell(True, 0.0), _cell(False), _cell(False)])
        y_pred = _batch([_cell(True, 0.2), _cell(False), _cell(False)])
        # When: alpha is tripled.
        l1 = float(AOILoss(alpha=1.0).compute_loss(y_true, y_pred)[0])
        l3 = float(AOILoss(alpha=3.0).compute_loss(y_true, y_pred)[0])
        # Then: the total loss triples exactly.
        assert l3 == pytest.approx(3.0 * l1)


class TestHardNegativeMining:
    def test_negative_count_capped_by_ratio(self) -> None:
        # Given: one correctly classified positive and nine background cells
        # each mispredicted as foreground (alpha=0 isolates the cls term).
        true_cells = [_cell(True)] + [_cell(False)] * 9
        pred_cells = [_cell(True)] + [_cell(True)] * 9
        y_true, y_pred = _batch(true_cells), _batch(pred_cells)
        # When: two different neg:pos ratios are applied.
        l3 = float(AOILoss(neg_pos_ratio=3, alpha=0.0).compute_loss(y_true, y_pred)[0])
        l6 = float(AOILoss(neg_pos_ratio=6, alpha=0.0).compute_loss(y_true, y_pred)[0])
        # Then: ratio-6 keeps twice as many negatives, doubling the loss.
        assert l3 > 0.0
        assert l6 == pytest.approx(2.0 * l3)


class TestZeroNegatives:
    def test_all_positive_batch_triggers_no_negatives_branch(self) -> None:
        # Given: a batch where every cell is positive (no background cells),
        # so neg_cls_loss_all is identically zero and the _no_negatives branch fires.
        y_true = tf.constant([[_cell(True)] * 4], dtype=tf.float32)
        # When: the predictor labels all cells as background (maximum cls confusion).
        y_pred = tf.constant([[_cell(False)] * 4], dtype=tf.float32)
        loss = AOILoss(alpha=0.0).compute_loss(y_true, y_pred)
        # Then: result is finite and > 0 (positive cls loss still contributes).
        assert tf.math.is_finite(loss[0])
        assert float(loss[0]) > 0.0


class TestNNegMin:
    def test_n_neg_min_enforces_floor_when_no_positives(self) -> None:
        # Given: an all-background batch (zero positives) where every cell is
        # mispredicted as foreground.
        y_bg = tf.constant([[_cell(False)] * 4], dtype=tf.float32)
        y_wrong = tf.constant([[_cell(True)] * 4], dtype=tf.float32)
        # When: n_neg_min=0 vs n_neg_min=2 (alpha=0 isolates the cls term).
        no_floor = float(AOILoss(n_neg_min=0, alpha=0.0).compute_loss(y_bg, y_wrong)[0])
        with_floor = float(AOILoss(n_neg_min=2, alpha=0.0).compute_loss(y_bg, y_wrong)[0])
        # Then: without a floor, zero negatives are kept → loss is zero;
        # with n_neg_min=2, two hard negatives are kept → loss is nonzero.
        assert no_floor == pytest.approx(0.0, abs=1e-5)
        assert with_floor > 0.0


class TestBatchSize:
    def test_returns_per_sample_loss_vector(self) -> None:
        # Given: a batch of two samples — first is perfect, second has a corner error.
        y_true = tf.constant(
            [[_cell(True, 0.0), _cell(False)], [_cell(True, 0.0), _cell(False)]],
            dtype=tf.float32,
        )
        y_pred = tf.constant(
            [[_cell(True, 0.0), _cell(False)], [_cell(True, 0.5), _cell(False)]],
            dtype=tf.float32,
        )
        # When: loss is computed across the full batch.
        loss = AOILoss(alpha=3.0).compute_loss(y_true, y_pred)
        # Then: a per-sample vector of length 2 is returned; the perfect sample
        # has zero loss and the imperfect sample has nonzero loss.
        assert tuple(loss.shape) == (2,)
        assert float(loss[0]) == pytest.approx(0.0, abs=1e-5)
        assert float(loss[1]) > 0.0


class TestZeroPositiveBatch:
    def test_all_background_batch_loss_is_finite(self) -> None:
        # Given: an all-background batch (n_positives=0) with mispredicted cells.
        # n_neg_min=2 forces hard-negative retention to exercise the max(1, N_pos)
        # normaliser path (max(1, 0) = 1, preventing division by zero).
        y_true = tf.constant([[_cell(False)] * 4], dtype=tf.float32)
        y_pred = tf.constant([[_cell(True)] * 4], dtype=tf.float32)
        # When: loss is computed.
        loss = AOILoss(n_neg_min=2, alpha=0.0).compute_loss(y_true, y_pred)
        # Then: result is finite and nonzero (two negatives were kept despite 0 positives).
        assert tf.math.is_finite(loss[0])
        assert float(loss[0]) > 0.0
