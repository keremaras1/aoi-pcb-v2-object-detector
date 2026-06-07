"""Tests for training/evaluation metrics, including the singleton-bug regression."""

import pytest
import tensorflow as tf

from aoi_pcb_ssd.model.metrics import (
    class_map,
    f1,
    mae,
    mse,
    precision,
    recall,
    root_mse,
)

from .conftest import _batch, _cell

# Prediction/target row: [bg, ic, 8 corner offsets, cx, cy].


class TestRegressionMetrics:
    def test_mae_zero_on_perfect_offsets(self) -> None:
        # Given: identical positive-cell offsets.
        y = _batch([_cell(True, 0.0)])
        # When/Then: MAE is exactly zero.
        assert float(mae(y, y)) == pytest.approx(0.0, abs=1e-6)

    def test_mse_and_rmse_on_known_error(self) -> None:
        # Given: a positive cell with a known offset error of 0.1 per coordinate.
        y_true = _batch([_cell(True, 0.0)])
        y_pred = _batch([_cell(True, 0.1)])
        # When: MSE and RMSE are computed.
        # Then: MSE = 0.01 and RMSE = 0.1 (eight identical errors averaged).
        assert float(mse(y_true, y_pred)) == pytest.approx(0.01, abs=1e-6)
        assert float(root_mse(y_true, y_pred)) == pytest.approx(0.1, abs=1e-6)

    def test_offsets_ignored_for_background_cells(self) -> None:
        # Given: one positive cell (small offset) and one background cell (large offset).
        y_true = _batch([_cell(True, 0.0), _cell(False, 0.0)])
        y_pred = _batch([_cell(True, 0.1), _cell(False, 0.9)])
        # When: MAE is computed.
        # Then: only the positive cell's offset contributes; background is ignored.
        assert float(mae(y_true, y_pred)) == pytest.approx(0.1, abs=1e-6)


class TestMetricStateResets:
    """Regression test for the module-level metric singleton bug."""

    def test_consecutive_calls_are_independent(self) -> None:
        # Given: two separate evaluation calls on the same ground truth.
        y_true = _batch([_cell(True, 0.0)])
        # When: MAE is computed twice with different predictions.
        first = float(mae(y_true, _batch([_cell(True, 0.05)])))
        second = float(mae(y_true, _batch([_cell(True, 0.1)])))
        # Then: each call yields its own batch's error (no state leakage).
        # A leaked singleton would average both calls to 0.075.
        assert first == pytest.approx(0.05, abs=1e-6)
        assert second == pytest.approx(0.1, abs=1e-6)


class TestClassMAP:
    def test_perfect_classification_scores_one(self) -> None:
        # Given/When: perfect predictions on a batch with one positive.
        y = _batch([_cell(True), _cell(False), _cell(False)])
        # Then: mAP is 1.0.
        assert float(class_map(y, y)) == pytest.approx(1.0)

    def test_all_wrong_classification_scores_zero(self) -> None:
        # Given: one IC and one background.
        y_true = _batch([_cell(True), _cell(False)])
        # When: every prediction is inverted.
        y_pred = _batch([_cell(False), _cell(True)])
        # Then: mAP is 0.0.
        assert float(class_map(y_true, y_pred)) == pytest.approx(0.0)

    def test_handles_never_predicted_class_without_nan(self) -> None:
        # Given: all cells are background; the IC class is never predicted.
        y = _batch([_cell(False)] * 3)
        # When: class_map is computed (IC class has TP=0, FP=0 → NaN internally).
        result = class_map(y, y)
        # Then: NaN is replaced with 0 by the _per_class_ap guard; result is finite.
        assert tf.math.is_finite(result)
        assert float(result) == pytest.approx(0.0)


class TestClassificationMetrics:
    def test_precision_and_recall_perfect(self) -> None:
        # Given/When: perfect predictions.
        y = _batch([_cell(True), _cell(False), _cell(False)])
        # Then: both precision and recall are 1.0.
        assert float(precision(y, y)) == pytest.approx(1.0)
        assert float(recall(y, y)) == pytest.approx(1.0)

    def test_imperfect_precision_one_false_positive(self) -> None:
        # Given: one true IC and one background cell.
        y_true = _batch([_cell(True), _cell(False)])
        # When: the model also predicts the background cell as IC (one FP).
        y_pred = _batch([_cell(True), _cell(True)])
        # Then: precision = TP/(TP+FP) = 1/2 = 0.5; recall = TP/(TP+FN) = 1/1 = 1.0.
        assert float(precision(y_true, y_pred)) == pytest.approx(0.5, abs=1e-6)
        assert float(recall(y_true, y_pred)) == pytest.approx(1.0, abs=1e-6)

    def test_f1_perfect(self) -> None:
        # Given/When: perfect predictions across multiple cells.
        y = _batch([_cell(True), _cell(False), _cell(False)])
        result = f1(y, y)
        # Then: the maximum per-column F1 is 1.0 (the IC column is perfectly scored).
        assert float(tf.reduce_max(result)) == pytest.approx(1.0)

    def test_imperfect_f1_is_between_zero_and_one(self) -> None:
        # Given: a batch of two samples — IC at position 0 in both, but the
        # second sample has its IC mispredicted as background (one FN).
        y_true = tf.constant([[_cell(True), _cell(False), _cell(False)]] * 2, dtype=tf.float32)
        y_pred = tf.constant(
            [
                [_cell(True), _cell(False), _cell(False)],  # correct
                [_cell(False), _cell(False), _cell(False)],  # IC missed
            ],
            dtype=tf.float32,
        )
        # When: F1 is computed.
        result = f1(y_true, y_pred)
        # Then: max per-column F1 is strictly between 0 and 1.
        score = float(tf.reduce_max(result))
        assert 0.0 < score < 1.0
