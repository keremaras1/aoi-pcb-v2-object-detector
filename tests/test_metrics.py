"""Tests for training/evaluation metrics, including the singleton-bug regression."""

import pytest
import tensorflow as tf

from aoi_pcb_ssd.model.metrics import (
    class_mAP,
    f1,
    mae,
    mse,
    precision,
    recall,
    root_mse,
)

# Prediction/target row: [bg, ic, 8 corner offsets, cx, cy].


def _cell(is_foreground: bool, offset: float = 0.0, cx: float = 0.5, cy: float = 0.5) -> list:
    cls = [0.0, 1.0] if is_foreground else [1.0, 0.0]
    return cls + [offset] * 8 + [cx, cy]


def _batch(cells: list[list]) -> tf.Tensor:
    return tf.constant([cells], dtype=tf.float32)  # (1, n_boxes, 12)


class TestRegressionMetrics:
    def test_mae_zero_on_perfect_offsets(self) -> None:
        y = _batch([_cell(True, 0.0)])
        assert float(mae(y, y)) == pytest.approx(0.0, abs=1e-6)

    def test_mse_and_rmse_on_known_error(self) -> None:
        y_true = _batch([_cell(True, 0.0)])
        y_pred = _batch([_cell(True, 0.1)])
        assert float(mse(y_true, y_pred)) == pytest.approx(0.01, abs=1e-6)
        assert float(root_mse(y_true, y_pred)) == pytest.approx(0.1, abs=1e-6)

    def test_offsets_ignored_for_background_cells(self) -> None:
        # Only the positive cell's offsets are scored; the background cell's
        # large offset error must not affect the MAE.
        y_true = _batch([_cell(True, 0.0), _cell(False, 0.0)])
        y_pred = _batch([_cell(True, 0.1), _cell(False, 0.9)])
        assert float(mae(y_true, y_pred)) == pytest.approx(0.1, abs=1e-6)


class TestMetricStateResets:
    """Regression test for the module-level metric singleton bug."""

    def test_consecutive_calls_are_independent(self) -> None:
        y_true = _batch([_cell(True, 0.0)])
        first = float(mae(y_true, _batch([_cell(True, 0.05)])))
        second = float(mae(y_true, _batch([_cell(True, 0.1)])))
        assert first == pytest.approx(0.05, abs=1e-6)
        # A leaked singleton would average both calls to 0.075; a fresh metric
        # per call yields exactly the second batch's error.
        assert second == pytest.approx(0.1, abs=1e-6)


class TestClassMAP:
    def test_perfect_classification_scores_one(self) -> None:
        y = _batch([_cell(True), _cell(False), _cell(False)])
        assert float(class_mAP(y, y)) == pytest.approx(1.0)

    def test_all_wrong_classification_scores_zero(self) -> None:
        y_true = _batch([_cell(True), _cell(False)])
        y_pred = _batch([_cell(False), _cell(True)])
        assert float(class_mAP(y_true, y_pred)) == pytest.approx(0.0)


class TestClassificationMetrics:
    def test_precision_and_recall_perfect(self) -> None:
        y = _batch([_cell(True), _cell(False), _cell(False)])
        assert float(precision(y, y)) == pytest.approx(1.0)
        assert float(recall(y, y)) == pytest.approx(1.0)

    def test_f1_perfect_for_active_column(self) -> None:
        # f1 scores each box position as a separate "class"; the foreground
        # cell's column must score 1.0 on a perfect prediction.
        y = _batch([_cell(True), _cell(False), _cell(False)])
        result = f1(y, y)
        assert float(result[0]) == pytest.approx(1.0)
