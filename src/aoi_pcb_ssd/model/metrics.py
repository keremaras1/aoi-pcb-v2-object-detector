# SPDX-License-Identifier: Apache-2.0
"""Training and evaluation metrics for IC corner-point detection.

All metric functions follow the Keras metric signature ``f(y_true, y_pred)``
and can be passed directly to ``model.compile(metrics=[...])``.

The prediction tensor layout (12 values per anchor):
  y[:, :, :-10]    — class scores (softmax, n_classes values)
  y[:, :, -10:-2]  — eight corner-point offsets (tl/tr/bl/br × x/y)
  y[:, :, -2:]     — grid-cell anchor centres (cx, cy)

``tf.keras.metrics.*`` objects own internal ``tf.Variable`` accumulators,
which ``tf.function`` tracing (used internally by ``model.fit``/``model.evaluate``)
only allows to be created on the first trace of a graph. The module-level
instances below are therefore each created exactly once; every metric function
calls ``reset_state()`` immediately before ``update_state()`` so its result
reflects only the current batch, with no accumulation across calls.
"""

import tensorflow as tf

_f1_metric = tf.keras.metrics.F1Score(threshold=0.5)
_precision_metric = tf.keras.metrics.Precision()
_recall_metric = tf.keras.metrics.Recall()
_mae_metric = tf.keras.metrics.MeanAbsoluteError()
_mse_metric = tf.keras.metrics.MeanSquaredError()
_root_mse_metric = tf.keras.metrics.RootMeanSquaredError()

# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------


def class_mAP(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Custom mAP: mean of (precision × recall) over all foreground classes.

    As described in Section V.C.1 of the paper. Because the grid-centre
    matching strategy is binary (each IC maps to exactly one anchor), there
    is no threshold to vary for a proper precision-recall curve. Instead,
    precision × recall is computed per class and averaged — giving a value
    in [0, 1] that jointly penalises both precision and recall failures.

    Args:
        y_true: Target tensor of shape ``(batch, n_boxes, 12)``.
        y_pred: Prediction tensor of shape ``(batch, n_boxes, 12)``.

    Returns:
        Scalar mean of per-class (precision × recall), averaged over all
        foreground classes (class index ≥ 1).
    """
    n_classes = tf.shape(y_true[:, :, :-10])[-1]
    class_true = tf.cast(tf.argmax(y_true[:, :, :-10], axis=-1), tf.float32)
    class_pred = tf.cast(tf.argmax(y_pred[:, :, :-10], axis=-1), tf.float32)

    # Traced into a graph by tf.map_fn below, so coverage.py cannot record
    # these lines; their behaviour is verified by tests/test_metrics.py.
    def _per_class_ap(c: tf.Tensor) -> tf.Tensor:  # pragma: no cover
        tp = tf.cast(tf.logical_and(tf.equal(class_true, c), tf.equal(class_pred, c)), tf.float32)
        fp = tf.cast(
            tf.logical_and(tf.not_equal(class_true, c), tf.equal(class_pred, c)), tf.float32
        )
        fn = tf.cast(
            tf.logical_and(tf.equal(class_true, c), tf.not_equal(class_pred, c)), tf.float32
        )
        n_tp = tf.reduce_sum(tp)
        precision = n_tp / (n_tp + tf.reduce_sum(fp))
        recall = n_tp / (n_tp + tf.reduce_sum(fn))
        ap = precision * recall
        return tf.where(tf.math.is_nan(ap), tf.zeros_like(ap), ap)

    ap_per_class = tf.map_fn(
        _per_class_ap,
        tf.cast(tf.range(1, n_classes), tf.float32),
        dtype=tf.float32,
    )
    return tf.reduce_mean(ap_per_class)


def f1(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """F1 score over foreground class predictions.

    Args:
        y_true: Target tensor of shape ``(batch, n_boxes, 12)``.
        y_pred: Prediction tensor of shape ``(batch, n_boxes, 12)``.

    Returns:
        Scalar F1 score.
    """
    _f1_metric.reset_state()
    class_true = tf.squeeze(y_true[:, :, 1:-10], axis=-1)
    class_pred = tf.squeeze(y_pred[:, :, 1:-10], axis=-1)
    _f1_metric.update_state(class_true, class_pred)
    return _f1_metric.result()


def precision(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Micro-average precision over foreground class predictions.

    Args:
        y_true: Target tensor of shape ``(batch, n_boxes, 12)``.
        y_pred: Prediction tensor of shape ``(batch, n_boxes, 12)``.

    Returns:
        Scalar precision value.
    """
    _precision_metric.reset_state()
    class_true = tf.reshape(y_true[:, :, 1:-10], [-1])
    class_pred = tf.reshape(y_pred[:, :, 1:-10], [-1])
    _precision_metric.update_state(class_true, class_pred)
    return _precision_metric.result()


def recall(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Micro-average recall over foreground class predictions.

    Args:
        y_true: Target tensor of shape ``(batch, n_boxes, 12)``.
        y_pred: Prediction tensor of shape ``(batch, n_boxes, 12)``.

    Returns:
        Scalar recall value.
    """
    _recall_metric.reset_state()
    class_true = tf.reshape(y_true[:, :, 1:-10], [-1])
    class_pred = tf.reshape(y_pred[:, :, 1:-10], [-1])
    _recall_metric.update_state(class_true, class_pred)
    return _recall_metric.result()


# ---------------------------------------------------------------------------
# Regression metrics (positive cells only)
# ---------------------------------------------------------------------------


def _positive_offsets(y_true: tf.Tensor, y_pred: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Extract predicted and true corner offsets for positive (IC) cells only."""
    true_off = tf.reshape(y_true[:, :, -10:-2], [-1, 8])
    pred_off = tf.reshape(y_pred[:, :, -10:-2], [-1, 8])
    y_true_2d = tf.reshape(y_true, (-1, tf.shape(y_true)[-1]))
    class_true = tf.cast(tf.argmax(y_true_2d[:, :-10], axis=-1), tf.float32)
    pos_idx = tf.reshape(tf.where(tf.not_equal(class_true, 0.0)), [-1])
    return tf.gather(true_off, pos_idx), tf.gather(pred_off, pos_idx)


def mae(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Mean absolute error of corner offsets for positive cells.

    Args:
        y_true: Target tensor of shape ``(batch, n_boxes, 12)``.
        y_pred: Prediction tensor of shape ``(batch, n_boxes, 12)``.

    Returns:
        Scalar MAE over positive-cell offset predictions.
    """
    _mae_metric.reset_state()
    true_off, pred_off = _positive_offsets(y_true, y_pred)
    _mae_metric.update_state(true_off, pred_off)
    return _mae_metric.result()


def mse(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Mean squared error of corner offsets for positive cells.

    Args:
        y_true: Target tensor of shape ``(batch, n_boxes, 12)``.
        y_pred: Prediction tensor of shape ``(batch, n_boxes, 12)``.

    Returns:
        Scalar MSE over positive-cell offset predictions.
    """
    _mse_metric.reset_state()
    true_off, pred_off = _positive_offsets(y_true, y_pred)
    _mse_metric.update_state(true_off, pred_off)
    return _mse_metric.result()


def root_mse(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Root mean squared error of corner offsets for positive cells.

    Args:
        y_true: Target tensor of shape ``(batch, n_boxes, 12)``.
        y_pred: Prediction tensor of shape ``(batch, n_boxes, 12)``.

    Returns:
        Scalar RMSE over positive-cell offset predictions.
    """
    _root_mse_metric.reset_state()
    true_off, pred_off = _positive_offsets(y_true, y_pred)
    _root_mse_metric.update_state(true_off, pred_off)
    return _root_mse_metric.result()
