# Portions of this file are derived from
# https://github.com/pierluigiferrari/ssd_keras
# Copyright 2018 Pierluigi Ferrari, licensed under the Apache License, Version 2.0.
# Modifications Copyright 2024 Kerem Aras, also licensed under the Apache License, Version 2.0.
"""Custom SSD multibox loss for IC corner-point detection (Section V of the paper).

The total loss is a weighted sum of two terms:

  total = (class_loss + α * loc_loss) / max(1, N_pos) * batch_size

Classification (Section V.A):
  Softmax cross-entropy. Hard negative mining keeps only the top-loss
  background cells, with a negative-to-positive ratio capped at ``neg_pos_ratio``.
  This prevents the overwhelming number of background cells from dominating
  training.

Localisation (Section V.B):
  A piecewise L1/L2 hybrid applied to positive cells only:
    loss(x) = |x|        if |x| < 1
    loss(x) = x²         if |x| ≥ 1
  This is the inverse of the standard Huber/Smooth-L1 loss — L1 for small
  errors keeps larger gradients at low learning rates; L2 for large errors
  pulls outliers aggressively early in training. All coordinates are
  normalised to [0,1] so errors are always < 1 in practice.
"""

import tensorflow as tf


class AOILoss:
    """Multibox loss combining classification and corner-point localisation.

    Args:
        neg_pos_ratio: Maximum ratio of hard-mined negatives to positives.
            For a batch with 3 positive cells, at most 9 negative cells
            contribute to the classification loss.
        n_neg_min: Minimum number of negatives to keep even when there are
            no positives in the batch.
        alpha: Weight applied to the localisation loss relative to the
            classification loss. The paper uses α=3.
    """

    def __init__(
        self,
        neg_pos_ratio: int = 3,
        n_neg_min: int = 0,
        alpha: float = 1.0,
    ) -> None:
        self.neg_pos_ratio = neg_pos_ratio
        self.n_neg_min = n_neg_min
        self.alpha = alpha

    def _loc_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Piecewise L1/L2 loss per anchor (summed over the 8 corner coordinates)."""
        abs_err = tf.math.abs(y_true - y_pred)
        elementwise = tf.where(tf.math.less(abs_err, 1.0), abs_err, tf.square(y_true - y_pred))
        return tf.reduce_sum(elementwise, axis=-1)

    def _cls_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Softmax cross-entropy per anchor (summed over class scores)."""
        y_pred = tf.maximum(y_pred, 1e-15)
        return -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)

    def compute_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Compute the total multibox loss for one batch.

        Note: Keras compiles this method into a TF graph during ``model.compile``.
        When reloading a saved model, use ``compile=False`` then re-call
        ``model.compile`` manually to avoid serialisation issues with the
        custom loss.

        Args:
            y_true: Encoded target tensor of shape ``(batch, n_boxes, 12)``.
            y_pred: Model output tensor of shape ``(batch, n_boxes, 12)``.

        Returns:
            Per-sample loss vector of shape ``(batch,)``.
        """
        batch_size = tf.shape(y_pred)[0]
        n_boxes = tf.shape(y_pred)[1]

        cls_loss = tf.cast(self._cls_loss(y_true[:, :, :-10], y_pred[:, :, :-10]), tf.float32)
        loc_loss = tf.cast(self._loc_loss(y_true[:, :, -10:-2], y_pred[:, :, -10:-2]), tf.float32)

        # Positive / negative masks from the encoded target
        negatives = y_true[:, :, 0]
        positives = tf.cast(tf.reduce_max(y_true[:, :, 1:-10], axis=-1), tf.float32)
        n_positive = tf.reduce_sum(positives)

        # Positive classification loss
        pos_cls_loss = tf.reduce_sum(cls_loss * positives, axis=-1)

        # Hard negative mining: keep the highest-loss background cells
        neg_cls_loss_all = cls_loss * negatives
        n_neg_losses = tf.math.count_nonzero(neg_cls_loss_all, dtype=tf.int32)
        n_neg_keep = tf.minimum(
            tf.maximum(self.neg_pos_ratio * tf.cast(n_positive, tf.int32), self.n_neg_min),
            n_neg_losses,
        )

        def _no_negatives():
            return tf.zeros([batch_size])

        def _select_negatives():
            flat = tf.reshape(neg_cls_loss_all, [-1])
            # top_k runs on the CPU: the tensorflow-metal kernel bakes the
            # flattened length into its compiled graph and aborts on partial
            # batches. Selection returns indices only, so placement does not
            # affect the loss value.
            with tf.device("/CPU:0"):
                _, indices = tf.math.top_k(flat, k=n_neg_keep, sorted=False)
            mask = tf.scatter_nd(
                indices=tf.expand_dims(indices, axis=1),
                updates=tf.ones_like(indices, dtype=tf.int32),
                shape=tf.shape(flat),
            )
            mask_2d = tf.cast(tf.reshape(mask, [batch_size, n_boxes]), tf.float32)
            return tf.reduce_sum(cls_loss * mask_2d, axis=-1)

        neg_cls_loss = tf.cond(tf.equal(n_neg_losses, 0), _no_negatives, _select_negatives)

        total_cls_loss = pos_cls_loss + neg_cls_loss
        total_loc_loss = tf.reduce_sum(loc_loss * positives, axis=-1)

        total = (total_cls_loss + self.alpha * total_loc_loss) / tf.maximum(1.0, n_positive)
        return total * tf.cast(batch_size, tf.float32)
