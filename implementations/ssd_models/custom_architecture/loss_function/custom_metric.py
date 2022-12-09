import tensorflow as tf


def custom_mse(y_true, y_pred):
    n_boxes = y_pred.shape[1]

    positives = tf.cast(tf.reduce_max(y_true[:, :, 1:-10], axis=-1), dtype=tf.float32)
    n_positive = tf.reduce_sum(positives)

    y_pred[:, :, [-10, -8, -6, -4]] += tf.expand_dims(y_pred[:, :, -2], axis=-1)
    y_pred[:, :, [-9, -7, -5, -3]] += tf.expand_dims(y_pred[:, :, -1], axis=-1)

    y_true[:, :, [-10, -8, -6, -4]] += tf.expand_dims(y_true[:, :, -2], axis=-1)
    y_true[:, :, [-9, -7, -5, -3]] += tf.expand_dims(y_true[:, :, -1], axis=-1)

    mse_per_center = tf.cast(tf.reduce_mean((y_true[:, :, -10:-2] - y_pred[:, :, -10:-2])**2, axis=-1), dtype=tf.float32)

    assert mse_per_center.shape[0] == n_boxes

    mse_per_batch = tf.reduce_sum(positives * mse_per_center) / tf.cast(n_positive, dtype=tf.float32)

    return mse_per_batch
