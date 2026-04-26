import tensorflow as tf

rmse_metric = tf.keras.metrics.RootMeanSquaredError()
mse_metric = tf.keras.metrics.MeanSquaredError()
mape_metric = tf.keras.metrics.MeanAbsolutePercentageError()
mae_metric = tf.keras.metrics.MeanAbsoluteError()


def mae(y_true, y_pred):
    pred_offsets = y_pred[:, :, -10:-2]
    true_offsets = y_true[:, :, -10:-2]

    true_offsets_reshaped = tf.reshape(true_offsets, [-1, tf.shape(true_offsets)[-1]])
    pred_offsets_reshaped = tf.reshape(pred_offsets, [-1, tf.shape(pred_offsets)[-1]])

    y_true_2d = tf.reshape(y_true, (-1, tf.shape(y_true)[-1]))

    class_true = tf.cast(tf.argmax(y_true_2d[:, :-10], axis=-1), dtype=tf.float32)

    positive_true_idx = tf.where(tf.not_equal(class_true, 0.0))
    positive_true_idx_1d = tf.reshape(positive_true_idx, [-1])

    pos_pred_offsets = tf.gather(pred_offsets_reshaped, positive_true_idx_1d)
    pos_true_offsets = tf.gather(true_offsets_reshaped, positive_true_idx_1d)

    mae_metric.update_state(pos_true_offsets, pos_pred_offsets)

    return mae_metric.result()


def mae_percentage(y_true, y_pred):
    pred_offsets = y_pred[:, :, -10:-2]
    true_offsets = y_true[:, :, -10:-2]

    true_offsets_reshaped = tf.reshape(true_offsets, [-1, tf.shape(true_offsets)[-1]])
    pred_offsets_reshaped = tf.reshape(pred_offsets, [-1, tf.shape(pred_offsets)[-1]])

    y_true_2d = tf.reshape(y_true, (-1, tf.shape(y_true)[-1]))

    class_true = tf.cast(tf.argmax(y_true_2d[:, :-10], axis=-1), dtype=tf.float32)

    positive_true_idx = tf.where(tf.not_equal(class_true, 0.0))
    positive_true_idx_1d = tf.reshape(positive_true_idx, [-1])

    pos_pred_offsets = tf.gather(pred_offsets_reshaped, positive_true_idx_1d)
    pos_true_offsets = tf.gather(true_offsets_reshaped, positive_true_idx_1d)

    mape_metric.update_state(pos_true_offsets, pos_pred_offsets)

    return mape_metric.result()


def mse(y_true, y_pred):
    pred_offsets = y_pred[:, :, -10:-2]
    true_offsets = y_true[:, :, -10:-2]

    true_offsets_reshaped = tf.reshape(true_offsets, [-1, tf.shape(true_offsets)[-1]])
    pred_offsets_reshaped = tf.reshape(pred_offsets, [-1, tf.shape(pred_offsets)[-1]])

    y_true_2d = tf.reshape(y_true, (-1, tf.shape(y_true)[-1]))

    class_true = tf.cast(tf.argmax(y_true_2d[:, :-10], axis=-1), dtype=tf.float32)

    positive_true_idx = tf.where(tf.not_equal(class_true, 0.0))
    positive_true_idx_1d = tf.reshape(positive_true_idx, [-1])

    pos_pred_offsets = tf.gather(pred_offsets_reshaped, positive_true_idx_1d)
    pos_true_offsets = tf.gather(true_offsets_reshaped, positive_true_idx_1d)

    mse_metric.update_state(pos_true_offsets, pos_pred_offsets)

    return mse_metric.result()


def root_mse(y_true, y_pred):
    pred_offsets = y_pred[:, :, -10:-2]
    true_offsets = y_true[:, :, -10:-2]

    true_offsets_reshaped = tf.reshape(true_offsets, [-1, tf.shape(true_offsets)[-1]])
    pred_offsets_reshaped = tf.reshape(pred_offsets, [-1, tf.shape(pred_offsets)[-1]])

    y_true_2d = tf.reshape(y_true, (-1, tf.shape(y_true)[-1]))

    class_true = tf.cast(tf.argmax(y_true_2d[:, :-10], axis=-1), dtype=tf.float32)

    positive_true_idx = tf.where(tf.not_equal(class_true, 0.0))
    positive_true_idx_1d = tf.reshape(positive_true_idx, [-1])

    pos_pred_offsets = tf.gather(pred_offsets_reshaped, positive_true_idx_1d)
    pos_true_offsets = tf.gather(true_offsets_reshaped, positive_true_idx_1d)

    rmse_metric.update_state(pos_true_offsets, pos_pred_offsets)

    return rmse_metric.result()
