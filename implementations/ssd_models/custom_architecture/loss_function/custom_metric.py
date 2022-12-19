import tensorflow as tf


def custom_mse(y_true, y_pred):
    
    batch_size = tf.shape(y_pred)[0]
    n_boxes = tf.shape(y_pred)[1]

    positives = tf.cast(tf.reduce_max(y_true[:, :, 1:-10], axis=-1), dtype=tf.float64)
    n_positive = tf.cast(tf.math.count_nonzero(positives), dtype=tf.float64)

    mse_per_center = tf.cast(tf.reduce_mean((y_true[:, :, -10:-2] - y_pred[:, :, -10:-2])**2, axis=-1)**2, dtype=tf.float64)
    
    mse_per_batch = tf.reduce_sum(mse_per_center * positives) / tf.cast(n_positive, dtype=tf.float64)
    mse_per_batch = mse_per_batch * tf.cast(batch_size, dtype=tf.float64)

    return mse_per_batch

def custom_mse(y_true, y_pred):
    
    batch_size = tf.shape(y_pred)[0]
    n_boxes = tf.shape(y_pred)[1]

    pred_positives = tf.cast(tf.reduce_max(y_pred[:, :, 1:-10], axis=-1), dtype=tf.float64)
    n_positive = tf.cast(tf.math.count_nonzero(pred_positives), dtype=tf.float64)

    mse_per_center = tf.cast(tf.reduce_mean((y_true[:, :, -10:-2] - y_pred[:, :, -10:-2])**2, axis=-1)**2, dtype=tf.float64)
    
    mse_per_batch = tf.reduce_sum(mse_per_center * positives) / tf.cast(n_positive, dtype=tf.float64)
    mse_per_batch = mse_per_batch * tf.cast(batch_size, dtype=tf.float64)

    return mse_per_batch
