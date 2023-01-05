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

def class_mAP(y_true, y_pred):
    class_true = tf.argmax(y_true[:, :, :-10], axis=-1)
    class_pred = tf.argmax(y_pred[:, :, :-10], axis=-1)
    
    average_precisions = tf.map_fn(lambda c: tf.metrics.average_precision_at_k(class_true, class_pred, k=c)[0], tf.range(2), dtype=tf.float64)
    
    return tf.reduce_mean(average_precisions)