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
    
    n_classes = tf.shape(y_true[:, :, :-10])[-1]
    
    class_true = tf.cast(tf.argmax(y_true[:, :, :-10], axis=-1), dtype=tf.float32)
    class_pred = tf.cast(tf.argmax(y_pred[:, :, :-10], axis=-1), dtype=tf.float32)
    
    def average_precision(c):
        non_negative_true = tf.equal(class_true, c)
        non_negative_pred = tf.equal(class_pred, c)
        
        negative_true = tf.not_equal(class_true, c)
        negative_pred = tf.not_equal(class_pred, c)
        
        tp = tf.cast(tf.logical_and(non_negative_true, non_negative_pred), dtype=tf.float32)
        fp = tf.cast(tf.logical_and(negative_true, non_negative_pred), dtype=tf.float32)
        fn = tf.cast(tf.logical_and(non_negative_true, negative_pred), dtype=tf.float32)
        
        n_tp = tf.cast(tf.reduce_sum(tp), dtype=tf.float32)
        n_fp = tf.cast(tf.reduce_sum(fp), dtype=tf.float32)
        n_fn = tf.cast(tf.reduce_sum(fn), dtype=tf.float32)
        
        precision = n_tp / (n_tp + n_fp)
        recall = n_tp / (n_tp + n_fn)
        
        average_p = precision * recall
        
        is_nan = tf.math.is_nan(average_p)
        
        final = tf.where(is_nan, tf.zeros_like(average_p), average_p)
        
        return final
    
    ap = tf.map_fn(average_precision, tf.range(1, n_classes, dtype=tf.float32), dtype=tf.float32)
    
    return tf.reduce_mean(ap)
