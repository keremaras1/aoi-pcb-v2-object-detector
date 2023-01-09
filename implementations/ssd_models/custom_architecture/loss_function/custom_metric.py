import tensorflow as tf


def offset_MAE(y_true, y_pred):
    
    pred_offsets = y_pred[:, :, -10:-2]
    true_offsets = y_true[:, :, -10:-2]
    
    class_true = tf.cast(tf.argmax(y_true[:, :, :-10], axis=-1), dtype=tf.float32)
    
    positive_true_idx = tf.where(tf.not_equal(class_true, 0.0))
    
    filtered_pred_offsets = tf.gather(pred_offsets, positive_true_idx, axis=1)
    positive_true_offsets = tf.gather(true_offsets, positive_true_idx, axis=1)
    
    absolute_error = tf.abs(positive_true_offsets - filtered_pred_offsets)
    
    mean_absolute_error = tf.reduce_mean(absolute_error)
    
    return mean_absolute_error


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
