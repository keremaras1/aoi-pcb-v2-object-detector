import tensorflow as tf

f1_metric = tf.keras.metrics.F1Score(threshold=0.5)
precision_metric = tf.keras.metrics.Precision()
recall_metric = tf.keras.metrics.Recall()


def f1(y_true, y_pred):
    class_true = tf.squeeze(y_true[:, :, 1:-10], axis=-1)
    class_pred = tf.squeeze(y_pred[:, :, 1:-10], axis=-1)

    f1_metric.update_state(class_true, class_pred)

    return f1_metric.result()


def precision_and_recall(y_true, y_pred):
    class_true = y_true[:, :, 1:-10]
    class_pred = y_pred[:, :, 1:-10]

    class_true_flat = tf.reshape(class_true, [-1])
    class_pred_flat = tf.reshape(class_pred, [-1])

    precision_metric.update_state(class_true_flat, class_pred_flat)

    micro_average_precision = precision_metric.result()

    recall_metric.update_state(class_true_flat, class_pred_flat)

    micro_average_recall = recall_metric.result()

    return micro_average_precision, micro_average_recall


def precision(y_true, y_pred):
    precision, _ = precision_and_recall(y_true, y_pred)

    return precision


def recall(y_true, y_pred):
    _, recall = precision_and_recall(y_true, y_pred)

    return recall
