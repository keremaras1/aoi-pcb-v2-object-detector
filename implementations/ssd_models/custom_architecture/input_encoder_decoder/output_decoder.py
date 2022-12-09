import numpy as np


def decode_detections(y_pred,
                      normalize_coords=True,
                      img_height=None,
                      img_width=None):

    y_decoded = np.copy(y_pred[:, :, -12:-2])
    y_decoded[:, :, 0] = np.argmax(y_pred[:, :, :-10], axis=-1)
    y_decoded[:, :, 1] = np.amax(y_pred[:, :, :-10], axis=-1)

    y_decoded[:, :, [2, 4, 6, 8]] = y_pred[:, :, [-10, -8, -6, -4]] + np.expand_dims(y_pred[:, :, -2], axis=-1)
    y_decoded[:, :, [3, 5, 7, 9]] = y_pred[:, :, [-9, -7, -5, -3]] + np.expand_dims(y_pred[:, :, -1], axis=-1)

    if normalize_coords:
        y_decoded[:, :, [2, 4, 6, 8]] *= img_width
        y_decoded[:, :, [3, 5, 7, 9]] *= img_height

    y_pred_final = []
    for batch_item in y_decoded:
        non_background = batch_item[np.nonzero(batch_item[:, 0])]
        y_pred_final.append(non_background)

    return y_pred_final
