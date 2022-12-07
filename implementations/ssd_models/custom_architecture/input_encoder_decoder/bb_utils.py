import numpy as np


def convert_center_to_corners(tensor):
    tensor1 = np.zeros((tensor.shape[0], tensor.shape[1], tensor.shape[2], 10), dtype=np.float)

    tensor1[..., 0] = tensor[..., 0] - (tensor[..., 2] / 2.0)  # set tl_x
    tensor1[..., 1] = tensor[..., 1] - (tensor[..., 3] / 2.0)  # set tl_y
    tensor1[..., 2] = tensor[..., 0] + (tensor[..., 2] / 2.0)  # set tr_x
    tensor1[..., 3] = tensor[..., 1] - (tensor[..., 3] / 2.0)  # set tr_y
    tensor1[..., 4] = tensor[..., 0] - (tensor[..., 2] / 2.0)  # set bl_x
    tensor1[..., 5] = tensor[..., 1] + (tensor[..., 3] / 2.0)  # set bl_y
    tensor1[..., 6] = tensor[..., 0] + (tensor[..., 2] / 2.0)  # set br_x
    tensor1[..., 7] = tensor[..., 1] + (tensor[..., 3] / 2.0)  # set br_y
    tensor1[..., 8] = tensor[..., 0]  # set cx
    tensor1[..., 9] = tensor[..., 1]  # set cy

    return tensor1


def get_matches(gt_box_centers, bb_box_centers, cell_hw):
    assert len(cell_hw) == 1, 'Expected a single feature map cell size instead got {i}'.format(i=len(cell_hw))

    # gt.shape = (n_gt, 2), bb.shape = (n_bb, 2)
    hw_tuple = cell_hw[0]
    m = gt_box_centers.shape[0]
    n = bb_box_centers.shape[0]

    gt_matrix = np.repeat(gt_box_centers, n, axis=0)
    bb_matrix = np.tile(bb_box_centers, (m, 1))

    dist_matrix = np.abs(bb_matrix - gt_matrix)

    valid_x = dist_matrix[:, 0] <= hw_tuple[1] / 2
    valid_y = dist_matrix[:, 1] <= hw_tuple[0] / 2

    valid_center_idx = np.argwhere(valid_x & valid_y)
    matches_mat = np.zeros((m, n), dtype=int)

    duplicate_checker = []
    for idx in valid_center_idx:
        i = int(idx / n)

        if i in duplicate_checker:
            continue
        else:
            duplicate_checker.append(i)

        j = idx % n

        matches_mat[i, j] = 1

    return matches_mat


def get_neutral_centers(positive_labels, all_centers):
    # positive_offsets: (n_matched, 8), positive centers: (n_matched, 2), negative_centers: (n_not_centers, 2)
    positive_centers = positive_labels[:, -2:]

    radius_x = np.max(np.abs(positive_labels[:, [-10, -8, -6, -4]] - np.expand_dims(positive_labels[:, -2], axis=-1)),
                      axis=0)
    radius_y = np.max(np.abs(positive_labels[:, [-9, -7, -5, -3]] - np.expand_dims(positive_labels[:, -1], axis=-1)),
                      axis=0)

    lower_bound_x = positive_centers[:, 0] - radius_x
    upper_bound_x = positive_centers[:, 0] + radius_x

    lower_bound_y = positive_centers[:, 1] - radius_y
    upper_bound_y = positive_centers[:, 1] + radius_y

    x_bounds = np.array((lower_bound_x, upper_bound_x)).T
    y_bounds = np.array((lower_bound_y, upper_bound_y)).T

    index = []
    for center in all_centers:
        if center.tolist() in positive_centers.tolist():
            index.append(False)
            continue

        x_crit = (x_bounds[:, 0] < center[0]) & (x_bounds[:, 1] > center[0])
        y_crit = (y_bounds[:, 0] < center[1]) & (y_bounds[:, 1] > center[1])
        if np.any(x_crit & y_crit):
            index.append(True)
        else:
            index.append(False)

    return np.argwhere(np.array(index) == True)
