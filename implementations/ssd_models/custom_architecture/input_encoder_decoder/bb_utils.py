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
