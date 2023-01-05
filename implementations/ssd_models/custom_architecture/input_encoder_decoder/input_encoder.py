import numpy as np

from input_encoder_decoder.bb_utils import get_matches, get_neutral_centers, get_matches2


class SSDInputEncoder:
    def __init__(self,
                 img_height,
                 img_width,
                 n_classes,
                 predictor_sizes,
                 normalize_coords=True,
                 background_id=0):

        self.img_height = img_height
        self.img_width = img_width
        self.n_classes = n_classes + 1
        self.normalize_coords = normalize_coords
        self.background_id = background_id
        self.predictor_sizes = np.array(predictor_sizes)

        self.boxes_list = []
        self.wh_list_diag = []
        self.steps_diag = []
        self.offsets_diag = []

        #######################################################
        # Compute the bounding boxes for each predictor layer.
        #######################################################

        for i in range(len(self.predictor_sizes)):
            boxes, center, step, offset = self.generate_bounding_boxes_per_layer(self.predictor_sizes[i],
                                                                                 diagnostics=True)
            self.boxes_list.append(boxes)
            self.steps_diag.append(step)
            self.offsets_diag.append(offset)

    def __call__(self, ground_truth_labels):
        class_id = 0
        tl_x = 1
        tl_y = 2
        tr_x = 3
        tr_y = 4
        bl_x = 5
        bl_y = 6
        br_x = 7
        br_y = 8
        cx = 9
        cy = 10

        batch_size = len(ground_truth_labels)

        y_encoded = self.generate_encoding_template(batch_size=batch_size)

        y_encoded[:, :, self.background_id] = 1  # All boxes are background boxes by default
        class_vectors = np.eye(self.n_classes)
        for i in range(batch_size):

            # If there is no ground truth for this batch item, there is nothing to match.
            if ground_truth_labels[i].size == 0:
                continue

            labels = ground_truth_labels[i].astype(float)  # The labels for this batch item

            if self.normalize_coords:
                labels[:, [tl_y, tr_y, bl_y, br_y, cy]] /= self.img_height
                labels[:, [tl_x, tr_x, bl_x, br_x, cx]] /= self.img_width

            classes_one_hot = class_vectors[labels[:, class_id].astype(int)]
            labels_one_hot = np.concatenate(
                [classes_one_hot, labels[:, [tl_x, tl_y, tr_x, tr_y, bl_x, bl_y, br_x, br_y]]], axis=-1)

            matches = get_matches2(labels[:, [cx, cy]], y_encoded[i, :, -2:])
            y_encoded[i, matches, :-2] = labels_one_hot

            # positives = y_encoded[i, y_encoded[i, :, 0] == 0]
            # neutral_idx = get_neutral_centers(positives, y_encoded[i, :, -2:])

            # y_encoded[i, neutral_idx, 1] = 1

        y_encoded[:, :, [-10, -8, -6, -4]] -= np.expand_dims(y_encoded[:, :, -2], axis=-1)
        y_encoded[:, :, [-9, -7, -5, -3]] -= np.expand_dims(y_encoded[:, :, -1], axis=-1)

        return y_encoded

    def generate_bounding_boxes_per_layer(self,
                                          feature_map_size,
                                          diagnostics=True):

        step_height = self.img_height / feature_map_size[0]
        step_width = self.img_width / feature_map_size[1]

        offset_height = 0.5
        offset_width = 0.5

        # Compute grid for anchor box center coordinates
        cy = np.linspace(step_height * offset_height, (offset_height + feature_map_size[0] - 1) * step_height,
                         feature_map_size[0])

        cx = np.linspace(step_width * offset_width, (offset_width + feature_map_size[1] - 1) * step_width,
                         feature_map_size[1])

        cx_grid, cy_grid = np.meshgrid(cx, cy)

        cx_grid = np.expand_dims(cx_grid, -1)
        cy_grid = np.expand_dims(cy_grid, -1)

        boxes_tensor = np.zeros((feature_map_size[0], feature_map_size[1], 1, 2))
        boxes_tensor[:, :, :, 0] = np.tile(cx_grid, (1, 1, 1))  # Set cx
        boxes_tensor[:, :, :, 1] = np.tile(cy_grid, (1, 1, 1))  # Set cy

        # Get tensor of shape (feature_map_size[0], feature_map_size[1], n_boxes, 8) with the final dimension
        # containing the corner coordinates of each bounding box

        if self.normalize_coords:
            boxes_tensor[:, :, :, 0] /= self.img_width
            boxes_tensor[:, :, :, 1] /= self.img_height
            step_height /= self.img_height
            step_width /= self.img_width

        if diagnostics:
            return boxes_tensor, (cy, cx), (step_height, step_width), (offset_height, offset_width)
        else:
            return boxes_tensor

    def generate_encoding_template(self, batch_size, diagnostics=False):
        boxes_batch = []
        for boxes in self.boxes_list:
            boxes = np.expand_dims(boxes, axis=0)
            boxes = np.tile(boxes, (batch_size, 1, 1, 1, 1))

            boxes = np.reshape(boxes, (batch_size, -1, 2))
            boxes_batch.append(boxes)

        boxes_tensor = np.concatenate(boxes_batch, axis=1)
        gt_tensor = np.zeros((batch_size, boxes_tensor.shape[1], 8))

        classes_tensor = np.zeros((batch_size, boxes_tensor.shape[1], self.n_classes))

        y_encoding_template = np.concatenate((classes_tensor, gt_tensor, boxes_tensor), axis=2)

        if diagnostics:
            return y_encoding_template, self.wh_list_diag, self.steps_diag, self.offsets_diag
        else:
            return y_encoding_template
