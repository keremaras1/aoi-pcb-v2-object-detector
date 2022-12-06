import keras.backend as K
from keras.layers import Layer, InputSpec
import numpy as np


class GridCenters(Layer):
    def __init__(self,
                 img_height,
                 img_width,
                 normalize_coords=True,
                 **kwargs):

        self.img_height = img_height
        self.img_width = img_width
        self.normalize_coords = normalize_coords
        super(GridCenters, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        super(GridCenters, self).build(input_shape)

    def call(self, x):
        if K.image_data_format() == 'channels_last':
            batch_size, feature_map_height, feature_map_width, feature_map_channels = K.int_shape(x)
        else:
            batch_size, feature_map_channels, feature_map_height, feature_map_width = K.int_shape(x)

        step_height = self.img_height / feature_map_height
        step_width = self.img_width / feature_map_width

        offset_height = 0.5
        offset_width = 0.5

        # Compute grid for anchor box center coordinates
        cy = np.linspace(step_height * offset_height, (offset_height + feature_map_height - 1) * step_height,
                         feature_map_height)

        cx = np.linspace(step_width * offset_width, (offset_width + feature_map_width - 1) * step_width,
                         feature_map_width)

        cx_grid, cy_grid = np.meshgrid(cx, cy)

        cx_grid = np.expand_dims(cx_grid, -1)
        cy_grid = np.expand_dims(cy_grid, -1)

        boxes_tensor = np.zeros((feature_map_height, feature_map_width, 1, 2))
        boxes_tensor[:, :, :, 0] = np.tile(cx_grid, (1, 1, 1))  # Set cx
        boxes_tensor[:, :, :, 1] = np.tile(cy_grid, (1, 1, 1))  # Set cy

        if self.normalize_coords:
            boxes_tensor[:, :, :, 0] /= self.img_width
            boxes_tensor[:, :, :, 1] /= self.img_height
            step_height /= self.img_height
            step_width /= self.img_width

        boxes_tensor = np.expand_dims(boxes_tensor, axis=0)
        boxes_tensor = K.tile(K.constant(boxes_tensor, dtype='float32'), (K.shape(x)[0], 1, 1, 1, 1))

        return boxes_tensor

    def compute_output_shape(self, input_shape):
        if K.image_data_format() == 'channels_last':
            batch_size, feature_map_height, feature_map_width, feature_map_channels = input_shape
        else:
            batch_size, feature_map_channels, feature_map_height, feature_map_width = input_shape
        return batch_size, feature_map_height, feature_map_width, 1, 2
    
    def get_config(self):
        config = {
            'img_height': self.img_height,
            'img_width': self.img_width,
            'normalize_coords': self.normalize_coords
        }
        base_config = super(GridCenters, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
