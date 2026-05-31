# Portions of this file are derived from
# https://github.com/pierluigiferrari/ssd_keras
# Copyright 2018 Pierluigi Ferrari, licensed under the Apache License, Version 2.0.
# Modifications Copyright 2024 Kerem Aras, also licensed under the Apache License, Version 2.0.
"""GridCenters: custom Keras layer that computes grid-cell anchor centre coordinates.

Unlike the upstream AnchorBoxes layer, this layer produces only the (cx, cy)
centre point of each feature map cell — no width, height, or variance terms.
The centres are computed once at graph construction time from the known image
and feature map dimensions, then tiled across the batch at runtime.
"""

import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.layers import InputSpec, Layer
from tensorflow.keras.utils import register_keras_serializable


@register_keras_serializable(package="aoi_pcb_ssd")
class GridCenters(Layer):
    """Generate (cx, cy) anchor centre coordinates for each feature map cell.

    Given an input feature map of shape ``(batch, H, W, C)``, computes a
    regular grid of cell-centre coordinates and returns a tensor of shape
    ``(batch, H, W, 1, 2)``. The centres are optionally normalised to [0, 1].

    Args:
        img_height: Original input image height in pixels.
        img_width: Original input image width in pixels.
        normalize_coords: If ``True``, divide coordinates by image dimensions
            so they fall in [0, 1].
    """

    def __init__(
        self,
        img_height: int,
        img_width: int,
        normalize_coords: bool = True,
        **kwargs,
    ) -> None:
        self.img_height = img_height
        self.img_width = img_width
        self.normalize_coords = normalize_coords
        super().__init__(**kwargs)

    def build(self, input_shape) -> None:
        self.input_spec = [InputSpec(shape=input_shape)]
        super().build(input_shape)

    def call(self, x):
        if K.image_data_format() == "channels_last":
            _, feature_map_height, feature_map_width, _ = K.int_shape(x)
        else:
            _, _, feature_map_height, feature_map_width = K.int_shape(x)

        step_h = self.img_height / feature_map_height
        step_w = self.img_width / feature_map_width
        offset = 0.5

        cy = np.linspace(
            step_h * offset,
            (offset + feature_map_height - 1) * step_h,
            feature_map_height,
        )
        cx = np.linspace(
            step_w * offset,
            (offset + feature_map_width - 1) * step_w,
            feature_map_width,
        )

        cx_grid, cy_grid = np.meshgrid(cx, cy)
        boxes = np.zeros((feature_map_height, feature_map_width, 1, 2))
        boxes[:, :, 0, 0] = cx_grid
        boxes[:, :, 0, 1] = cy_grid

        if self.normalize_coords:
            boxes[:, :, :, 0] /= self.img_width
            boxes[:, :, :, 1] /= self.img_height

        boxes = np.expand_dims(boxes, axis=0)
        return K.tile(K.constant(boxes, dtype="float32"), (K.shape(x)[0], 1, 1, 1, 1))

    def compute_output_shape(self, input_shape):
        if K.image_data_format() == "channels_last":
            _, fh, fw, _ = input_shape
        else:
            _, _, fh, fw = input_shape
        return (input_shape[0], fh, fw, 1, 2)

    def get_config(self):
        config = {
            "img_height": self.img_height,
            "img_width": self.img_width,
            "normalize_coords": self.normalize_coords,
        }
        return {**super().get_config(), **config}
