# Portions of this file are derived from
# https://github.com/pierluigiferrari/ssd_keras
# Copyright 2018 Pierluigi Ferrari, licensed under the Apache License, Version 2.0.
# Modifications Copyright 2024 Kerem Aras, also licensed under the Apache License, Version 2.0.
"""Custom SSD architecture for IC corner-point detection (Figure 3 of the paper).

Six-block convolutional feature extractor followed by a three-branch detection
head that simultaneously predicts IC class scores, four corner-point offsets,
and grid-cell anchor centres. The final output is a (64, 12) tensor per image.

Architecture constants below are defined in Section IV.B / Figure 3 of the
paper. Do not change them — they are the values used to produce the published
results.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Concatenate,
    Conv2D,
    GaussianNoise,
    Input,
    MaxPooling2D,
    ReLU,
    Rescaling,
    Reshape,
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import register_keras_serializable

from aoi_pcb_ssd.model.grid_centers import GridCenters


@register_keras_serializable(package="aoi_pcb_ssd")
class ChannelSwap(tf.keras.layers.Layer):
    """Permute image channels to a user-specified order.

    Replaces the upstream Lambda approach with a fully serializable layer,
    enabling ``load_model`` to work under default ``safe_mode=True``.

    Args:
        order: Channel index permutation, e.g. ``[2, 1, 0]`` for RGB→BGR.
    """

    def __init__(self, order: list[int], **kwargs) -> None:
        self.order = list(order)
        super().__init__(**kwargs)

    def call(self, x):
        return tf.stack([x[..., i] for i in self.order], axis=-1)

    def get_config(self):
        return {**super().get_config(), "order": self.order}


# Paper-defined architecture constants (Section IV.B, Figure 3).
_FILTERS = (32, 64, 128, 256, 512, 512)
_KERNEL_LARGE = (5, 5)  # blocks 1–2: wider receptive field (paper §IV.B, citing [29])
_KERNEL_SMALL = (3, 3)  # blocks 3–6
_BN_MOMENTUM = 0.99
_GAUSSIAN_NOISE_STD = 0.1


def build_custom_model(
    image_size: tuple[int, int, int],
    n_classes: int,
    l2_regularization: float = 0.0,
    normalize_coords: bool = True,
    subtract_mean: float | None = None,
    divide_by_stddev: float | None = None,
    swap_channels: list[int] | bool = False,
    return_predictor_sizes: bool = False,
) -> Model | tuple[Model, np.ndarray]:
    """Build the custom SSD model (Figure 3 of the paper).

    Args:
        image_size: ``(height, width, channels)`` of the input images.
        n_classes: Number of foreground classes (background is added internally).
        l2_regularization: L2 regularisation coefficient applied to all
            convolutional kernels.
        normalize_coords: Whether ``GridCenters`` should normalise anchor
            coordinates to [0, 1].
        subtract_mean: If set, subtract this scalar from input pixels before
            any other processing (e.g. 127.5 to centre the range).
        divide_by_stddev: If set, divide input pixels by this scalar after
            mean subtraction (e.g. 127.5 to map to [−1, 1]).
        swap_channels: If a list of indices is provided, permute the input
            channels in that order before processing.
        return_predictor_sizes: If ``True``, also return the spatial dimensions
            of the detection head feature map as a ``(1, 2)`` array.

    Returns:
        The compiled Keras ``Model``, or a ``(model, predictor_sizes)`` tuple
        when ``return_predictor_sizes=True``.
    """
    n_boxes = 1
    n_classes += 1
    l2_reg = l2_regularization
    img_height, img_width, img_channels = image_size

    # --- Input preprocessing (serializable; no Lambda layers) -----------------
    x = Input(shape=(img_height, img_width, img_channels))
    x1 = x
    if subtract_mean is not None or divide_by_stddev is not None:
        _scale = 1.0 / divide_by_stddev if divide_by_stddev is not None else 1.0
        _offset = float(-subtract_mean * _scale) if subtract_mean is not None else 0.0
        x1 = Rescaling(scale=_scale, offset=_offset, name="input_rescaling")(x1)
    if swap_channels:
        x1 = ChannelSwap(order=list(swap_channels), name="input_channel_swap")(x1)

    # --- Feature extractor (6 blocks) ----------------------------------------
    x1 = GaussianNoise(_GAUSSIAN_NOISE_STD)(x1)

    conv1 = Conv2D(
        _FILTERS[0],
        _KERNEL_LARGE,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(l2_reg),
        name="conv1",
    )(x1)
    conv1 = BatchNormalization(axis=3, momentum=_BN_MOMENTUM, name="bn1")(conv1)
    conv1 = ReLU(name="relu1")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), name="pool1")(conv1)

    conv2 = Conv2D(
        _FILTERS[1],
        _KERNEL_LARGE,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(l2_reg),
        name="conv2",
    )(pool1)
    conv2 = BatchNormalization(axis=3, momentum=_BN_MOMENTUM, name="bn2")(conv2)
    conv2 = ReLU(name="relu2")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), name="pool2")(conv2)

    conv3 = Conv2D(
        _FILTERS[2],
        _KERNEL_SMALL,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(l2_reg),
        name="conv3",
    )(pool2)
    conv3 = BatchNormalization(axis=3, momentum=_BN_MOMENTUM, name="bn3")(conv3)
    conv3 = ReLU(name="relu3")(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), name="pool3")(conv3)

    conv4 = Conv2D(
        _FILTERS[3],
        _KERNEL_SMALL,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(l2_reg),
        name="conv4",
    )(pool3)
    conv4 = BatchNormalization(axis=3, momentum=_BN_MOMENTUM, name="bn4")(conv4)
    conv4 = ReLU(name="relu4")(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2), name="pool4")(conv4)

    conv5 = Conv2D(
        _FILTERS[4],
        _KERNEL_SMALL,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(l2_reg),
        name="conv5",
    )(pool4)
    conv5 = BatchNormalization(axis=3, momentum=_BN_MOMENTUM, name="bn5")(conv5)
    conv5 = ReLU(name="relu5")(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2), name="pool5")(conv5)

    conv6 = Conv2D(
        _FILTERS[5],
        _KERNEL_SMALL,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(l2_reg),
        name="conv6",
    )(pool5)
    conv6 = BatchNormalization(axis=3, momentum=_BN_MOMENTUM, name="bn6")(conv6)
    conv6 = ReLU(name="relu6")(conv6)

    # --- Three-branch detection head (all from conv6) -------------------------
    classes7 = Conv2D(
        n_boxes * n_classes,
        _KERNEL_SMALL,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(l2_reg),
        name="classes7",
    )(conv6)
    corners7 = Conv2D(
        n_boxes * 8,
        _KERNEL_SMALL,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(l2_reg),
        name="corners7",
    )(conv6)
    centers7 = GridCenters(
        img_height, img_width, normalize_coords=normalize_coords, name="centers7"
    )(corners7)

    classes7_reshaped = Reshape((-1, n_classes), name="classes7_reshaped")(classes7)
    corners7_reshaped = Reshape((-1, 8), name="corners7_reshaped")(corners7)
    centers7_reshaped = Reshape((-1, 2), name="centers7_reshaped")(centers7)

    classes_softmax = Activation("softmax", name="classes_softmax")(classes7_reshaped)

    predictions = Concatenate(axis=2, name="predictions")(
        [classes_softmax, corners7_reshaped, centers7_reshaped]
    )

    model = Model(inputs=x, outputs=predictions)

    if return_predictor_sizes:
        predictor_sizes = np.array([classes7.shape[1:3]])
        return model, predictor_sizes
    return model
