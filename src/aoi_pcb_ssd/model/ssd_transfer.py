# Portions of this file are derived from
# https://github.com/pierluigiferrari/ssd_keras
# Copyright 2018 Pierluigi Ferrari, licensed under the Apache License, Version 2.0.
# Modifications Copyright 2024 Kerem Aras, also licensed under the Apache License, Version 2.0.
"""Transfer-learning SSD architecture with MobileNetV2 backbone (Figure 4 of the paper).

Replaces the custom six-block feature extractor with a frozen (or fine-tuned)
MobileNetV2 pretrained on ImageNet. The backbone is followed by a single
adapter Conv-BN-ReLU block and the same three-branch detection head as the
custom architecture, producing an identical (64, 12) output tensor.

The paper evaluates two variants (Section VI.B):
  - Frozen backbone: ``trainable_backbone=False`` (default)
  - Fine-tuned backbone: ``trainable_backbone=True``
"""

import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Concatenate,
    Conv2D,
    GaussianNoise,
    Input,
    ReLU,
    Reshape,
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

from aoi_pcb_ssd.model.grid_centers import GridCenters

_BN_MOMENTUM = 0.99
_KERNEL = (3, 3)
_GAUSSIAN_NOISE_STD = 0.1
_ADAPTER_FILTERS = 512  # maps MobileNetV2's 1280 output channels → 512


def build_transfer_model(
    image_size: tuple[int, int, int],
    n_classes: int,
    l2_regularization: float = 0.0,
    normalize_coords: bool = True,
    trainable_backbone: bool = False,
    weights: str | None = "imagenet",
    return_predictor_sizes: bool = False,
) -> Model | tuple[Model, np.ndarray]:
    """Build the MobileNetV2 transfer-learning model (Figure 4 of the paper).

    Args:
        image_size: ``(height, width, channels)`` of the input images.
        n_classes: Number of foreground classes (background is added internally).
        l2_regularization: L2 regularisation coefficient for the adapter and
            detection head conv layers. The MobileNetV2 backbone is unaffected.
        normalize_coords: Whether ``GridCenters`` should normalise anchor
            coordinates to [0, 1].
        trainable_backbone: If ``False`` (default), the MobileNetV2 weights are
            frozen. Set to ``True`` for the fine-tuned variant.
        weights: Backbone weight initialisation. Pass ``"imagenet"`` for the
            pretrained weights or ``None`` for random initialisation (useful
            for tests that should not trigger a network download).
        return_predictor_sizes: If ``True``, also return the spatial dimensions
            of the detection head feature map as a ``(1, 2)`` array.

    Returns:
        The Keras ``Model``, or a ``(model, predictor_sizes)`` tuple when
        ``return_predictor_sizes=True``.
    """
    n_boxes = 1
    n_classes += 1
    l2_reg = l2_regularization
    img_height, img_width, img_channels = image_size

    backbone = MobileNetV2(
        include_top=False,
        weights=weights,
        input_shape=(img_height, img_width, img_channels),
        pooling=None,
    )
    backbone.trainable = trainable_backbone

    x = Input(shape=(img_height, img_width, img_channels))
    x1 = preprocess_input(x)
    x1 = GaussianNoise(_GAUSSIAN_NOISE_STD)(x1)
    x1 = backbone(x1)

    # Adapter: maps MobileNetV2 output (8×8×1280) → 8×8×512
    x1 = Conv2D(
        _ADAPTER_FILTERS,
        _KERNEL,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(l2_reg),
        name="conv1",
    )(x1)
    x1 = BatchNormalization(axis=3, momentum=_BN_MOMENTUM, name="bn1")(x1)
    x1 = ReLU(name="relu1")(x1)

    # Three-branch detection head (identical to the custom architecture)
    classes1 = Conv2D(
        n_boxes * n_classes,
        _KERNEL,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(l2_reg),
        name="classes1",
    )(x1)
    offsets1 = Conv2D(
        n_boxes * 8,
        _KERNEL,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(l2_reg),
        name="offsets1",
    )(x1)
    centers1 = GridCenters(
        img_height, img_width, normalize_coords=normalize_coords, name="centers1"
    )(offsets1)

    classes1_reshaped = Reshape((-1, n_classes), name="classes1_reshape")(classes1)
    offsets1_reshaped = Reshape((-1, 8), name="offsets1_reshape")(offsets1)
    centers1_reshaped = Reshape((-1, 2), name="centers1_reshape")(centers1)

    classes_softmax = Activation("softmax", name="classes_softmax")(classes1_reshaped)

    predictions = Concatenate(axis=2, name="predictions")(
        [classes_softmax, offsets1_reshaped, centers1_reshaped]
    )

    model = Model(inputs=x, outputs=predictions)

    if return_predictor_sizes:
        predictor_sizes = np.array([classes1.shape[1:3]])
        return model, predictor_sizes
    return model
