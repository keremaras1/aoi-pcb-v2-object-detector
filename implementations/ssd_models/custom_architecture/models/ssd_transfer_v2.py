import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D, BatchNormalization, ReLU, ELU, Reshape, \
    Concatenate, Activation, GaussianNoise
from keras_cv.models import ResNetV2Backbone, YOLOV8Backbone, MobileNetV3Backbone, EfficientNetV2Backbone
from tensorflow.keras.regularizers import l2

from custom_layers.GridCenters import GridCenters


def tl_build_model(image_size,
                   n_classes,
                   variant,
                   l2_regularization=0.0,
                   normalize_coords=True,
                   return_predictor_sizes=False):
    n_predictor_layers = 1
    n_boxes = 1
    n_classes += 1
    l2_reg = l2_regularization
    img_height, img_width, img_channels = image_size[0], image_size[1], image_size[2]

    ####################################################################
    # Build the network
    ####################################################################

    assert variant is not None, "Missing TL Backbone Variant"
    assert variant in ['mobilenet', 'resnet', 'yolo', 'efficientnet'], "Invalid TL Backbone Variant"

    if variant == 'mobilenet':
        backbone = MobileNetV3Backbone.from_preset("mobilenet_v3_large")
    elif variant == 'resnet':
        backbone = ResNetV2Backbone.from_preset("resnet18_v2")
    elif variant == 'yolo':
        backbone = YOLOV8Backbone.from_preset("yolo_v8_m_backbone")
    elif variant == 'efficientnet':
        backbone = EfficientNetV2Backbone.from_preset("efficientnetv2_b0")
    backbone.trainable = True

    x = Input(shape=(img_height, img_width, img_channels))

    x1 = GaussianNoise(0.1)(x)
    x1 = backbone(x1)
    # BASE NETWORK

    conv_loc = Conv2D(128, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal',
                      kernel_regularizer=l2(l2_reg), name='conv_loc')(x1)
    conv_loc = BatchNormalization(axis=3, momentum=0.99, name='bn_loc')(conv_loc)
    conv_loc = ReLU(name='relu_loc')(conv_loc)

    conv_loc2 = Conv2D(64, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal',
                       kernel_regularizer=l2(l2_reg), name='conv_loc2')(conv_loc)
    conv_loc2 = BatchNormalization(axis=3, momentum=0.99, name='bn_loc2')(conv_loc2)
    conv_loc2 = ReLU(name='relu_loc2')(conv_loc2)

    classes = Conv2D(n_boxes * n_classes, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='classes')(x1)
    corners = Conv2D(n_boxes * 8, (1, 1), strides=(1, 1), padding="same", kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='corners')(conv_loc2)
    centers = GridCenters(img_height, img_width, normalize_coords=normalize_coords, name='centers')(corners)

    classes_reshaped = Reshape((-1, n_classes), name='classes_reshaped')(classes)
    corners_reshaped = Reshape((-1, 8), name='corners_reshaped')(corners)
    centers_reshaped = Reshape((-1, 2), name='centers_reshaped')(centers)

    classes_softmax = Activation('softmax', name='classes_softmax')(classes_reshaped)

    predictions = Concatenate(axis=2, name='predictions')([classes_softmax, corners_reshaped, centers_reshaped])

    model = Model(inputs=x, outputs=predictions)

    if return_predictor_sizes:
        # The spatial dimensions are the same for the `classes` and `boxes` predictor layers.
        predictor_sizes = np.array([classes._keras_shape[1:3]])
        return model, predictor_sizes
    else:
        return model
