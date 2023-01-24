import numpy as np
from keras.models import Model
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, BatchNormalization, ReLU, ELU, Reshape, Concatenate, Activation, GaussianNoise
from keras.regularizers import l2
import keras.backend as K

from custom_layers.GridCenters import GridCenters


def build_model(image_size,
                n_classes,
                l2_regularization=0.0,
                normalize_coords=True,
                subtract_mean=None,
                divide_by_stddev=None,
                swap_channels=False,
                return_predictor_sizes=False):

    n_predictor_layers = 1
    n_boxes = 1
    n_classes += 1
    l2_reg = l2_regularization
    img_height, img_width, img_channels = image_size[0], image_size[1], image_size[2]

    def identity_layer(tensor):
        return tensor

    def input_mean_normalization(tensor):
        return tensor - np.array(subtract_mean)

    def input_stddev_normalization(tensor):
        return tensor / np.array(divide_by_stddev)

    def input_channel_swap(tensor):
        if len(swap_channels) == 3:
            return K.stack(
                [tensor[..., swap_channels[0]], tensor[..., swap_channels[1]], tensor[..., swap_channels[2]]], axis=-1)
        elif len(swap_channels) == 4:
            return K.stack([tensor[..., swap_channels[0]], tensor[..., swap_channels[1]], tensor[..., swap_channels[2]],
                            tensor[..., swap_channels[3]]], axis=-1)

    ####################################################################
    # Build the network
    ####################################################################

    x = Input(shape=(img_height, img_width, img_channels))
    x = GaussianNoise(0.1)(x)

    # The following identity layer is only needed so that the subsequent lambda layers can be optional.
    x1 = Lambda(identity_layer, output_shape=(img_height, img_width, img_channels), name='identity_layer')(x)
    if not (subtract_mean is None):
        x1 = Lambda(input_mean_normalization, output_shape=(img_height, img_width, img_channels),
                    name='input_mean_normalization')(x1)
    if not (divide_by_stddev is None):
        x1 = Lambda(input_stddev_normalization, output_shape=(img_height, img_width, img_channels),
                    name='input_stddev_normalization')(x1)
    if swap_channels:
        x1 = Lambda(input_channel_swap, output_shape=(img_height, img_width, img_channels), name='input_channel_swap')(
            x1)

    # BASE NETWORK

    conv1 = Conv2D(32, (5, 5), strides=(1, 1), padding="same", kernel_initializer='he_normal',
                   kernel_regularizer=l2(l2_reg), name='conv1')(x1)
    conv1 = BatchNormalization(axis=3, momentum=0.99, name='bn1')(
        conv1)  # Tensorflow uses filter format [filter_height, filter_width, in_channels, out_channels], hence axis = 3
    conv1 = ReLU(name='relu1')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), name='pool1')(conv1)

    conv2 = Conv2D(64, (5, 5), strides=(1, 1), padding="same", kernel_initializer='he_normal',
                   kernel_regularizer=l2(l2_reg), name='conv2')(pool1)
    conv2 = BatchNormalization(axis=3, momentum=0.99, name='bn2')(conv2)
    conv2 = ReLU(name='relu2')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), name='pool2')(conv2)

    conv3 = Conv2D(128, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal',
                   kernel_regularizer=l2(l2_reg), name='conv3')(pool2)
    conv3 = BatchNormalization(axis=3, momentum=0.99, name='bn3')(conv3)
    conv3 = ReLU(name='relu3')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), name='pool3')(conv3)

    conv4 = Conv2D(256, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal',
                   kernel_regularizer=l2(l2_reg), name='conv4')(pool3)
    
    conv4 = BatchNormalization(axis=3, momentum=0.99, name='bn4')(conv4)
    conv4 = ReLU(name='relu4')(conv4)
    
    ######### ADDITIONAL BASE NETWORK LAYERS ###########################
    
    pool4 = MaxPooling2D(pool_size=(2, 2), name='pool4')(conv4)

    conv5 = Conv2D(512, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv5')(pool4)
    conv5 = BatchNormalization(axis=3, momentum=0.99, name='bn5')(conv5)
    conv5 = ReLU(name='relu5')(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2), name='pool5')(conv5)

    conv6 = Conv2D(512, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv6')(pool5)
    conv6 = BatchNormalization(axis=3, momentum=0.99, name='bn6')(conv6)
    conv6 = ReLU(name='relu6')(conv6)
    #pool6 = MaxPooling2D(pool_size=(2, 2), name='pool6')(conv6)

    #conv7 = Conv2D(32, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv7')(pool6)
    #conv7 = BatchNormalization(axis=3, momentum=0.99, name='bn7')(conv7)
    #conv7 = ELU(name='elu7')(conv7)
    
    ####################################################################

    classes7 = Conv2D(n_boxes * n_classes, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal',
                      kernel_regularizer=l2(l2_reg), name='classes7')(conv6)
    corners7 = Conv2D(n_boxes * 8, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal',
                    kernel_regularizer=l2(l2_reg), name='corners7')(conv6)
    centers7 = GridCenters(img_height, img_width, normalize_coords=normalize_coords, name='centers7')(corners7)

    classes7_reshaped = Reshape((-1, n_classes), name='classes7_reshaped')(classes7)
    corners7_reshaped = Reshape((-1, 8), name='corners7_reshaped')(corners7)
    centers7_reshaped = Reshape((-1, 2), name='centers7_reshaped')(centers7)

    classes_softmax = Activation('softmax', name='classes_softmax')(classes7_reshaped)

    predictions = Concatenate(axis=2, name='predictions')([classes_softmax, corners7_reshaped, centers7_reshaped])

    model = Model(inputs=x, outputs=predictions)

    if return_predictor_sizes:
        # The spatial dimensions are the same for the `classes` and `boxes` predictor layers.
        predictor_sizes = np.array([classes4._keras_shape[1:3]])
        return model, predictor_sizes
    else:
        return model
