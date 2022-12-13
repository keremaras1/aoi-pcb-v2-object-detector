import numpy as np
from keras.models import Model
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, BatchNormalization, ELU, Reshape, Concatenate, Activation, GaussianNoise
from keras.applications import MobileNetV2, mobilenet_v2
from keras.regularizers import l2
import keras.backend as K

from custom_layers.GridCenters import GridCenters


def resnet_build_model(image_size,
                        n_classes,
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
    
    backbone = MobileNetV2(include_top=False,
                        weights='imagenet',
                        input_shape=(img_height, img_width, img_channels),
                        pooling=None)
    backbone.trainable = False

    x = Input(shape=(img_height, img_width, img_channels))
    x = GaussianNoise(0.1)(x)
    
    x = mobilenet_v2.preprocess_input(x)
    x1 = backbone(x)
    # BASE NETWORK

    conv1 = Conv2D(32, (5, 5), strides=(1, 1), padding="same", kernel_initializer='he_normal',
                   kernel_regularizer=l2(l2_reg), name='conv1')(x1)
    conv1 = BatchNormalization(axis=3, momentum=0.99, name='bn1')(
        conv1)  # Tensorflow uses filter format [filter_height, filter_width, in_channels, out_channels], hence axis = 3
    conv1 = ELU(name='elu1')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), name='pool1')(conv1)

    conv2 = Conv2D(48, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal',
                   kernel_regularizer=l2(l2_reg), name='conv2')(pool1)
    conv2 = BatchNormalization(axis=3, momentum=0.99, name='bn2')(conv2)
    conv2 = ELU(name='elu2')(conv2)
    #pool2 = MaxPooling2D(pool_size=(2, 2), name='pool2')(conv2)

    #conv3 = Conv2D(64, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal',
    #               kernel_regularizer=l2(l2_reg), name='conv3')(pool2)
    #conv3 = BatchNormalization(axis=3, momentum=0.99, name='bn3')(conv3)
    #conv3 = ELU(name='elu3')(conv3)
    #pool3 = MaxPooling2D(pool_size=(2, 2), name='pool3')(conv3)

    #conv4 = Conv2D(64, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal',
    #               kernel_regularizer=l2(l2_reg), name='conv4')(pool3)
    
    # conv4 = L2Normalization(gamma_init=20, name='conv4')(conv4)
    
    
    #conv4 = BatchNormalization(axis=3, momentum=0.99, name='bn4')(conv4)
    #conv4 = ELU(name='elu4')(conv4)
    
    ######### ADDITIONAL BASE NETWORK LAYERS ###########################
    
    #pool4 = MaxPooling2D(pool_size=(2, 2), name='pool4')(conv4)

    #conv5 = Conv2D(48, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv5')(pool4)
    #conv5 = BatchNormalization(axis=3, momentum=0.99, name='bn5')(conv5)
    #conv5 = ELU(name='elu5')(conv5)
    #pool5 = MaxPooling2D(pool_size=(2, 2), name='pool5')(conv5)

    #conv6 = Conv2D(48, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv6')(pool5)
    #conv6 = BatchNormalization(axis=3, momentum=0.99, name='bn6')(conv6)
    #conv6 = ELU(name='elu6')(conv6)
    #pool6 = MaxPooling2D(pool_size=(2, 2), name='pool6')(conv6)

    #conv7 = Conv2D(32, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv7')(pool6)
    #conv7 = BatchNormalization(axis=3, momentum=0.99, name='bn7')(conv7)
    #conv7 = ELU(name='elu7')(conv7)
    
    ####################################################################

    classes4 = Conv2D(n_boxes * n_classes, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal',
                      kernel_regularizer=l2(l2_reg), name='classes4')(conv2)
    boxes4 = Conv2D(n_boxes * 8, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal',
                    kernel_regularizer=l2(l2_reg), name='boxes4')(conv2)
    centers4 = GridCenters(img_height, img_width, normalize_coords=normalize_coords, name='centers4')(boxes4)

    classes4_reshaped = Reshape((-1, n_classes), name='classes4_reshape')(classes4)
    boxes4_reshaped = Reshape((-1, 8), name='boxes4_reshape')(boxes4)
    centers4_reshaped = Reshape((-1, 2), name='anchors4_reshape')(centers4)

    classes_softmax = Activation('softmax', name='classes_softmax')(classes4_reshaped)

    predictions = Concatenate(axis=2, name='predictions')([classes_softmax, boxes4_reshaped, centers4_reshaped])

    model = Model(inputs=x, outputs=predictions)

    if return_predictor_sizes:
        # The spatial dimensions are the same for the `classes` and `boxes` predictor layers.
        predictor_sizes = np.array([classes4._keras_shape[1:3]])
        return model, predictor_sizes
    else:
        return model
