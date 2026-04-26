import wandb
from wandb.keras import WandbCallback
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from math import ceil
from sklearn.model_selection import train_test_split

from models.ssd_custom_wandb import build_model
from loss_function.custom_loss import AOILoss
from performance_metrics.metrics_classification import f1, precision, recall
from performance_metrics.metrics_regression import mae, root_mse, mse

from input_encoder_decoder.input_encoder_new import SSDInputEncoder
from input_encoder_decoder.data_generator import DataGenerator

img_height = 256  # Height of the input images
img_width = 256  # Width of the input images
img_channels = 3  # Number of color channels of the input images
intensity_mean = 127.5  # Set this to your preference (maybe `None`). The current settings transform the input pixel values to the interval `[-1,1]`.
intensity_range = 127.5  # Set this to your preference (maybe `None`). The current settings transform the input pixel values to the interval `[-1,1]`.
n_classes = 1  # Number of positive classes
normalize_coords = True  # Whether the model is supposed to use coordinates relative to the image size
batch_size = 16


def create_model(params):
    K.clear_session()

    model = build_model(image_size=(img_height, img_width, img_channels),
                        n_classes=n_classes,
                        l2_regularization=params['l2_reg'],
                        normalize_coords=normalize_coords,
                        subtract_mean=intensity_mean,
                        divide_by_stddev=intensity_range,
                        cfg=params
                        )

    aoi_loss = AOILoss(neg_pos_ratio=params['neg_pos_ratio'], alpha=params['alpha'])

    model.compile(optimizer=Adam(learning_rate=params['initial_lr'], epsilon=1e-8), loss=aoi_loss.compute_loss,
                  metrics=[precision, recall, f1, mae, root_mse, mse])

    return model


def prepare_data():
    encoder = SSDInputEncoder(img_height,
                              img_width,
                              n_classes,
                              predictor_sizes=[(8, 8)],
                              normalize_coords=True,
                              background_id=0)

    generator = DataGenerator(parent_dir='/home/kerem/AOI/Datasets/train_pcb_crops', encoder=encoder, augmentation=True,
                              probability=0.1)

    X, y = generator.get_data()

    X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

    return X_tr, X_ts, y_tr, y_ts


def train():
    with wandb.init():
        config = wandb.config

        model = create_model(config)

        model.summary()

        wandb_callback = WandbCallback(verbose=0, save_model=False)

        reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                      factor=0.2,
                                      patience=3,
                                      min_delta=0.001,
                                      cooldown=0,
                                      min_lr=0.000001)

        early_stopping = EarlyStopping(monitor='val_loss',
                                       min_delta=0.001,
                                       patience=7,
                                       verbose=1)

        cbacks = [wandb_callback, reduce_lr, early_stopping]

        history = model.fit(X_train,
                            y_train,
                            callbacks=cbacks,
                            batch_size=batch_size,
                            epochs=config['epochs'],
                            validation_data=(X_test, y_test),
                            validation_steps=ceil(X_test.shape[0] // batch_size),
                            verbose=2
                            )

        # wandb.log({'loss': history.history['loss'],
        #           'val_loss': history.history['val_loss'],
        #           'f1': history.history['f1'],
        #           'val_f1': history.history['val_f1'],
        #           'r2': history.history['r2'],
        #           'val_r2': history.history['val_r2']})

        del model
        K.clear_session()


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_data()

    print('Train dataset: ', X_train.shape)
    print('Test dataset: ', X_test.shape)
    print('Train labels: ', y_train.shape)
    print('Test labels: ', y_test.shape)

    train()
