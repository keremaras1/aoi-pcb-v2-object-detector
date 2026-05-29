"""CLI script for training the AOI-PCB-SSD model.

Loads data, builds the selected model architecture, and runs the training loop
with early stopping and learning rate scheduling as configured in config.json.
The trained model and training logs are saved to the output directory.

Usage::

    python scripts/train.py --architecture custom
    python scripts/train.py --architecture transfer
    python scripts/train.py --architecture custom --config config.json --output-dir experiments/run_1
"""

import argparse
import shutil
from datetime import datetime
from pathlib import Path
import sys

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aoi_pcb_ssd.config_loader import Config
from aoi_pcb_ssd.data.data_generator import DataGenerator
from aoi_pcb_ssd.encoding.input_encoder import SSDInputEncoder
from aoi_pcb_ssd.model.loss import AOILoss
from aoi_pcb_ssd.model.metrics import class_mAP, mae
from aoi_pcb_ssd.model.ssd_custom import build_custom_model
from aoi_pcb_ssd.model.ssd_transfer import build_transfer_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the AOI-PCB-SSD model.")
    parser.add_argument(
        "--architecture", choices=["custom", "transfer"], required=True,
        help="Model architecture: 'custom' (Figure 3) or 'transfer' (Figure 4).",
    )
    parser.add_argument(
        "--config", default="config.json",
        help="Path to the JSON configuration file (default: config.json).",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory to save the trained model and logs. Defaults to experiments/run_<timestamp>.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config(args.config)
    m = cfg.model
    t = cfg.training

    # Resolve output directory
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("experiments") / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, output_dir / "config.json")
    print(f"Output directory: {output_dir}")

    # Report available GPUs
    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPUs available: {len(gpus)}")
    for gpu in gpus:
        print(f"  {gpu}")

    # --- Data loading ---
    encoder = SSDInputEncoder(
        img_height=m.img_height,
        img_width=m.img_width,
        n_classes=m.n_classes,
        predictor_sizes=m.predictor_sizes,
        normalize_coords=m.normalize_coords,
    )

    print(f"Loading data from {cfg.generator.train_data.crop_save_dir} ...")
    generator = DataGenerator(
        parent_dir=cfg.generator.train_data.crop_save_dir,
        encoder=encoder,
        augmentation=cfg.augmentation.enabled,
        probability=cfg.augmentation.probability,
    )
    X, y = generator.get_data()

    # Split matches the paper's training setup: 80/20, shuffled, random_state=42
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=t.val_split, shuffle=True, random_state=t.random_seed,
    )
    print(f"Train: {X_train.shape}  Val: {X_val.shape}")

    # --- Model setup ---
    image_size = (m.img_height, m.img_width, m.img_channels)
    if args.architecture == "custom":
        model = build_custom_model(
            image_size=image_size,
            n_classes=m.n_classes,
            l2_regularization=m.l2_regularization,
            normalize_coords=m.normalize_coords,
            subtract_mean=m.subtract_mean,
            divide_by_stddev=m.divide_by_stddev,
        )
    else:
        model = build_transfer_model(
            image_size=image_size,
            n_classes=m.n_classes,
            l2_regularization=m.l2_regularization,
            normalize_coords=m.normalize_coords,
        )
    model.summary()

    aoi_loss = AOILoss(**cfg.get_init_kwargs("training.loss"))
    model.compile(
        optimizer=Adam(**cfg.get_init_kwargs("training.optimizer")),
        loss=aoi_loss.compute_loss,
        metrics=[class_mAP, mae],
    )

    # --- Callbacks ---
    model_path = output_dir / "model.keras"
    training_callbacks = [
        EarlyStopping(**cfg.get_init_kwargs("training.early_stopping")),
        ReduceLROnPlateau(**cfg.get_init_kwargs("training.lr_schedule")),
        CSVLogger(str(output_dir / f"training_{model_path.stem}.csv")),
    ]

    # --- Training ---
    history = model.fit(
        X_train, y_train,
        batch_size=t.batch_size,
        epochs=t.epochs,
        validation_data=(X_val, y_val),
        callbacks=training_callbacks,
        verbose=2,
    )

    # --- Save model ---
    model.save(str(model_path))
    print(f"Model saved to {model_path}")

    # --- Summary ---
    n_epochs = len(history.history["loss"])
    best_val_loss = min(history.history["val_loss"])
    best_epoch = history.history["val_loss"].index(best_val_loss) + 1
    print(f"Trained for {n_epochs} epochs.")
    print(f"Best val_loss: {best_val_loss:.6f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()
