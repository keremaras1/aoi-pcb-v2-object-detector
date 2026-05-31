"""CLI script for evaluating a trained AOI-PCB-SSD model.

Loads a saved model, reconstructs the same 80/20 validation split used during
training, runs evaluation, and optionally saves visualisation images showing
ground-truth (red) and predicted (blue) IC corner circles.

Usage::

    python scripts/evaluate.py
    python scripts/evaluate.py --save-visuals
    python scripts/evaluate.py --model-path experiments/run_20240501_120000/model.keras
    python scripts/evaluate.py --model-path experiments/run_20240501_120000/model.keras \\
        --config path/to/other_config.json
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aoi_pcb_ssd.config_loader import Config
from aoi_pcb_ssd.data.data_generator import DataGenerator
from aoi_pcb_ssd.encoding.input_encoder import SSDInputEncoder
from aoi_pcb_ssd.encoding.output_decoder import decode_detections
from aoi_pcb_ssd.model.grid_centers import GridCenters
from aoi_pcb_ssd.model.loss import AOILoss
from aoi_pcb_ssd.model.metrics import class_mAP, mae

_CLASSES = ["background", "ic"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained AOI-PCB-SSD model.")
    parser.add_argument(
        "--model-path",
        default=None,
        help=(
            "Path to the saved model file (.keras). "
            "Defaults to the most recently modified run in experiments/."
        ),
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Path to a JSON configuration file. "
            "Defaults to config.json inside the run directory."
        ),
    )
    parser.add_argument(
        "--save-visuals",
        action="store_true",
        help="Save prediction overlay images to <run-dir>/visuals/.",
    )
    parser.add_argument(
        "--n-visuals",
        type=int,
        default=15,
        help="Number of prediction overlay images to save (default: 15).",
    )
    return parser.parse_args()


def draw_keypoints(
    image: np.ndarray,
    actual_dets: list[np.ndarray],
    predicted_dets: list[np.ndarray],
) -> np.ndarray:
    """Draw ground-truth (red) and predicted (blue) IC corner circles onto an image.

    Args:
        image: HxWx3 uint8 RGB image array.
        actual_dets: Decoded ground-truth detections for this image.
        predicted_dets: Decoded model predictions for this image.

    Returns:
        BGR image array with corner circles and confidence labels drawn on it.
    """
    img = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)

    for det in actual_dets:
        corners = det[2:].reshape(-1, 2).astype(int)
        for pt in corners:
            cv2.circle(img, tuple(pt), 4, (0, 0, 255), 2)  # red — ground truth

    for det in predicted_dets:
        corners = det[2:].reshape(-1, 2).astype(int)
        for pt in corners:
            cv2.circle(img, tuple(pt), 4, (255, 0, 0), 2)  # blue — prediction
        cx = int((corners[0, 0] + corners[3, 0]) / 2)
        cy = int((corners[0, 1] + corners[3, 1]) / 2)
        label = f"{_CLASSES[int(det[0])]}: {det[1]:.2f}"
        cv2.putText(
            img, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA
        )

    return img


def main() -> None:
    args = parse_args()

    # Resolve model path — default to the most recently modified run
    if args.model_path:
        model_path = Path(args.model_path)
    else:
        runs = sorted(
            Path("experiments").glob("*/model.keras"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not runs:
            raise FileNotFoundError("No trained model found in experiments/. Run train.py first.")
        model_path = runs[0]
    print(f"Model: {model_path}")

    config_path = args.config or str(model_path.parent / "config.json")
    cfg = Config(config_path)
    print(f"Config: {config_path}")
    m = cfg.model
    t = cfg.training

    # --- Data loading ---
    # Reconstruct the same 80/20 validation split used during training so
    # evaluation is always on the held-out set, not the training data.
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
        augmentation=False,
    )
    X, y = generator.get_data()

    _, X_val, _, y_val = train_test_split(
        X,
        y,
        test_size=t.val_split,
        shuffle=True,
        random_state=t.random_seed,
    )
    print(f"Validation data shape: {X_val.shape}, Labels shape: {y_val.shape}")

    # --- Load model ---
    # compile=False skips Keras's attempt to deserialise the saved compile config,
    # which fails for custom loss methods. We re-compile immediately after.
    aoi_loss = AOILoss(**cfg.get_init_kwargs("training.loss"))
    model = tf.keras.models.load_model(
        str(model_path),
        compile=False,
        custom_objects={"GridCenters": GridCenters},
    )
    model.compile(
        optimizer=Adam(**cfg.get_init_kwargs("training.optimizer")),
        loss=aoi_loss.compute_loss,
        metrics=[class_mAP, mae],
    )
    model.summary()

    # --- Evaluate ---
    results = model.evaluate(X_val, y_val, batch_size=t.batch_size, verbose=1)
    print("\nEvaluation results:")
    for name, value in zip(model.metrics_names, results):
        print(f"  {name}: {value:.6f}")

    # --- Predict ---
    predictions = model.predict(X_val, batch_size=t.batch_size, verbose=0)
    print(f"\nPredictions shape: {predictions.shape}")

    # --- Save visualisations ---
    if args.save_visuals:
        visuals_dir = model_path.parent / "visuals"
        visuals_dir.mkdir(parents=True, exist_ok=True)

        n = min(args.n_visuals, len(X_val))
        decode_kwargs = dict(
            normalize_coords=m.normalize_coords,
            img_height=m.img_height,
            img_width=m.img_width,
        )
        actual_all = decode_detections(y_val[:n], **decode_kwargs)
        pred_all = decode_detections(predictions[:n], **decode_kwargs)

        for idx in range(n):
            img = draw_keypoints(X_val[idx], actual_all[idx], pred_all[idx])
            cv2.imwrite(str(visuals_dir / f"pred_{idx:04d}.jpg"), img)

        print(f"Saved {n} visualisation images to {visuals_dir}")


if __name__ == "__main__":
    main()
