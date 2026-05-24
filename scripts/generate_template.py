"""Generate template assets from a PCB image and its IC corner coordinates.

Usage:
    python scripts/generate_template.py \\
        --pcb-image  path/to/pcb.jpg \\
        --corners-csv path/to/pcb_ic_corners.csv \\
        --output-dir  templates/pcb_template_N/template_object_1/

Run this once per PCB board design after marking IC positions with
``mark_pcb_coords.py``. Outputs:

  ``output-dir/cropped_ic_<i>.jpg``  — one cutout per IC
  ``output-dir/background_pcb.jpg``  — PCB with IC regions filled
  ``output-dir/ic_centroids.csv``    — centroid + size labels

These files become the input to ``generate_dataset.py``.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def _read_corners(path: Path) -> np.ndarray:
    return pd.read_csv(path).to_numpy()


def _corners_to_centroids(corners: np.ndarray) -> np.ndarray:
    cx = (corners[:, 0] + corners[:, 2]) / 2
    cy = (corners[:, 1] + corners[:, 3]) / 2
    w = corners[:, 2] - corners[:, 0]
    h = corners[:, 3] - corners[:, 1]
    return np.column_stack([cx, cy, w, h])


def _dominant_color(img: np.ndarray) -> np.ndarray:
    """Return the single most representative colour via 1-cluster k-means."""
    data = np.reshape(img, (-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, _, centers = cv2.kmeans(data, 1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    return centers[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PCB template assets.")
    parser.add_argument("--pcb-image",   required=True, help="Source PCB image (.jpg).")
    parser.add_argument("--corners-csv", required=True, help="IC corner boxes CSV (x_min,y_min,x_max,y_max).")
    parser.add_argument("--output-dir",  required=True, help="Directory to write template assets.")
    args = parser.parse_args()

    pcb_path = Path(args.pcb_image)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(pcb_path))
    if img is None:
        raise FileNotFoundError(f"Could not read PCB image: {pcb_path}")

    corners = _read_corners(Path(args.corners_csv))
    centroids = _corners_to_centroids(corners)
    dominant = _dominant_color(img)

    # IC cutouts
    labels = []
    for i, c in enumerate(corners):
        cutout = img[c[1]:c[3], c[0]:c[2], :]
        name = f"cropped_ic_{i}.jpg"
        cv2.imwrite(str(out_dir / name), cutout)
        labels.append([name, *map(float, centroids[i])])
    print(f"Saved {len(labels)} IC cutouts → {out_dir}/")

    # Background (IC regions filled with dominant colour)
    background = img.copy()
    for c in corners:
        background[c[1]:c[3], c[0]:c[2], :] = dominant
    cv2.imwrite(str(out_dir / "background_pcb.jpg"), background)
    print(f"Saved background_pcb.jpg")

    # Centroid labels
    pd.DataFrame(
        labels, columns=["ic_frame", "cx", "cy", "w", "h"]
    ).to_csv(out_dir / "ic_centroids.csv", index=False)
    print(f"Saved ic_centroids.csv")


if __name__ == "__main__":
    main()
