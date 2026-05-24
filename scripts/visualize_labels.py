"""Overlay ground-truth IC corner circles on generated dataset images.

Usage:
    python scripts/visualize_labels.py \\
        --data-dir  datasets/train \\
        --n-images  20 \\
        --output-dir visualizations/

Reads ``labels.csv`` from ``--data-dir``, draws a blue circle at each of the
four predicted corner points and at the IC centre, then saves the annotated
images to ``--output-dir`` rather than displaying them interactively — making
it safe to run in headless CI or remote environments.

Use this to sanity-check that ``generate_dataset.py`` produced correct labels
before spending time on training.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise ground-truth labels on dataset images.")
    parser.add_argument("--data-dir",   required=True, help="Directory containing images and labels.csv.")
    parser.add_argument("--n-images",   type=int, default=10, help="Number of images to annotate.")
    parser.add_argument("--output-dir", required=True, help="Directory to write annotated images.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(data_dir / "labels.csv")
    img_names = labels["frame"].unique()[: args.n_images]

    for name in img_names:
        img_path = data_dir / name
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: could not read {img_path}, skipping.")
            continue

        rows = labels[labels["frame"] == name]
        # Columns after 'frame': tl_x, tl_y, tr_x, tr_y, bl_x, bl_y, br_x, br_y, cx, cy, class_id
        coord_cols = ["tl_x", "tl_y", "tr_x", "tr_y", "bl_x", "bl_y", "br_x", "br_y", "cx", "cy"]

        for _, row in rows.iterrows():
            coords = row[coord_cols].to_numpy(dtype=int).reshape(-1, 2)
            for pt in coords:
                cv2.circle(img, tuple(pt), 4, (255, 0, 0), 2)

        out_path = out_dir / name
        cv2.imwrite(str(out_path), img)

    print(f"Annotated {len(img_names)} images → {out_dir}/")


if __name__ == "__main__":
    main()
