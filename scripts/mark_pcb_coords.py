"""GUI tool: click IC corner boxes on a PCB image and save them as a CSV.

Usage:
    python scripts/mark_pcb_coords.py --img-path path/to/pcb.jpg

Left-click each IC twice — once for the top-left corner and once for the
bottom-right corner. Every two clicks define one bounding box. Close the
window when all ICs are marked.

Output:
    <image_name>_ic_corners.csv next to the source image, with columns
    ``x_min, y_min, x_max, y_max`` (one row per IC).
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

_clicks: list[tuple[int, int]] = []


def _on_click(event, x: int, y: int, flags, param) -> None:
    if event == cv2.EVENT_LBUTTONDOWN:
        _clicks.append((x, y))
        cv2.circle(param, (x, y), 5, (255, 0, 0), 2)
        cv2.imshow("Mark IC corners", param)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mark IC corner boxes on a PCB image.")
    parser.add_argument("--img-path", required=True, help="Path to the PCB image file.")
    args = parser.parse_args()

    img_path = Path(args.img_path)
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")

    cv2.imshow("Mark IC corners", img)
    cv2.setMouseCallback("Mark IC corners", _on_click, img)
    print("Left-click each IC: top-left corner then bottom-right corner.")
    print("Close the window when done.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(_clicks) % 2 != 0:
        print(f"Warning: {len(_clicks)} clicks recorded — expected an even number. Last click ignored.")
        _clicks.pop()

    corners = np.array(_clicks).reshape(-1, 4)
    out_path = img_path.parent / f"{img_path.stem}_ic_corners.csv"
    pd.DataFrame(corners, columns=["x_min", "y_min", "x_max", "y_max"]).to_csv(out_path, index=False)
    print(f"Saved {len(corners)} IC corner boxes → {out_path}")


if __name__ == "__main__":
    main()
