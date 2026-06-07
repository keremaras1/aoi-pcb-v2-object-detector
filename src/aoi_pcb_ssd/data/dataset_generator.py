# SPDX-License-Identifier: Apache-2.0
"""Synthetic PCB dataset generator.

Composites IC cutouts onto PCB template backgrounds with random rotation and
Gaussian-distributed placement offsets, then crops square patches that are
guaranteed to contain at least one IC. This is the data generation pipeline
described in Section III.B of the paper.
"""

import math
import random
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from aoi_pcb_ssd.data.template import Template

_LABEL_COLUMNS = [
    "frame",
    "tl_x",
    "tl_y",
    "tr_x",
    "tr_y",
    "bl_x",
    "bl_y",
    "br_x",
    "br_y",
    "cx",
    "cy",
    "class_id",
]
_CROP_RATIOS = np.linspace(0.1, 0.9, num=5, endpoint=True)


def _centroids_to_corners(centroids: np.ndarray) -> np.ndarray:
    """Convert (cx, cy, w, h) centroid rows to eight corner-point rows.

    Args:
        centroids: Array of shape (N, 4) with columns cx, cy, w, h.

    Returns:
        Integer array of shape (N, 10) with columns
        tl_x, tl_y, tr_x, tr_y, bl_x, bl_y, br_x, br_y, cx, cy.
    """
    x_min = centroids[:, 0] - centroids[:, 2] / 2
    x_max = centroids[:, 0] + centroids[:, 2] / 2
    y_min = centroids[:, 1] - centroids[:, 3] / 2
    y_max = centroids[:, 1] + centroids[:, 3] / 2
    corners = np.array(
        [x_min, y_min, x_max, y_min, x_min, y_max, x_max, y_max, centroids[:, 0], centroids[:, 1]]
    ).T
    return np.rint(corners).astype(int)


def _rotate(
    origin: tuple[float, float], point: tuple[float, float], angle: float
) -> tuple[float, float]:
    """Rotate a point counterclockwise around an origin by angle (radians)."""
    ox, oy = origin
    px, py = point
    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


def _rotate_coords(coords: np.ndarray, angle: float) -> np.ndarray:
    """Rotate corner coordinates around the centre point by angle (degrees)."""
    center = coords[-1]
    for i, corner in enumerate(coords[:-1]):
        coords[i] = _rotate(center, corner, math.radians(-angle))
    return np.reshape(coords, (-1,))


def _get_new_corners(
    corners: np.ndarray, offset_x: int, offset_y: int, rotation: float
) -> np.ndarray:
    """Apply translation and rotation to a row of corner coordinates."""
    corners_mat = np.reshape(corners, (-1, 2))
    corners_mat += np.array([offset_x, offset_y])
    return _rotate_coords(corners_mat, rotation)


def _check_ic_in_crop(bounds: np.ndarray, coords: np.ndarray) -> list[int]:
    """Return indices of ICs whose all corner points fall within the crop bounds.

    Args:
        bounds: Array [xmin, ymin, xmax, ymax] defining the crop rectangle.
        coords: Array of shape (N, 10) with corner and centre coordinates.

    Returns:
        List of row indices where all x and y coordinates are within bounds.
    """
    x_coords = coords[:, [0, 2, 4, 6, 8]]
    y_coords = coords[:, [1, 3, 5, 7, 9]]
    idx = []
    for i in range(coords.shape[0]):
        in_x = np.logical_and(x_coords[i] >= bounds[0], x_coords[i] < bounds[2])
        in_y = np.logical_and(y_coords[i] >= bounds[1], y_coords[i] < bounds[3])
        if np.all(in_x) and np.all(in_y):
            idx.append(i)
    return idx


def _shift_coords_for_crop(crop_xmin: int, crop_ymin: int, corners: np.ndarray) -> np.ndarray:
    """Subtract crop origin from corner coordinates so they are crop-relative."""
    corners[:, [0, 2, 4, 6, 8]] -= crop_xmin
    corners[:, [1, 3, 5, 7, 9]] -= crop_ymin
    return corners


class PCBDatasetGenerator:
    """Generates synthetic 256×256 PCB training images from template assets.

    Two-phase generation (matching the original pipeline):
    1. ``generate()`` — composites full-size images from templates and saves
       them together with a ``labels.csv`` of eight-point corner annotations.
    2. ``generate(version="cropped")`` — draws random square crops from the
       full-size images, discarding any crop that contains no IC, and saves
       the cropped images with crop-relative labels.

    Args:
        templates_dir: Root directory containing ``pcb_template_N/`` subdirs,
            each with one or more ``template_object_*/`` sub-subdirs.
        save_dir: Directory to write full-size generated images and labels.
        crop_save_dir: Directory to write cropped images and crop labels.
        rotation_range: IC rotation bound in degrees. Rotations are drawn from a
            zero-mean Gaussian with ``rotation_range`` as the ~2σ bound (about
            95% of draws fall within ``[-rotation_range, rotation_range]``).
        img_size: Side length (pixels) of output crops after resizing.
        dataset_size: Number of images to generate per phase.
        placement_offset_x: Horizontal IC placement-offset bound in pixels, as
            the ~2σ bound of a zero-mean Gaussian (see ``rotation_range``).
        placement_offset_y: Vertical IC placement-offset bound in pixels, as the
            ~2σ bound of a zero-mean Gaussian (see ``rotation_range``).
        seed: Random seed for reproducibility. Passed to both :mod:`random`
            and :mod:`numpy.random` at the start of each ``generate()`` call.
    """

    def __init__(
        self,
        templates_dir: str | Path,
        save_dir: str | Path,
        crop_save_dir: str | Path,
        rotation_range: int,
        img_size: int,
        dataset_size: int,
        placement_offset_x: int,
        placement_offset_y: int,
        seed: int = 42,
    ) -> None:
        self.templates_path = Path(templates_dir)
        self.save_path = Path(save_dir)
        self.crop_save_path = Path(crop_save_dir)
        self.rotation_range = rotation_range
        self.img_size = img_size
        self.dataset_size = dataset_size
        self.offsets = (placement_offset_x, placement_offset_y)
        self.seed = seed

        self.template_objects: list[Template] = []
        self._load_templates()

        self.labels = pd.DataFrame(columns=_LABEL_COLUMNS)
        self.crop_labels = pd.DataFrame(columns=_LABEL_COLUMNS)

    def _load_templates(self) -> None:
        for template_dir in sorted(self.templates_path.iterdir()):
            if not template_dir.is_dir():
                continue
            for subdir in sorted(template_dir.iterdir()):
                if subdir.is_dir():
                    self.template_objects.append(Template(subdir))

    def _get_rotation(self) -> float:
        # Zero-mean Gaussian with rotation_range as the ~2-sigma bound, so about
        # 95% of rotations fall within [-rotation_range, rotation_range] degrees.
        raw = np.random.normal(0, 1)
        return self.rotation_range * raw / 2

    def _get_ic_placement(
        self, cx: float, cy: float, rotated_ic: Image.Image
    ) -> tuple[int, int, int, int]:
        """Return (paste_x, paste_y, offset_x, offset_y) for one IC."""
        width, height = rotated_ic.size
        # Zero-mean Gaussian per axis with the placement offset as the ~2-sigma
        # bound, so about 95% of offsets fall within [-offset, offset] pixels.
        gauss_x = np.random.normal(0, 1)
        gauss_y = np.random.normal(0, 1)
        offset_x = round(self.offsets[0] * gauss_x / 2)
        offset_y = round(self.offsets[1] * gauss_y / 2)
        paste_x = int(cx - width / 2) + offset_x
        paste_y = int(cy - height / 2) + offset_y
        return paste_x, paste_y, offset_x, offset_y

    def _generate_one_image(self, image_name: str) -> None:
        template = random.choice(self.template_objects)
        background = template.get_background().copy()
        ic_cutouts = template.get_ic_cutouts()
        centroids = template.get_labels_df().loc[:, ["cx", "cy", "w", "h"]].to_numpy()
        corners = _centroids_to_corners(centroids)

        for i, ic in enumerate(ic_cutouts):
            rotation = self._get_rotation()
            rgba = ic.convert("RGBA")
            rotated = rgba.rotate(rotation, expand=True)
            paste_x, paste_y, off_x, off_y = self._get_ic_placement(
                centroids[i, 0], centroids[i, 1], rotated
            )
            corners[i] = _get_new_corners(corners[i], off_x, off_y, rotation)
            background.paste(rotated, (paste_x, paste_y), rotated)

        self.save_path.mkdir(parents=True, exist_ok=True)
        background.save(self.save_path / image_name)

        frame_col = np.full(corners.shape[0], image_name).tolist()
        class_col = np.ones(corners.shape[0], dtype=int).tolist()
        img_df = pd.DataFrame(
            corners,
            columns=["tl_x", "tl_y", "tr_x", "tr_y", "bl_x", "bl_y", "br_x", "br_y", "cx", "cy"],
        )
        img_df.insert(0, "frame", frame_col)
        img_df["class_id"] = class_col
        self.labels = pd.concat([self.labels, img_df], ignore_index=True)

    def _get_coords_for_image(self, img_name: str) -> np.ndarray:
        df = self.labels.loc[self.labels["frame"] == img_name]
        return df.drop(columns=["frame", "class_id"]).to_numpy()

    def _resize_coords(self, org_width: int, org_height: int, coords: np.ndarray) -> np.ndarray:
        coords = coords.astype(float)
        coords[:, [0, 2, 4, 6, 8]] *= self.img_size / org_width
        coords[:, [1, 3, 5, 7, 9]] *= self.img_size / org_height
        return coords.astype(int)

    def _generate_one_crop(self, img_path: Path, crop_name: str) -> bool:
        img = Image.open(img_path)
        width, height = img.size
        crop_ratio = np.random.choice(_CROP_RATIOS)

        xmin = np.random.randint(0, width)
        ymin = np.random.randint(0, height)
        xmax = min(width, int(xmin + crop_ratio * width))
        ymax = min(height, int(ymin + crop_ratio * width))

        while (xmax - xmin) != (ymax - ymin):
            xmin = np.random.randint(0, width)
            ymin = np.random.randint(0, height)
            xmax = min(width, int(xmin + crop_ratio * width))
            ymax = min(height, int(ymin + crop_ratio * width))

        bounds = np.array([xmin, ymin, xmax, ymax])
        corners = self._get_coords_for_image(img_path.name)
        ic_idx = _check_ic_in_crop(bounds, corners)

        if not ic_idx:
            return False

        crop_img = img.crop((xmin, ymin, xmax, ymax))
        crop_width, crop_height = crop_img.size
        valid_corners = _shift_coords_for_crop(xmin, ymin, corners[ic_idx, :].copy())

        if crop_width != self.img_size or crop_height != self.img_size:
            crop_img = crop_img.resize(
                (self.img_size, self.img_size), resample=Image.Resampling.BICUBIC
            )
            valid_corners = self._resize_coords(crop_width, crop_height, valid_corners)

        self.crop_save_path.mkdir(parents=True, exist_ok=True)
        crop_img.save(self.crop_save_path / crop_name)

        frame_col = np.full(valid_corners.shape[0], crop_name).tolist()
        class_col = np.ones(valid_corners.shape[0], dtype=int).tolist()
        crop_df = pd.DataFrame(
            valid_corners,
            columns=["tl_x", "tl_y", "tr_x", "tr_y", "bl_x", "bl_y", "br_x", "br_y", "cx", "cy"],
        )
        crop_df.insert(0, "frame", frame_col)
        crop_df["class_id"] = class_col
        self.crop_labels = pd.concat([self.crop_labels, crop_df], ignore_index=True)
        return True

    def generate(self, version: Literal["uncropped", "cropped"] = "uncropped") -> None:
        """Generate the dataset.

        Args:
            version: ``"uncropped"`` generates full-size composited images.
                ``"cropped"`` generates square crop patches from previously
                generated full-size images. Run uncropped first.

        Raises:
            ValueError: If an unrecognised version string is provided.
        """
        if version not in ("uncropped", "cropped"):
            raise ValueError(f"version must be 'uncropped' or 'cropped', got {version!r}")

        random.seed(self.seed)
        np.random.seed(self.seed)

        if version == "uncropped":
            for i in tqdm(range(self.dataset_size), desc="Generating images"):
                self._generate_one_image(f"PCB_{i}.jpg")
            self.labels.to_csv(self.save_path / "labels.csv", index=False)
        else:
            jpg_files = [p for p in self.save_path.iterdir() if p.suffix == ".jpg"]
            i = 0
            pbar = tqdm(total=self.dataset_size, desc="Generating crops")
            while i < self.dataset_size:
                img_path = random.choice(jpg_files)
                if self._generate_one_crop(img_path, f"PCB_crop_{i}.jpg"):
                    i += 1
                    pbar.update(1)
            pbar.close()
            self.crop_labels.to_csv(self.crop_save_path / "labels.csv", index=False)
