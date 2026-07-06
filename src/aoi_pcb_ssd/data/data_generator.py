# SPDX-License-Identifier: Apache-2.0
"""Keras-side data loader: loads images, parses labels, augments, and encodes.

This module bridges the on-disk synthetic dataset (produced by
``PCBDatasetGenerator``) and the Keras training loop. It loads all images into
a single NumPy array, parses the ``labels.csv`` into per-image label arrays,
optionally applies the augmentation chain, and encodes the labels via the
injected ``SSDInputEncoder`` instance.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from PIL import Image
from tqdm import tqdm

from aoi_pcb_ssd.data.augmentation import DataAugmentationChain
from aoi_pcb_ssd.data.utils import sort_alphanumeric


class DataGenerator:
    """Load a synthetic PCB crop dataset and prepare it for ``model.fit()``.

    Args:
        parent_dir: Directory containing the crop ``.jpg`` images and a
            ``labels.csv`` produced by :class:`PCBDatasetGenerator`.
        encoder: A callable ``SSDInputEncoder`` instance. Called with the list
            of unencoded label arrays and returns the encoded target tensor.
        augmentation: Whether to apply the augmentation chain before encoding.
        probability: Per-augmentation gate probability passed to
            :class:`DataAugmentationChain`.
        seed: Random seed for augmentation reproducibility.
    """

    def __init__(
        self,
        parent_dir: str | Path,
        encoder: Any,
        augmentation: bool = True,
        probability: float = 0.5,
        seed: int = 42,
    ) -> None:
        self.parent_dir = Path(parent_dir)
        self.encoder = encoder
        self.augmentation = augmentation
        self.probability = probability
        self.seed = seed

        self.img_filenames: list[str] = [
            name for name in sort_alphanumeric(self.parent_dir) if name.endswith(".jpg")
        ]
        self.X: NDArray[np.uint8] = np.empty(0, dtype=np.uint8)
        self.y: list[NDArray] = []
        self.y_encoded: NDArray = np.empty(0)

    def _img_to_np(self) -> None:
        # PIL releases the GIL during JPEG decode, so threads decode in
        # parallel; ``map`` preserves input order, keeping each image row
        # aligned with its labels.csv row.
        def load(name: str) -> NDArray[np.uint8]:
            return np.array(Image.open(self.parent_dir / name))

        with ThreadPoolExecutor(max_workers=min(32, os.cpu_count() or 4)) as pool:
            arrays = list(
                tqdm(
                    pool.map(load, self.img_filenames),
                    desc="Loading images",
                    total=len(self.img_filenames),
                )
            )
        self.X = np.array(arrays)

    def _parse_csv(self) -> None:
        df = pd.read_csv(self.parent_dir / "labels.csv")
        for name in tqdm(df["frame"].unique(), desc="Parsing labels"):
            rows = df[df["frame"] == name]
            # Reorder columns to [class_id, tl_x, tl_y, tr_x, tr_y, bl_x, bl_y, br_x, br_y, cx, cy]
            self.y.append(rows.iloc[:, [11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]].to_numpy())

    def _augment(self) -> None:
        chain = DataAugmentationChain(self.X, self.y, probability=self.probability, seed=self.seed)
        self.X, self.y = chain()

    def _encode(self) -> None:
        self.y_encoded = self.encoder(self.y)

    def get_data(self) -> tuple[NDArray[np.uint8], NDArray]:
        """Load, augment, and encode the full dataset.

        Returns:
            Tuple ``(X, y_encoded)`` where ``X`` has shape
            ``(N, H, W, C)`` and ``y_encoded`` has shape ``(N, n_boxes, 12)``.
        """
        self._img_to_np()
        self._parse_csv()
        if self.augmentation:
            self._augment()
        self._encode()
        return self.X, self.y_encoded
