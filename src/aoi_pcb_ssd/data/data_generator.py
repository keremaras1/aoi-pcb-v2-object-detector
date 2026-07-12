# SPDX-License-Identifier: Apache-2.0
"""Keras-side data loader: loads images, parses labels, augments, and encodes.

This module bridges the on-disk synthetic dataset (produced by
``PCBDatasetGenerator``) and the Keras training loop. It loads all images into
a single NumPy array, parses the ``labels.csv`` into per-image label arrays,
optionally applies the augmentation chain, and encodes the labels via the
injected ``SSDInputEncoder`` instance.
"""

from __future__ import annotations

import hashlib
import inspect
import os
import pickle
import zipfile
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from PIL import Image
from tqdm import tqdm

from aoi_pcb_ssd.data.augmentation import DataAugmentationChain
from aoi_pcb_ssd.data.utils import sort_alphanumeric

# Bump when the cached array layout or the behaviour of the prep steps baked
# into cache files (_img_to_np, _parse_csv) changes. DataAugmentationChain is
# covered separately by the source hash below.
_CACHE_VERSION = 1

_AUGMENTATION_SOURCE_HASH = hashlib.sha1(
    inspect.getsource(DataAugmentationChain).encode()
).hexdigest()


class DataGenerator:
    """Load a synthetic PCB crop dataset and prepare it for ``model.fit()``.

    The prepared (post-augmentation, unencoded) arrays are cached on disk and
    reused by later runs over the same inputs. The cache key covers the image
    and label file names, sizes, and mtimes together with the augmentation
    parameters and the augmentation chain's source code, so regenerating the
    dataset — or re-cloning it, which rewrites mtimes — triggers a recompute.
    Encoding always runs fresh, so encoder configuration is deliberately not
    part of the key. A cache file stores the raw pixel array and occupies
    roughly ``N x H x W x C`` bytes.

    Args:
        parent_dir: Directory containing the crop ``.jpg`` images and a
            ``labels.csv`` produced by :class:`PCBDatasetGenerator`.
        encoder: A callable ``SSDInputEncoder`` instance. Called with the list
            of unencoded label arrays and returns the encoded target tensor.
        augmentation: Whether to apply the augmentation chain before encoding.
        probability: Per-augmentation gate probability passed to
            :class:`DataAugmentationChain`.
        seed: Random seed for augmentation reproducibility.
        use_cache: Whether to read and write the prepared-dataset disk cache.
        cache_dir: Directory holding cache files. Defaults to
            ``parent_dir/.cache``.
    """

    def __init__(
        self,
        parent_dir: str | Path,
        encoder: Any,
        augmentation: bool = True,
        probability: float = 0.5,
        seed: int = 42,
        use_cache: bool = True,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.parent_dir = Path(parent_dir)
        self.encoder = encoder
        self.augmentation = augmentation
        self.probability = probability
        self.seed = seed
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir) if cache_dir is not None else self.parent_dir / ".cache"

        self.img_filenames: list[str] = [
            name for name in sort_alphanumeric(self.parent_dir) if name.endswith(".jpg")
        ]
        self.X: NDArray[np.uint8] = np.empty(0, dtype=np.uint8)
        self.y: list[NDArray] = []
        self.y_encoded: NDArray = np.empty(0)

    def _img_to_np(self) -> None:
        arrays = []
        for name in tqdm(self.img_filenames, desc="Loading images"):
            arrays.append(np.array(Image.open(self.parent_dir / name)))
        self.X = np.array(arrays)

    def _parse_csv(self) -> None:
        df = pd.read_csv(self.parent_dir / "labels.csv")
        labels = []
        for name in tqdm(df["frame"].unique(), desc="Parsing labels"):
            rows = df[df["frame"] == name]
            # Reorder columns to [class_id, tl_x, tl_y, tr_x, tr_y, bl_x, bl_y, br_x, br_y, cx, cy]
            labels.append(rows.iloc[:, [11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]].to_numpy())
        self.y = labels

    def _augment(self) -> None:
        chain = DataAugmentationChain(self.X, self.y, probability=self.probability, seed=self.seed)
        self.X, self.y = chain()

    def _encode(self) -> None:
        self.y_encoded = self.encoder(self.y)

    def _cache_path(self) -> Path:
        sig = hashlib.sha1(
            f"v{_CACHE_VERSION}|{_AUGMENTATION_SOURCE_HASH}|"
            f"{self.augmentation}|{self.probability}|{self.seed}".encode()
        )
        for name in [*self.img_filenames, "labels.csv"]:
            stat = (self.parent_dir / name).stat()
            sig.update(f"|{name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
        return self.cache_dir / f"dataprep_{sig.hexdigest()[:16]}.npz"

    def _load_cache(self, path: Path) -> bool:
        if not path.is_file():
            return False
        try:
            # allow_pickle is required for the object-dtype label array; cache
            # files are generated locally by _save_cache, not untrusted input.
            with np.load(path, allow_pickle=True) as data:
                X, y = data["X"], data["y"]
        except (
            OSError,
            EOFError,
            ValueError,
            KeyError,
            zipfile.BadZipFile,
            pickle.UnpicklingError,
        ) as exc:
            print(f"Cache file unusable ({type(exc).__name__}: {exc}), recomputing: {path}")
            return False
        self.X = X
        self.y = list(y)
        print(f"Loaded prepared dataset from cache: {path}")
        return True

    def _save_cache(self, path: Path) -> None:
        if len(self.X) != len(self.y):
            raise ValueError(
                f"Refusing to cache a misaligned dataset: "
                f"{len(self.X)} images vs {len(self.y)} label arrays"
            )
        # A 1-D object array keeps each per-image label array intact;
        # np.array(..., dtype=object) would merge same-shape labels into a
        # single multidimensional block.
        y_obj = np.empty(len(self.y), dtype=object)
        y_obj[:] = self.y
        # Write-then-rename so an interrupted run cannot leave a truncated
        # file at the final path. A failed write only costs the speedup of
        # the next run, so it warns instead of aborting the prepared run.
        tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp_path, "wb") as file:
                np.savez(file, X=self.X, y=y_obj)
            os.replace(tmp_path, path)
        except OSError as exc:
            with suppress(OSError):
                tmp_path.unlink(missing_ok=True)
            print(f"Could not write dataset cache ({type(exc).__name__}: {exc}): {path}")

    def get_data(self) -> tuple[NDArray[np.uint8], NDArray]:
        """Load, augment, and encode the full dataset.

        A cache hit skips image decoding, label parsing, and augmentation;
        encoding runs on every call.

        Returns:
            Tuple ``(X, y_encoded)`` where ``X`` has shape
            ``(N, H, W, C)`` and ``y_encoded`` has shape ``(N, n_boxes, 12)``.
        """
        cache_path = self._cache_path() if self.use_cache else None
        if cache_path is None or not self._load_cache(cache_path):
            self._img_to_np()
            self._parse_csv()
            if self.augmentation:
                self._augment()
            if cache_path is not None:
                self._save_cache(cache_path)
        self._encode()
        return self.X, self.y_encoded
