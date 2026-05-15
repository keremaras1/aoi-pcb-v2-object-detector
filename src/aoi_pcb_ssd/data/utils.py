"""Utility functions for file sorting and image normalisation."""

import re
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

MAX_PIXEL_VALUE: Final[float] = 255.0


def sort_alphanumeric(directory: str | Path) -> list[str]:
    """Return filenames in a directory sorted in human alphanumeric order.

    Alphanumeric sort treats embedded digit runs as numbers so that
    ``pcb_9.jpg`` sorts before ``pcb_10.jpg`` rather than after it.

    Args:
        directory: Path to the directory to list.

    Returns:
        Sorted list of filenames (not full paths).
    """
    def _key(name: str) -> list[int | str]:
        return [
            int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", name)
        ]

    return sorted([p.name for p in Path(directory).iterdir()], key=_key)


def normalize_values(dataset: NDArray[np.uint8]) -> NDArray[np.float64]:
    """Scale a uint8 image array to the [0, 1] float range.

    Args:
        dataset: Array of uint8 pixel values in the range [0, 255].

    Returns:
        Float64 array with values in [0, 1].
    """
    return dataset.astype(np.float64) / MAX_PIXEL_VALUE


def rescale_values(dataset: NDArray[np.float64]) -> NDArray[np.uint8]:
    """Scale a normalised float array back to uint8 pixel values.

    Args:
        dataset: Float64 array with values in [0, 1].

    Returns:
        uint8 array with values in [0, 255].
    """
    return (dataset * MAX_PIXEL_VALUE).astype(np.uint8)
