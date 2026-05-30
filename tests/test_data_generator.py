"""Tests for the Keras-side DataGenerator."""

from pathlib import Path
from typing import Any
from unittest.mock import create_autospec

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from aoi_pcb_ssd.data.data_generator import DataGenerator
from aoi_pcb_ssd.encoding.input_encoder import SSDInputEncoder

# labels.csv column order written by PCBDatasetGenerator.
_CSV_COLUMNS = [
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


@pytest.fixture
def encoder() -> Any:
    """A signature-checked mock of the real SSDInputEncoder.

    Calling it returns a fixed (2, 64, 12) target tensor; the call is
    recorded so tests can assert what the generator passed in.
    """
    mock = create_autospec(SSDInputEncoder, instance=True)
    mock.return_value = np.zeros((2, 64, 12))
    return mock


@pytest.fixture
def crop_dataset(tmp_path: Path) -> Path:
    """Two 16x16 crop images plus a matching labels.csv."""
    for i in range(2):
        Image.new("RGB", (16, 16), (i * 50, 100, 150)).save(tmp_path / f"PCB_crop_{i}.jpg")
    rows = [
        # frame, corners (tl,tr,bl,br), centre, class_id
        ["PCB_crop_0.jpg", 1, 1, 5, 1, 1, 5, 5, 5, 3, 3, 1],
        ["PCB_crop_1.jpg", 2, 2, 6, 2, 2, 6, 6, 6, 4, 4, 1],
    ]
    pd.DataFrame(rows, columns=_CSV_COLUMNS).to_csv(tmp_path / "labels.csv", index=False)
    return tmp_path


class TestImageLoading:
    def test_only_jpg_files_loaded(self, crop_dataset: Path, encoder: Any) -> None:
        gen = DataGenerator(crop_dataset, encoder=encoder)
        # labels.csv must not be picked up as an image.
        assert gen.img_filenames == ["PCB_crop_0.jpg", "PCB_crop_1.jpg"]

    def test_images_stacked_to_array(self, crop_dataset: Path, encoder: Any) -> None:
        gen = DataGenerator(crop_dataset, encoder=encoder)
        gen._img_to_np()
        assert gen.X.shape == (2, 16, 16, 3)
        assert gen.X.dtype == np.uint8


class TestLabelParsing:
    def test_class_id_moved_to_first_column(self, crop_dataset: Path, encoder: Any) -> None:
        gen = DataGenerator(crop_dataset, encoder=encoder)
        gen._parse_csv()
        # One label array per frame, each (n_ics, 11) with class_id first.
        assert len(gen.y) == 2
        assert gen.y[0].shape == (1, 11)
        assert gen.y[0][0, 0] == 1  # class_id
        # Remaining columns are the corner/centre coordinates in order.
        assert list(gen.y[0][0, 1:]) == [1, 1, 5, 1, 1, 5, 5, 5, 3, 3]


class TestGetData:
    def test_returns_images_and_encoded_targets(self, crop_dataset: Path, encoder: Any) -> None:
        gen = DataGenerator(crop_dataset, encoder=encoder, augmentation=False)
        X, y_encoded = gen.get_data()
        assert X.shape == (2, 16, 16, 3)
        assert y_encoded.shape == (2, 64, 12)

    def test_encoder_receives_parsed_labels(self, crop_dataset: Path, encoder: Any) -> None:
        gen = DataGenerator(crop_dataset, encoder=encoder, augmentation=False)
        gen.get_data()
        encoder.assert_called_once()
        (passed_labels,) = encoder.call_args.args
        assert len(passed_labels) == 2
        assert passed_labels[0][0, 0] == 1  # class_id preserved through to the encoder

    def test_augmentation_branch_runs(self, crop_dataset: Path, encoder: Any) -> None:
        gen = DataGenerator(crop_dataset, encoder=encoder, augmentation=True, probability=0.0)
        X, _ = gen.get_data()
        assert X.shape == (2, 16, 16, 3)
