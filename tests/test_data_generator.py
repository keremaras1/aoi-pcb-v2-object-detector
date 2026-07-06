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
    """Two 16×16 crop images plus a matching labels.csv."""
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
        # Given: a directory with 2 JPEG images and a labels.csv.
        gen = DataGenerator(crop_dataset, encoder=encoder)
        # Then: img_filenames contains only the JPEG files (CSV excluded).
        assert gen.img_filenames == ["PCB_crop_0.jpg", "PCB_crop_1.jpg"]

    def test_images_loaded_as_uint8_array(self, crop_dataset: Path, encoder: Any) -> None:
        # Given: two 16×16 RGB images.
        gen = DataGenerator(crop_dataset, encoder=encoder, augmentation=False)
        # When: data is loaded end-to-end.
        X, _ = gen.get_data()
        # Then: images are stacked into a uint8 array with the expected shape.
        assert X.shape == (2, 16, 16, 3)
        assert X.dtype == np.uint8

    def test_parallel_load_matches_serial_reference(self, tmp_path: Path, encoder: Any) -> None:
        # Given: twelve crops whose pixel content encodes their index, so any
        # ordering mix-up between threads changes the loaded array.
        for i in range(12):
            colour = (i * 20, 255 - i * 20, i * 10)
            Image.new("RGB", (16, 16), colour).save(tmp_path / f"PCB_crop_{i}.jpg")
        gen = DataGenerator(tmp_path, encoder=encoder)
        # When: images are loaded through _img_to_np.
        gen._img_to_np()
        # Then: the result is identical to a serial per-file load in filename order.
        serial = np.array([np.array(Image.open(tmp_path / name)) for name in gen.img_filenames])
        assert np.array_equal(gen.X, serial)
        assert gen.X.dtype == serial.dtype


class TestLabelParsing:
    def test_encoder_receives_class_id_in_first_column(
        self, crop_dataset: Path, encoder: Any
    ) -> None:
        # Given: a labels.csv where class_id is the rightmost column.
        gen = DataGenerator(crop_dataset, encoder=encoder, augmentation=False)
        # When: get_data() is called (internally parses and reformats the CSV).
        gen.get_data()
        (passed_labels,) = encoder.call_args.args
        # Then: the encoder receives labels with class_id moved to column 0,
        # followed by the eight corner coordinates and cell centre.
        assert len(passed_labels) == 2
        assert passed_labels[0].shape == (1, 11)
        assert passed_labels[0][0, 0] == 1  # class_id
        assert list(passed_labels[0][0, 1:]) == [1, 1, 5, 1, 1, 5, 5, 5, 3, 3]


class TestGetData:
    def test_returns_images_and_encoded_targets(self, crop_dataset: Path, encoder: Any) -> None:
        # Given: a dataset with 2 images, an encoder that returns (2, 64, 12).
        gen = DataGenerator(crop_dataset, encoder=encoder, augmentation=False)
        # When: get_data() is called.
        X, y_encoded = gen.get_data()
        # Then: shapes match the dataset size and encoder contract.
        assert X.shape == (2, 16, 16, 3)
        assert y_encoded.shape == (2, 64, 12)

    def test_encoder_receives_parsed_labels(self, crop_dataset: Path, encoder: Any) -> None:
        # Given: a dataset with 2 annotated images.
        gen = DataGenerator(crop_dataset, encoder=encoder, augmentation=False)
        # When: get_data() runs.
        gen.get_data()
        # Then: the encoder is called exactly once with all 2 label arrays,
        # each preserving the class_id in column 0.
        encoder.assert_called_once()
        (passed_labels,) = encoder.call_args.args
        assert len(passed_labels) == 2
        assert passed_labels[0][0, 0] == 1  # class_id preserved through to the encoder

    def test_augmentation_branch_runs(self, crop_dataset: Path, encoder: Any) -> None:
        # Given: augmentation enabled but with probability=0 (transforms are no-ops).
        gen = DataGenerator(crop_dataset, encoder=encoder, augmentation=True, probability=0.0)
        # When: get_data() is called.
        X, _ = gen.get_data()
        # Then: shape is unchanged (augmentation path was exercised without modifying images).
        assert X.shape == (2, 16, 16, 3)
