"""Tests for the Keras-side DataGenerator."""

import os
from pathlib import Path
from typing import Any
from unittest.mock import create_autospec, patch

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
def real_encoder() -> SSDInputEncoder:
    """A real encoder over a 2×2 grid, so encoded targets reflect the labels."""
    return SSDInputEncoder(img_height=16, img_width=16, n_classes=1, predictor_sizes=[(2, 2)])


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


class TestCaching:
    def test_cache_hit_returns_identical_data_without_recompute(
        self, crop_dataset: Path, real_encoder: SSDInputEncoder
    ) -> None:
        # Given: a first run with augmentation that populates the cache.
        gen_fresh = DataGenerator(crop_dataset, encoder=real_encoder, probability=0.5)
        X_fresh, y_fresh = gen_fresh.get_data()
        # When: a second generator runs on the same directory with image
        # decoding forbidden, so only a cache hit can produce data.
        gen_cached = DataGenerator(crop_dataset, encoder=real_encoder, probability=0.5)
        with patch.object(gen_cached, "_img_to_np", side_effect=AssertionError("cache miss")):
            X_cached, y_cached = gen_cached.get_data()
        # Then: the cached run reproduces the fresh computation exactly,
        # including dtypes of the images and of every raw label array.
        assert np.array_equal(X_fresh, X_cached)
        assert X_cached.dtype == X_fresh.dtype
        assert np.array_equal(y_fresh, y_cached)
        assert [a.dtype for a in gen_cached.y] == [a.dtype for a in gen_fresh.y]
        for fresh, cached in zip(gen_fresh.y, gen_cached.y):
            assert np.array_equal(fresh, cached)

    @pytest.mark.parametrize(
        "changed", [{"probability": 1.0}, {"seed": 7}, {"augmentation": False}]
    )
    def test_changed_parameters_invalidate_cache(
        self, crop_dataset: Path, encoder: Any, changed: dict
    ) -> None:
        # Given: a cache populated with the default parameters.
        DataGenerator(crop_dataset, encoder=encoder).get_data()
        # When: a generator with one differing augmentation parameter runs.
        gen = DataGenerator(crop_dataset, encoder=encoder, **changed)
        with patch.object(gen, "_img_to_np", wraps=gen._img_to_np) as spy:
            gen.get_data()
        # Then: the parameter change forces a recompute instead of a cache hit.
        spy.assert_called_once()

    def test_labels_csv_edit_invalidates_cache(self, crop_dataset: Path, encoder: Any) -> None:
        # Given: a populated cache, then a same-byte-length coordinate edit
        # to labels.csv (image files untouched).
        DataGenerator(crop_dataset, encoder=encoder, augmentation=False).get_data()
        csv_path = crop_dataset / "labels.csv"
        csv_path.write_text(csv_path.read_text().replace(",3,3,1\n", ",4,4,1\n"))
        stat = csv_path.stat()
        os.utime(csv_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        # When: a new generator runs over the edited dataset.
        gen = DataGenerator(crop_dataset, encoder=encoder, augmentation=False)
        with patch.object(gen, "_img_to_np", wraps=gen._img_to_np) as spy:
            gen.get_data()
        # Then: the label change forces a recompute.
        spy.assert_called_once()

    def test_use_cache_false_ignores_existing_cache(self, crop_dataset: Path, encoder: Any) -> None:
        # Given: a cache already populated by an earlier run.
        DataGenerator(crop_dataset, encoder=encoder, augmentation=False).get_data()
        # When: a generator with use_cache=False runs with cache reading forbidden.
        gen = DataGenerator(crop_dataset, encoder=encoder, augmentation=False, use_cache=False)
        with patch.object(gen, "_load_cache", side_effect=AssertionError("cache read")):
            X, _ = gen.get_data()
        # Then: the data comes from a fresh compute, never from the cache.
        assert X.shape == (2, 16, 16, 3)

    def test_cache_roundtrip_preserves_heterogeneous_labels(
        self, tmp_path: Path, real_encoder: SSDInputEncoder
    ) -> None:
        # Given: crop_0 with two IC rows and crop_1 with one, so per-image
        # label arrays have different shapes (2, 11) and (1, 11).
        for i in range(2):
            Image.new("RGB", (16, 16), (i * 50, 100, 150)).save(tmp_path / f"PCB_crop_{i}.jpg")
        rows = [
            ["PCB_crop_0.jpg", 1, 1, 5, 1, 1, 5, 5, 5, 3, 3, 1],
            ["PCB_crop_0.jpg", 8, 8, 12, 8, 8, 12, 12, 12, 10, 10, 1],
            ["PCB_crop_1.jpg", 2, 2, 6, 2, 2, 6, 6, 6, 4, 4, 1],
        ]
        pd.DataFrame(rows, columns=_CSV_COLUMNS).to_csv(tmp_path / "labels.csv", index=False)
        gen_fresh = DataGenerator(tmp_path, encoder=real_encoder, augmentation=False)
        gen_fresh.get_data()
        # When: a second generator must satisfy get_data() from the cache alone.
        gen_cached = DataGenerator(tmp_path, encoder=real_encoder, augmentation=False)
        with patch.object(gen_cached, "_img_to_np", side_effect=AssertionError("cache miss")):
            gen_cached.get_data()
        # Then: raw labels round-trip exactly — shapes, values, and dtypes.
        assert [a.shape for a in gen_cached.y] == [(2, 11), (1, 11)]
        for fresh, cached in zip(gen_fresh.y, gen_cached.y):
            assert cached.dtype == fresh.dtype
            assert np.array_equal(fresh, cached)
        assert gen_cached.X.dtype == np.uint8

    def test_recompute_after_hit_on_same_instance_stays_consistent(
        self, crop_dataset: Path, encoder: Any
    ) -> None:
        # Given: an instance whose first get_data() call is a cache hit.
        DataGenerator(crop_dataset, encoder=encoder, augmentation=False).get_data()
        gen = DataGenerator(crop_dataset, encoder=encoder, augmentation=False)
        gen.get_data()
        # When: the dataset changes on disk and the same instance reloads.
        img_path = crop_dataset / "PCB_crop_0.jpg"
        stat = img_path.stat()
        os.utime(img_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        X, _ = gen.get_data()
        # Then: labels are parsed fresh rather than appended to the cached
        # ones, keeping images and labels aligned.
        assert len(gen.y) == len(X) == 2

    def test_failed_cache_write_does_not_abort_data_preparation(
        self, crop_dataset: Path, encoder: Any
    ) -> None:
        # Given: a cache_dir path occupied by a regular file, so the cache
        # write cannot succeed.
        blocker = crop_dataset / "blocked"
        blocker.write_text("")
        gen = DataGenerator(crop_dataset, encoder=encoder, augmentation=False, cache_dir=blocker)
        # When: data is prepared.
        X, _ = gen.get_data()
        # Then: preparation succeeds despite the failed cache write.
        assert X.shape == (2, 16, 16, 3)

    def test_save_cache_refuses_misaligned_arrays(self, crop_dataset: Path, encoder: Any) -> None:
        # Given: an instance whose images and labels disagree in length.
        gen = DataGenerator(crop_dataset, encoder=encoder, augmentation=False, use_cache=False)
        gen.get_data()
        gen.y = gen.y[:1]
        # When/Then: writing the cache is refused.
        with pytest.raises(ValueError, match="misaligned"):
            gen._save_cache(gen.cache_dir / "dataprep_test.npz")

    def test_use_cache_false_bypasses_cache(self, crop_dataset: Path, encoder: Any) -> None:
        # Given: caching disabled.
        gen = DataGenerator(crop_dataset, encoder=encoder, augmentation=False, use_cache=False)
        # When: data is prepared.
        gen.get_data()
        # Then: no cache directory is created.
        assert not (crop_dataset / ".cache").exists()

    def test_cache_invalidated_when_input_changes(self, crop_dataset: Path, encoder: Any) -> None:
        # Given: a populated cache, after which an image file changes.
        DataGenerator(crop_dataset, encoder=encoder, augmentation=False).get_data()
        img_path = crop_dataset / "PCB_crop_0.jpg"
        Image.new("RGB", (16, 16), (200, 20, 20)).save(img_path)
        stat = img_path.stat()
        os.utime(img_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        # When: a new generator runs over the changed dataset.
        gen = DataGenerator(crop_dataset, encoder=encoder, augmentation=False)
        with patch.object(gen, "_img_to_np", wraps=gen._img_to_np) as spy:
            gen.get_data()
        # Then: the changed input forces a recompute instead of a cache hit.
        spy.assert_called_once()

    def test_unreadable_cache_file_recomputed(
        self, crop_dataset: Path, real_encoder: SSDInputEncoder
    ) -> None:
        # Given: a populated cache whose file is then corrupted.
        gen_fresh = DataGenerator(crop_dataset, encoder=real_encoder, augmentation=False)
        X_fresh, y_fresh = gen_fresh.get_data()
        (cache_file,) = (crop_dataset / ".cache").glob("*.npz")
        cache_file.write_bytes(b"not an npz")
        # When: a second generator runs against the corrupt cache.
        gen = DataGenerator(crop_dataset, encoder=real_encoder, augmentation=False)
        X, y = gen.get_data()
        # Then: it falls back to a recompute and returns the same data.
        assert np.array_equal(X_fresh, X)
        assert np.array_equal(y_fresh, y)

    def test_cache_dir_parameter_overrides_default_location(
        self, crop_dataset: Path, encoder: Any
    ) -> None:
        # Given: an explicit cache directory.
        cache_dir = crop_dataset / "custom_cache"
        gen = DataGenerator(crop_dataset, encoder=encoder, augmentation=False, cache_dir=cache_dir)
        # When: data is prepared.
        gen.get_data()
        # Then: cache files land there, not in the default parent_dir/.cache.
        assert list(cache_dir.glob("*.npz"))
        assert not (crop_dataset / ".cache").exists()
