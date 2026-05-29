"""Tests for the data utility helpers."""

from pathlib import Path

import numpy as np
import pytest

from aoi_pcb_ssd.data.utils import (
    MAX_PIXEL_VALUE,
    normalize_values,
    rescale_values,
    sort_alphanumeric,
)


class TestNormalizeValues:
    def test_max_maps_to_one(self) -> None:
        data = np.array([[[255, 255, 255]]], dtype=np.uint8)
        result = normalize_values(data)
        assert result.max() == pytest.approx(1.0)

    def test_zero_stays_zero(self) -> None:
        data = np.zeros((2, 2, 3), dtype=np.uint8)
        assert normalize_values(data).max() == 0.0

    def test_midpoint(self) -> None:
        data = np.array([[[128]]], dtype=np.uint8)
        result = normalize_values(data)
        assert result[0, 0, 0] == pytest.approx(128 / 255.0)

    def test_output_dtype_is_float(self) -> None:
        data = np.ones((4, 4, 3), dtype=np.uint8) * 100
        assert normalize_values(data).dtype == np.float64


class TestRescaleValues:
    def test_one_maps_to_255(self) -> None:
        data = np.ones((2, 2, 3), dtype=np.float64)
        assert rescale_values(data).max() == 255

    def test_zero_stays_zero(self) -> None:
        data = np.zeros((2, 2, 3), dtype=np.float64)
        assert rescale_values(data).max() == 0

    def test_output_dtype_is_uint8(self) -> None:
        data = np.ones((2, 2, 3), dtype=np.float64) * 0.5
        assert rescale_values(data).dtype == np.uint8

    def test_roundtrip(self) -> None:
        original = np.array([[[0, 128, 255]]], dtype=np.uint8)
        roundtripped = rescale_values(normalize_values(original))
        # Allow +/-1 for floating-point rounding.
        assert np.abs(original.astype(int) - roundtripped.astype(int)).max() <= 1


class TestMaxPixelValueConstant:
    def test_value_is_255(self) -> None:
        assert MAX_PIXEL_VALUE == 255.0


class TestSortAlphanumeric:
    def test_numeric_order(self, tmp_path: Path) -> None:
        # Lexicographic order would place pcb_10 before pcb_2; numeric order must not.
        for name in ["pcb_10.jpg", "pcb_2.jpg", "pcb_1.jpg", "pcb_20.jpg"]:
            (tmp_path / name).touch()
        result = sort_alphanumeric(tmp_path)
        assert result == ["pcb_1.jpg", "pcb_2.jpg", "pcb_10.jpg", "pcb_20.jpg"]

    def test_case_insensitive_text(self, tmp_path: Path) -> None:
        for name in ["B.jpg", "a.jpg", "C.jpg"]:
            (tmp_path / name).touch()
        assert sort_alphanumeric(tmp_path) == ["a.jpg", "B.jpg", "C.jpg"]

    def test_returns_names_not_paths(self, tmp_path: Path) -> None:
        (tmp_path / "pcb_0.jpg").touch()
        assert sort_alphanumeric(tmp_path) == ["pcb_0.jpg"]

    def test_accepts_str_path(self, tmp_path: Path) -> None:
        (tmp_path / "pcb_0.jpg").touch()
        assert sort_alphanumeric(str(tmp_path)) == ["pcb_0.jpg"]

    def test_empty_directory(self, tmp_path: Path) -> None:
        assert sort_alphanumeric(tmp_path) == []
