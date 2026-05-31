"""Tests for the synthetic PCB dataset generator."""

import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from aoi_pcb_ssd.data.dataset_generator import (
    _LABEL_COLUMNS,
    PCBDatasetGenerator,
    _centroids_to_corners,
    _check_ic_in_crop,
    _get_new_corners,
    _rotate,
    _rotate_coords,
    _shift_coords_for_crop,
)


def _make_templates_dir(root: Path, n_ics: int = 2) -> Path:
    """Build templates_dir/pcb_template_1/template_object_1/ with one template."""
    obj = root / "pcb_template_1" / "template_object_1"
    obj.mkdir(parents=True)
    Image.new("RGB", (96, 96), (40, 40, 40)).save(obj / "background_pcb.jpg")
    rows = []
    for i in range(n_ics):
        Image.new("RGB", (40, 40), (200, 120, 60)).save(obj / f"cropped_ic_{i}.jpg")
        rows.append({"ic_frame": f"ic_{i}", "cx": 48, "cy": 48, "w": 40, "h": 40})
    pd.DataFrame(rows, columns=["ic_frame", "cx", "cy", "w", "h"]).to_csv(
        obj / "ic_centroids.csv", index=False
    )
    return root


@pytest.fixture
def generator(tmp_path: Path) -> PCBDatasetGenerator:
    """A generator over one synthetic template, sized for a fast smoke test."""
    templates_dir = _make_templates_dir(tmp_path / "templates", n_ics=2)
    return PCBDatasetGenerator(
        templates_dir=templates_dir,
        save_dir=tmp_path / "generated",
        crop_save_dir=tmp_path / "crops",
        rotation_range=2,
        img_size=32,
        dataset_size=2,
        placement_offset_x=2,
        placement_offset_y=2,
        seed=42,
    )


class TestPureHelpers:
    def test_centroids_to_corners(self) -> None:
        # Given: a centroid at (50, 50) with width=20, height=10.
        corners = _centroids_to_corners(np.array([[50, 50, 20, 10]]))
        # Then: tl=(40,45), tr=(60,45), bl=(40,55), br=(60,55), centre=(50,50).
        assert list(corners[0]) == [40, 45, 60, 45, 40, 55, 60, 55, 50, 50]

    def test_check_ic_in_crop_keeps_only_fully_contained(self) -> None:
        # Given: a 100×100 crop and two ICs — one inside, one partially outside.
        bounds = np.array([0, 0, 100, 100])
        coords = np.array(
            [
                [10, 10, 20, 10, 10, 20, 20, 20, 15, 15],  # inside
                [90, 90, 120, 90, 90, 120, 120, 120, 105, 105],  # spills out
            ]
        )
        # When/Then: only the fully-contained IC is returned.
        assert _check_ic_in_crop(bounds, coords) == [0]

    def test_shift_coords_for_crop_is_origin_relative(self) -> None:
        # Given: corner coords and a crop origin at (5, 10).
        coords = np.array([[10, 20, 30, 20, 10, 40, 30, 40, 20, 30]])
        # When: shifted to be crop-relative.
        shifted = _shift_coords_for_crop(5, 10, coords.copy())
        # Then: x-coords decrease by 5, y-coords decrease by 10.
        assert list(shifted[0]) == [5, 10, 25, 10, 5, 30, 25, 30, 15, 20]

    def test_rotate_90_degrees_counterclockwise(self) -> None:
        # Given: a point at (1, 0) and the origin as pivot.
        # When: rotated 90° counterclockwise (π/2 radians).
        qx, qy = _rotate((0.0, 0.0), (1.0, 0.0), math.pi / 2)
        # Then: the point lands at (0, 1).
        assert qx == pytest.approx(0.0, abs=1e-10)
        assert qy == pytest.approx(1.0, abs=1e-10)

    def test_rotate_180_degrees(self) -> None:
        # Given: a point at (1, 0) and the origin as pivot.
        # When: rotated 180° (π radians).
        qx, qy = _rotate((0.0, 0.0), (1.0, 0.0), math.pi)
        # Then: the point lands at (-1, 0).
        assert qx == pytest.approx(-1.0, abs=1e-10)
        assert qy == pytest.approx(0.0, abs=1e-10)

    def test_rotate_coords_identity_at_zero_degrees(self) -> None:
        # Given: 4 corners + 1 centre in (5, 2) layout (as _rotate_coords expects).
        coords = np.array([[10.0, 20.0], [30.0, 20.0], [10.0, 40.0], [30.0, 40.0], [20.0, 30.0]])
        # When: rotated by 0°.
        result = _rotate_coords(coords.copy(), angle=0.0)
        # Then: the flattened output equals the original coordinates.
        np.testing.assert_allclose(result, coords.flatten(), atol=1e-10)

    def test_get_new_corners_zero_transform_is_identity(self) -> None:
        # Given: a flat 10-element corner row [tl_x, tl_y, ..., cx, cy].
        corners = np.array([10.0, 20.0, 30.0, 20.0, 10.0, 40.0, 30.0, 40.0, 20.0, 30.0])
        # When: zero offset and zero rotation are applied.
        result = _get_new_corners(corners.copy(), offset_x=0, offset_y=0, rotation=0.0)
        # Then: the output is identical to the input.
        np.testing.assert_allclose(result, corners, atol=1e-10)


class TestTemplateLoading:
    def test_loads_template_objects(self, generator: PCBDatasetGenerator) -> None:
        assert len(generator.template_objects) == 1

    def test_ignores_non_directory_entries(self, tmp_path: Path) -> None:
        templates = _make_templates_dir(tmp_path / "templates", n_ics=1)
        (templates / "stray.txt").write_text("not a template")
        gen = PCBDatasetGenerator(
            templates_dir=templates,
            save_dir=tmp_path / "generated",
            crop_save_dir=tmp_path / "crops",
            rotation_range=2,
            img_size=32,
            dataset_size=1,
            placement_offset_x=2,
            placement_offset_y=2,
        )
        assert len(gen.template_objects) == 1


class TestGenerateUncropped:
    def test_creates_images_and_labels(self, generator: PCBDatasetGenerator) -> None:
        # Given: a generator configured for dataset_size=2.
        # When: uncropped generation is run.
        generator.generate("uncropped")
        images = sorted(p.name for p in generator.save_path.glob("PCB_*.jpg"))
        # Then: exactly 2 images and a labels.csv are written.
        assert images == ["PCB_0.jpg", "PCB_1.jpg"]
        assert (generator.save_path / "labels.csv").exists()

    def test_labels_have_expected_columns(self, generator: PCBDatasetGenerator) -> None:
        # Given/When: uncropped generation with 2 images × 2 ICs each.
        generator.generate("uncropped")
        labels = pd.read_csv(generator.save_path / "labels.csv")
        # Then: labels.csv has the canonical column set and 4 rows.
        assert list(labels.columns) == _LABEL_COLUMNS
        assert len(labels) == 4

    def test_seed_is_reproducible(self, tmp_path: Path) -> None:
        # Given: two independent generators with the same seed.
        def run(out: str) -> pd.DataFrame:
            templates = _make_templates_dir(tmp_path / out / "t", n_ics=2)
            gen = PCBDatasetGenerator(
                templates_dir=templates,
                save_dir=tmp_path / out / "gen",
                crop_save_dir=tmp_path / out / "crop",
                rotation_range=2,
                img_size=32,
                dataset_size=2,
                placement_offset_x=2,
                placement_offset_y=2,
                seed=7,
            )
            gen.generate("uncropped")
            return pd.read_csv(gen.save_path / "labels.csv")

        # When: both runs complete.
        # Then: the label CSVs are identical.
        pd.testing.assert_frame_equal(run("a"), run("b"))


class TestGenerateCropped:
    def test_creates_crops_and_labels(self, generator: PCBDatasetGenerator) -> None:
        # Given: uncropped images already generated.
        generator.generate("uncropped")
        # When: the cropped phase runs.
        generator.generate("cropped")
        crops = list(generator.crop_save_path.glob("PCB_crop_*.jpg"))
        labels = pd.read_csv(generator.crop_save_path / "labels.csv")
        # Then: 2 crop images and a matching labels.csv are written.
        assert len(crops) == 2
        assert list(labels.columns) == _LABEL_COLUMNS

    def test_crops_resized_to_img_size(self, generator: PCBDatasetGenerator) -> None:
        # Given: uncropped generation done; img_size=32.
        generator.generate("uncropped")
        generator.generate("cropped")
        crop = next(generator.crop_save_path.glob("PCB_crop_*.jpg"))
        # Then: each crop is exactly 32×32 pixels.
        assert Image.open(crop).size == (32, 32)

    def test_crop_without_ic_is_rejected(self, generator: PCBDatasetGenerator) -> None:
        # Given: uncropped images available; patches force a crop that misses the IC.
        generator.generate("uncropped")
        img_path = next(generator.save_path.glob("PCB_*.jpg"))
        # When: a crop whose bounds exclude the IC is attempted.
        with (
            patch("aoi_pcb_ssd.data.dataset_generator.np.random.choice", return_value=0.1),
            patch("aoi_pcb_ssd.data.dataset_generator.np.random.randint", return_value=0),
        ):
            # Then: _generate_one_crop returns False (crop is discarded).
            assert generator._generate_one_crop(img_path, "miss.jpg") is False


class TestVersionValidation:
    def test_invalid_version_raises(self, generator: PCBDatasetGenerator) -> None:
        # Given: a generator.
        # When: generate() is called with an unrecognised version string.
        # Then: ValueError mentions the valid options.
        with pytest.raises(ValueError, match="uncropped"):
            generator.generate("bogus")  # type: ignore[arg-type]
