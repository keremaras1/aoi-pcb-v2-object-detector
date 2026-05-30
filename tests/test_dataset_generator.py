"""Tests for the synthetic PCB dataset generator."""

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
        # (cx, cy, w, h) -> tl, tr, bl, br, centre.
        corners = _centroids_to_corners(np.array([[50, 50, 20, 10]]))
        assert list(corners[0]) == [40, 45, 60, 45, 40, 55, 60, 55, 50, 50]

    def test_check_ic_in_crop_keeps_only_fully_contained(self) -> None:
        bounds = np.array([0, 0, 100, 100])
        coords = np.array(
            [
                [10, 10, 20, 10, 10, 20, 20, 20, 15, 15],  # inside
                [90, 90, 120, 90, 90, 120, 120, 120, 105, 105],  # spills out
            ]
        )
        assert _check_ic_in_crop(bounds, coords) == [0]

    def test_shift_coords_for_crop_is_origin_relative(self) -> None:
        coords = np.array([[10, 20, 30, 20, 10, 40, 30, 40, 20, 30]])
        shifted = _shift_coords_for_crop(5, 10, coords.copy())
        assert list(shifted[0]) == [5, 10, 25, 10, 5, 30, 25, 30, 15, 20]


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
        generator.generate("uncropped")
        images = sorted(p.name for p in generator.save_path.glob("PCB_*.jpg"))
        assert images == ["PCB_0.jpg", "PCB_1.jpg"]
        assert (generator.save_path / "labels.csv").exists()

    def test_labels_have_expected_columns(self, generator: PCBDatasetGenerator) -> None:
        generator.generate("uncropped")
        labels = pd.read_csv(generator.save_path / "labels.csv")
        assert list(labels.columns) == _LABEL_COLUMNS
        # Two images x two ICs each.
        assert len(labels) == 4

    def test_seed_is_reproducible(self, tmp_path: Path) -> None:
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

        pd.testing.assert_frame_equal(run("a"), run("b"))


class TestGenerateCropped:
    def test_creates_crops_and_labels(self, generator: PCBDatasetGenerator) -> None:
        generator.generate("uncropped")
        generator.generate("cropped")
        crops = list(generator.crop_save_path.glob("PCB_crop_*.jpg"))
        assert len(crops) == 2
        labels = pd.read_csv(generator.crop_save_path / "labels.csv")
        assert list(labels.columns) == _LABEL_COLUMNS

    def test_crops_resized_to_img_size(self, generator: PCBDatasetGenerator) -> None:
        generator.generate("uncropped")
        generator.generate("cropped")
        crop = next(generator.crop_save_path.glob("PCB_crop_*.jpg"))
        assert Image.open(crop).size == (32, 32)

    def test_crop_without_ic_is_rejected(self, generator: PCBDatasetGenerator) -> None:
        # A crop too small to contain the centred IC yields no label and is skipped.
        generator.generate("uncropped")
        img_path = next(generator.save_path.glob("PCB_*.jpg"))
        with (
            patch("aoi_pcb_ssd.data.dataset_generator.np.random.choice", return_value=0.1),
            patch("aoi_pcb_ssd.data.dataset_generator.np.random.randint", return_value=0),
        ):
            assert generator._generate_one_crop(img_path, "miss.jpg") is False


class TestVersionValidation:
    def test_invalid_version_raises(self, generator: PCBDatasetGenerator) -> None:
        with pytest.raises(ValueError, match="uncropped"):
            generator.generate("bogus")  # type: ignore[arg-type]
