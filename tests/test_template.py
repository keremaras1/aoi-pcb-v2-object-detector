"""Tests for the Template asset loader."""

from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from aoi_pcb_ssd.data.template import Template


def _make_template(
    path: Path,
    n_ics: int = 2,
    with_background: bool = True,
    with_csv: bool = True,
) -> Path:
    """Create a synthetic template object directory under ``path``."""
    path.mkdir(parents=True, exist_ok=True)
    if with_background:
        Image.new("RGB", (64, 64), (50, 50, 50)).save(path / "background_pcb.jpg")
    rows = []
    for i in range(n_ics):
        Image.new("RGB", (16, 16), (200, 100, 50)).save(path / f"cropped_ic_{i}.jpg")
        rows.append({"ic_frame": f"ic_{i}", "cx": 32, "cy": 32, "w": 16, "h": 16})
    if with_csv:
        pd.DataFrame(rows, columns=["ic_frame", "cx", "cy", "w", "h"]).to_csv(
            path / "ic_centroids.csv", index=False
        )
    return path


class TestTemplateLoading:
    def test_loads_background_and_cutouts(self, tmp_path: Path) -> None:
        template = Template(_make_template(tmp_path / "t", n_ics=2))
        assert template.get_background() is not None
        assert len(template.get_ic_cutouts()) == 2

    def test_background_is_pil_image(self, tmp_path: Path) -> None:
        template = Template(_make_template(tmp_path / "t"))
        assert isinstance(template.get_background(), Image.Image)

    def test_labels_have_centroid_columns(self, tmp_path: Path) -> None:
        template = Template(_make_template(tmp_path / "t"))
        labels = template.get_labels_df()
        for column in ("cx", "cy", "w", "h"):
            assert column in labels.columns

    def test_cutout_count_matches_csv_rows(self, tmp_path: Path) -> None:
        template = Template(_make_template(tmp_path / "t", n_ics=3))
        assert len(template.get_ic_cutouts()) == len(template.get_labels_df())


class TestTemplateValidation:
    def test_missing_background_raises(self, tmp_path: Path) -> None:
        path = _make_template(tmp_path / "t", n_ics=2, with_background=False)
        with pytest.raises(ValueError, match="No background image"):
            Template(path)

    def test_missing_cutouts_raises(self, tmp_path: Path) -> None:
        path = _make_template(tmp_path / "t", n_ics=0)
        with pytest.raises(ValueError, match="No IC cutout"):
            Template(path)

    def test_missing_labels_raises_only_on_access(self, tmp_path: Path) -> None:
        # A template with no CSV still loads; the error surfaces on access.
        path = _make_template(tmp_path / "t", n_ics=2, with_csv=False)
        template = Template(path)
        with pytest.raises(ValueError, match="No label CSV"):
            template.get_labels_df()
