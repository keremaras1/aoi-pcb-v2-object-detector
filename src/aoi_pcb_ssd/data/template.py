"""Template asset loader for synthetic PCB dataset generation."""

from pathlib import Path

import pandas as pd
from PIL import Image

from aoi_pcb_ssd.data.utils import sort_alphanumeric


class Template:
    """Loads and holds the assets for one PCB template directory.

    A template directory contains the outputs of ``generate_template.py``:
    - One ``background_pcb.jpg`` — the PCB image with IC regions filled with
      the dominant background colour.
    - One or more ``cropped_ic_*.jpg`` — individual IC cutout images.
    - One ``ic_centroids.csv`` — centroid coordinates and dimensions per IC.

    Args:
        template_path: Path to the template object directory
            (e.g. ``templates/pcb_template_1/template_object_1/``).

    Raises:
        ValueError: If no background image or no IC cutouts are found.
    """

    def __init__(self, template_path: str | Path) -> None:
        self.path = Path(template_path)
        self.background: Image.Image | None = None
        self.ic_cutouts: list[Image.Image] = []
        self.labels: pd.DataFrame | None = None
        self._load()

    def _load(self) -> None:
        for name in sort_alphanumeric(self.path):
            full_path = self.path / name
            if name.endswith(".jpg"):
                if name.startswith("background"):
                    self.background = Image.open(full_path)
                else:
                    self.ic_cutouts.append(Image.open(full_path))
            elif name.endswith(".csv"):
                self.labels = pd.read_csv(full_path)

        if self.background is None:
            raise ValueError(f"No background image found in {self.path}")
        if not self.ic_cutouts:
            raise ValueError(f"No IC cutout images found in {self.path}")

    def get_background(self) -> Image.Image:
        """Return the background PCB image.

        Returns:
            PIL Image of the background with IC regions filled.
        """
        assert self.background is not None
        return self.background

    def get_ic_cutouts(self) -> list[Image.Image]:
        """Return the list of IC cutout images.

        Returns:
            List of PIL Images, one per IC position on the template.
        """
        return self.ic_cutouts

    def get_labels_df(self) -> pd.DataFrame:
        """Return the IC centroid label DataFrame.

        Returns:
            DataFrame with columns ``ic_frame, cx, cy, w, h``.

        Raises:
            ValueError: If no CSV label file was found in the template directory.
        """
        if self.labels is None:
            raise ValueError(f"No label CSV found in {self.path}")
        return self.labels
