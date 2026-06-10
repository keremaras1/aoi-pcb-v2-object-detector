"""Tests for the data augmentation chain."""

from unittest.mock import patch

import numpy as np
import pytest

from aoi_pcb_ssd.data.augmentation import DataAugmentationChain

# Label layout: [class_id, tl_x, tl_y, tr_x, tr_y, bl_x, bl_y, br_x, br_y, cx, cy].
_X_INDICES = [1, 3, 5, 7, 9]
_Y_INDICES = [2, 4, 6, 8, 10]


def _label() -> np.ndarray:
    """One IC label row with known corner coordinates."""
    return np.array([[1, 2, 3, 4, 3, 2, 7, 4, 7, 3, 5]], dtype=float)


def _chain(probability: float) -> DataAugmentationChain:
    """A chain with empty data; transforms are exercised directly."""
    return DataAugmentationChain(
        np.zeros((0, 4, 4, 3), dtype=np.uint8), [], probability=probability, seed=0
    )


class TestHorizontalFlip:
    def test_applied_mirrors_x_coordinates(self) -> None:
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        image[:, 0] = 255  # left edge marker
        img, label = _chain(1.0).horizontal_flip(image, _label())
        # x' = width - x for every x coordinate.
        assert list(label[0, _X_INDICES]) == [8, 6, 8, 6, 7]
        # y coordinates are untouched.
        assert list(label[0, _Y_INDICES]) == [3, 3, 7, 7, 5]
        # Image is flipped left-right: marker now on the right edge.
        assert np.all(img[:, -1] == 255)

    def test_skipped_leaves_data_unchanged(self) -> None:
        image = np.arange(300, dtype=np.uint8).reshape(10, 10, 3)
        label = _label()
        img, out = _chain(0.0).horizontal_flip(image.copy(), label.copy())
        assert np.array_equal(img, image)
        assert np.array_equal(out, label)


class TestVerticalFlip:
    def test_applied_mirrors_y_coordinates(self) -> None:
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        image[0, :] = 255  # top edge marker
        img, label = _chain(1.0).vertical_flip(image, _label())
        # y' = height - y for every y coordinate.
        assert list(label[0, _Y_INDICES]) == [7, 7, 3, 3, 5]
        assert list(label[0, _X_INDICES]) == [2, 4, 2, 4, 3]
        assert np.all(img[-1, :] == 255)

    def test_skipped_leaves_data_unchanged(self) -> None:
        image = np.arange(300, dtype=np.uint8).reshape(10, 10, 3)
        label = _label()
        img, out = _chain(0.0).vertical_flip(image.copy(), label.copy())
        assert np.array_equal(img, image)
        assert np.array_equal(out, label)


class TestPerpendicularRotate:
    def test_square_image_shape_preserved(self) -> None:
        image = np.random.randint(0, 256, (8, 8, 3), dtype=np.uint8)
        img, label = _chain(1.0).perpendicular_rotate(image, _label())
        assert img.shape == (8, 8, 3)
        assert label.shape == (1, 11)

    @pytest.mark.parametrize("angle", [90, 180, 270])
    def test_each_angle_preserves_shape_and_bounds(self, angle: int) -> None:
        image = np.random.randint(0, 256, (8, 8, 3), dtype=np.uint8)
        with patch("aoi_pcb_ssd.data.augmentation.random.choice", return_value=angle):
            img, label = _chain(1.0).perpendicular_rotate(image, _label())
        assert img.shape == (8, 8, 3)
        coords = label[0, 1:]
        assert coords.min() >= 0
        assert coords.max() <= 8

    def test_90_degree_rotation_maps_corners_exactly(self) -> None:
        # Given: an 8×8 image and a label with corners tl=(2,3), tr=(4,3),
        # bl=(2,7), br=(4,7), centre=(3,5).
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        with patch("aoi_pcb_ssd.data.augmentation.random.choice", return_value=90):
            _, label = _chain(1.0).perpendicular_rotate(image, _label())
        # Then: each (x, y) maps to (y, w − x) with w = 8 (the rotated width),
        # i.e. a 90° rotation of the corner points.
        assert list(label[0, 1:]) == [3, 6, 3, 4, 7, 6, 7, 4, 5, 5]


class TestPhotometricTransforms:
    def test_brightness_changes_pixels(self) -> None:
        image = np.full((4, 4, 3), 100, dtype=np.uint8)
        img, _ = _chain(1.0).random_brightness(image, _label())
        assert not np.array_equal(img, np.full((4, 4, 3), 100, dtype=np.uint8))

    def test_contrast_stays_within_uint8_range(self) -> None:
        image = np.full((4, 4, 3), 200, dtype=np.uint8)
        img, _ = _chain(1.0).random_contrast(image, _label())
        assert img.dtype == np.uint8
        assert img.max() <= 255

    def test_lighting_noise_permutes_channels(self) -> None:
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        image[..., 0], image[..., 1], image[..., 2] = 10, 20, 30
        img, _ = _chain(1.0).random_lighting_noise(image, _label())
        # A channel permutation preserves the multiset of channel values.
        assert sorted(img[0, 0]) == [10, 20, 30]


class TestProbabilityGate:
    @pytest.mark.parametrize("name", DataAugmentationChain._AUGMENTATIONS)
    def test_each_transform_skips_at_zero_probability(self, name: str) -> None:
        image = np.arange(192, dtype=np.uint8).reshape(8, 8, 3)
        label = _label()
        transform = getattr(_chain(0.0), name)
        img, out = transform(image.copy(), label.copy())
        assert np.array_equal(img, image)
        assert np.array_equal(out, label)


class TestCallReproducibility:
    def _dataset(self) -> tuple[np.ndarray, list]:
        images = np.random.randint(0, 256, (5, 8, 8, 3), dtype=np.uint8)
        labels = [_label() for _ in range(5)]
        return images, labels

    def test_same_seed_same_output(self) -> None:
        x1, y1 = self._dataset()
        x2, y2 = x1.copy(), [lbl.copy() for lbl in y1]
        out1, _ = DataAugmentationChain(x1, y1, probability=1.0, seed=7)()
        out2, _ = DataAugmentationChain(x2, y2, probability=1.0, seed=7)()
        assert np.array_equal(out1, out2)

    def test_zero_probability_is_noop(self) -> None:
        images, labels = self._dataset()
        original = images.copy()
        out, _ = DataAugmentationChain(images, labels, probability=0.0, seed=7)()
        assert np.array_equal(out, original)
