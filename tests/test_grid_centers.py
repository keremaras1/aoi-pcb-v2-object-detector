"""Tests for the GridCenters anchor-centre layer."""

import numpy as np
import pytest
import tensorflow as tf

from aoi_pcb_ssd.model import GridCenters

_IMG = 256
_FEATURE_MAP = (8, 8)


def _input() -> tf.Tensor:
    return tf.zeros((2, _FEATURE_MAP[0], _FEATURE_MAP[1], 16))


class TestOutputShape:
    def test_call_output_shape(self) -> None:
        # Given/When: a 2-item batch passed through GridCenters.
        out = GridCenters(_IMG, _IMG)(_input())
        # Then: output has the expected (batch, H, W, 1, 2) layout.
        assert tuple(out.shape) == (2, 8, 8, 1, 2)

    def test_compute_output_shape(self) -> None:
        # Given: a GridCenters layer (not yet built).
        layer = GridCenters(_IMG, _IMG)
        # Then: compute_output_shape returns the correct symbolic shape.
        assert layer.compute_output_shape((None, 8, 8, 16)) == (None, 8, 8, 1, 2)


class TestCoordinateNormalisation:
    def test_normalised_centres_in_unit_range(self) -> None:
        # Given/When: normalised grid centres for a 256×256 / 8×8 geometry.
        out = GridCenters(_IMG, _IMG, normalize_coords=True)(_input())
        # Then: all values are in [0, 1].
        assert float(tf.reduce_min(out)) >= 0.0
        assert float(tf.reduce_max(out)) <= 1.0

    def test_pixel_centres_span_image(self) -> None:
        # Given/When: pixel-space grid centres.
        out = GridCenters(_IMG, _IMG, normalize_coords=False)(_input())
        # Then: values exceed 1 (not normalised) but stay within the image.
        assert float(tf.reduce_max(out)) > 1.0
        assert float(tf.reduce_max(out)) <= _IMG


class TestPrecomputedGrid:
    def test_centers_attribute_set_after_build(self) -> None:
        # Given: a GridCenters layer.
        layer = GridCenters(_IMG, _IMG)
        # When: the layer is called (triggering build).
        layer(_input())
        # Then: _centers is a constant of shape (1, H, W, 1, 2) — one per feature cell.
        assert hasattr(layer, "_centers")
        assert tuple(layer._centers.shape) == (1, *_FEATURE_MAP, 1, 2)

    def test_first_cell_centre_pixel_value(self) -> None:
        # Given: a 256×256 image with an 8×8 feature map, pixel-space coords.
        # step = 256 / 8 = 32 px; first centre = step × 0.5 = 16 px.
        layer = GridCenters(_IMG, _IMG, normalize_coords=False)
        out = layer(_input())
        # When: the top-left cell centre is read.
        cx = float(out[0, 0, 0, 0, 0])
        cy = float(out[0, 0, 0, 0, 1])
        # Then: both x and y are 16 pixels from the image origin.
        assert cx == pytest.approx(16.0)
        assert cy == pytest.approx(16.0)

    def test_last_cell_centre_pixel_value(self) -> None:
        # Given: same 256×256 / 8×8 geometry.
        # Last centre = step × (8 − 0.5) = 32 × 7.5 = 240 px.
        layer = GridCenters(_IMG, _IMG, normalize_coords=False)
        out = layer(_input())
        # When: the bottom-right cell centre is read.
        cx = float(out[0, 7, 7, 0, 0])
        cy = float(out[0, 7, 7, 0, 1])
        # Then: both x and y are 240 pixels from the image origin.
        assert cx == pytest.approx(240.0)
        assert cy == pytest.approx(240.0)


class TestChannelsFirst:
    """The layer reads the feature map dimensions correctly under either format."""

    def test_compute_output_shape(self) -> None:
        # Given: channels-first data format is active.
        tf.keras.backend.set_image_data_format("channels_first")
        try:
            layer = GridCenters(_IMG, _IMG)
            # When: compute_output_shape is called with a channels-first shape.
            result = layer.compute_output_shape((None, 16, 8, 8))
        finally:
            tf.keras.backend.set_image_data_format("channels_last")
        # Then: output shape is (batch, H, W, 1, 2) — layout is always channels-last.
        assert result == (None, 8, 8, 1, 2)

    def test_call_output_shape(self) -> None:
        # Given: channels-first data format.
        tf.keras.backend.set_image_data_format("channels_first")
        try:
            # When: a channels-first input (B, C, H, W) is passed through.
            out = GridCenters(_IMG, _IMG)(tf.zeros((2, 16, 8, 8)))
        finally:
            tf.keras.backend.set_image_data_format("channels_last")
        # Then: output shape is identical to the channels-last case.
        assert tuple(out.shape) == (2, 8, 8, 1, 2)

    def test_channels_first_matches_channels_last_values(self) -> None:
        # Given: a reference output produced under the default channels-last format.
        ref = GridCenters(_IMG, _IMG, normalize_coords=True)(tf.zeros((1, 8, 8, 4)))
        # When: the same geometry is computed under channels-first format.
        tf.keras.backend.set_image_data_format("channels_first")
        try:
            under_cf = GridCenters(_IMG, _IMG, normalize_coords=True)(tf.zeros((1, 4, 8, 8)))
        finally:
            tf.keras.backend.set_image_data_format("channels_last")
        # Then: grid-centre coordinates are numerically identical.
        np.testing.assert_allclose(ref.numpy(), under_cf.numpy(), atol=1e-6)


class TestSerialisation:
    def test_get_config_roundtrip(self) -> None:
        # Given: a GridCenters layer with non-default settings.
        layer = GridCenters(_IMG, 128, normalize_coords=False)
        # When: config is serialised and a new layer is reconstructed.
        config = layer.get_config()
        restored = GridCenters.from_config(config)
        # Then: all constructor arguments are preserved exactly.
        assert restored.img_height == _IMG
        assert restored.img_width == 128
        assert restored.normalize_coords is False
