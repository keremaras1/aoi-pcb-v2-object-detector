"""Tests for the GridCenters anchor-centre layer."""

import tensorflow as tf

from aoi_pcb_ssd.model import GridCenters

_IMG = 256
_FEATURE_MAP = (8, 8)


def _input() -> tf.Tensor:
    return tf.zeros((2, _FEATURE_MAP[0], _FEATURE_MAP[1], 16))


class TestOutputShape:
    def test_call_output_shape(self) -> None:
        out = GridCenters(_IMG, _IMG)(_input())
        assert tuple(out.shape) == (2, 8, 8, 1, 2)

    def test_compute_output_shape(self) -> None:
        layer = GridCenters(_IMG, _IMG)
        assert layer.compute_output_shape((None, 8, 8, 16)) == (None, 8, 8, 1, 2)


class TestCoordinateNormalisation:
    def test_normalised_centres_in_unit_range(self) -> None:
        out = GridCenters(_IMG, _IMG, normalize_coords=True)(_input())
        assert float(tf.reduce_min(out)) >= 0.0
        assert float(tf.reduce_max(out)) <= 1.0

    def test_pixel_centres_span_image(self) -> None:
        out = GridCenters(_IMG, _IMG, normalize_coords=False)(_input())
        # Pixel-space centres exceed 1 and stay within the image bounds.
        assert float(tf.reduce_max(out)) > 1.0
        assert float(tf.reduce_max(out)) <= _IMG


class TestChannelsFirst:
    """The layer reads the feature map dimensions correctly under either format."""

    def test_compute_output_shape(self) -> None:
        tf.keras.backend.set_image_data_format("channels_first")
        try:
            layer = GridCenters(_IMG, _IMG)
            assert layer.compute_output_shape((None, 16, 8, 8)) == (None, 8, 8, 1, 2)
        finally:
            tf.keras.backend.set_image_data_format("channels_last")

    def test_call_output_shape(self) -> None:
        tf.keras.backend.set_image_data_format("channels_first")
        try:
            out = GridCenters(_IMG, _IMG)(tf.zeros((2, 16, 8, 8)))
            assert tuple(out.shape) == (2, 8, 8, 1, 2)
        finally:
            tf.keras.backend.set_image_data_format("channels_last")


class TestSerialisation:
    def test_get_config_roundtrip(self) -> None:
        layer = GridCenters(_IMG, 128, normalize_coords=False)
        config = layer.get_config()
        assert config["img_height"] == _IMG
        assert config["img_width"] == 128
        assert config["normalize_coords"] is False

        restored = GridCenters.from_config(config)
        assert restored.img_height == _IMG
        assert restored.img_width == 128
        assert restored.normalize_coords is False
