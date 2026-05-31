"""Tests for the custom SSD architecture (Figure 3 of the paper)."""

import numpy as np
import pytest
import tensorflow as tf

from aoi_pcb_ssd.model import build_custom_model
from aoi_pcb_ssd.model.ssd_custom import ChannelSwap

_IMAGE_SIZE = (256, 256, 3)
_N_CLASSES = 1
_OUTPUT_SHAPE = (64, 12)  # 8x8 grid; 2 class scores + 8 corners + 2 anchor centre


@pytest.fixture(scope="module")
def model() -> tf.keras.Model:
    """Build one custom model shared across the module's tests."""
    return build_custom_model(_IMAGE_SIZE, n_classes=_N_CLASSES, l2_regularization=0.0)


class TestModelOutputShape:
    def test_output_shape(self, model: tf.keras.Model) -> None:
        y = model(tf.random.normal((2, *_IMAGE_SIZE)), training=False)
        assert tuple(y.shape) == (2, *_OUTPUT_SHAPE)

    def test_output_shape_batch_1(self, model: tf.keras.Model) -> None:
        y = model(tf.random.normal((1, *_IMAGE_SIZE)), training=False)
        assert tuple(y.shape) == (1, *_OUTPUT_SHAPE)

    def test_predictor_sizes_is_single_8x8_map(self) -> None:
        _, predictor_sizes = build_custom_model(
            _IMAGE_SIZE, n_classes=_N_CLASSES, return_predictor_sizes=True
        )
        assert predictor_sizes.tolist() == [[8, 8]]


class TestPaperArchitecture:
    """The built graph must reflect the feature extractor described in Figure 3."""

    def test_first_two_blocks_use_wide_kernels(self, model: tf.keras.Model) -> None:
        # Paper Section IV.B: (5x5) kernels in blocks 1-2 for a wider receptive field.
        assert model.get_layer("conv1").kernel_size == (5, 5)
        assert model.get_layer("conv2").kernel_size == (5, 5)

    def test_remaining_blocks_use_small_kernels(self, model: tf.keras.Model) -> None:
        for name in ("conv3", "conv4", "conv5", "conv6"):
            assert model.get_layer(name).kernel_size == (3, 3)

    def test_filter_progression(self, model: tf.keras.Model) -> None:
        expected = {
            "conv1": 32,
            "conv2": 64,
            "conv3": 128,
            "conv4": 256,
            "conv5": 512,
            "conv6": 512,
        }
        for name, filters in expected.items():
            assert model.get_layer(name).filters == filters

    def test_gaussian_noise_std(self, model: tf.keras.Model) -> None:
        noise = next(
            layer for layer in model.layers if isinstance(layer, tf.keras.layers.GaussianNoise)
        )
        assert noise.stddev == 0.1

    def test_classification_branch_is_softmax_normalised(self, model: tf.keras.Model) -> None:
        # The two class scores per cell must sum to 1 (softmax output).
        y = model(tf.random.normal((1, *_IMAGE_SIZE)), training=False)
        class_sums = tf.reduce_sum(y[..., :2], axis=-1)
        assert tf.reduce_all(tf.abs(class_sums - 1.0) < 1e-4)


class TestInputPreprocessing:
    def test_preprocessing_options_build_and_run(self) -> None:
        # Given: the config.json preprocessing options (mean/stddev + swap_channels).
        model = build_custom_model(
            _IMAGE_SIZE,
            n_classes=_N_CLASSES,
            subtract_mean=127.5,
            divide_by_stddev=127.5,
            swap_channels=[2, 1, 0],
        )
        # When: a forward pass is run.
        y = model(tf.random.normal((1, *_IMAGE_SIZE)), training=False)
        # Then: output shape is unchanged — preprocessing does not alter spatial dims.
        assert tuple(y.shape) == (1, *_OUTPUT_SHAPE)

    def test_rescaling_is_numerically_equivalent_to_mean_stddev_formula(self) -> None:
        # Given: model built with subtract_mean=127.5, divide_by_stddev=127.5.
        # When: a uint8-range input is passed through just the Rescaling layer.
        x = tf.constant([[[100.0, 200.0, 50.0]]])  # arbitrary pixel values
        layer = tf.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1.0)
        rescaled = layer(x).numpy()
        # Then: result equals (x - 127.5) / 127.5 exactly.
        expected = (np.array([[[100.0, 200.0, 50.0]]]) - 127.5) / 127.5
        np.testing.assert_allclose(rescaled, expected, atol=1e-6)


class TestChannelSwap:
    def test_swap_permutes_channels(self) -> None:
        # Given: an image where each channel has a distinct constant value.
        image = np.zeros((1, 2, 2, 3), dtype=np.float32)
        image[..., 0], image[..., 1], image[..., 2] = 10.0, 20.0, 30.0
        # When: channels are swapped to BGR order [2, 1, 0].
        out = ChannelSwap(order=[2, 1, 0])(image).numpy()
        # Then: channels appear in reversed order.
        assert out[0, 0, 0, 0] == 30.0
        assert out[0, 0, 0, 1] == 20.0
        assert out[0, 0, 0, 2] == 10.0

    def test_get_config_roundtrip(self) -> None:
        # Given: a ChannelSwap layer with a non-default order.
        layer = ChannelSwap(order=[2, 1, 0], name="swap")
        # When: config is serialised and a new layer is reconstructed.
        restored = ChannelSwap.from_config(layer.get_config())
        # Then: the order is preserved exactly.
        assert restored.order == [2, 1, 0]


class TestSaveLoadRoundTrip:
    """Verify that saved models reload under default safe_mode=True."""

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"subtract_mean": 127.5, "divide_by_stddev": 127.5},
            {"swap_channels": [2, 1, 0]},
        ],
        ids=["no-preprocessing", "mean-stddev", "swap-channels"],
    )
    def test_custom_model_reloads_with_identical_predictions(self, tmp_path, kwargs) -> None:
        # Given: a custom model (small image size to keep the test fast).
        model = build_custom_model((32, 32, 3), n_classes=1, **kwargs)
        x = np.random.rand(2, 32, 32, 3).astype(np.float32)
        # When: the model is saved and reloaded under default safe_mode=True.
        path = str(tmp_path / "model.keras")
        model.save(path)
        loaded = tf.keras.models.load_model(path)  # safe_mode=True by default
        # Then: predictions are bit-identical.
        np.testing.assert_array_equal(model.predict(x, verbose=0), loaded.predict(x, verbose=0))


class TestGradientFlow:
    def test_gradients_reach_trainable_weights(self, model: tf.keras.Model) -> None:
        x = tf.random.normal((2, *_IMAGE_SIZE))
        with tf.GradientTape() as tape:
            y = model(x, training=True)
            loss = tf.reduce_mean(tf.square(y))
        grads = tape.gradient(loss, model.trainable_weights)
        assert any(g is not None for g in grads)
