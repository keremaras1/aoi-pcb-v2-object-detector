"""Tests for the custom SSD architecture (Figure 3 of the paper)."""

import pytest
import tensorflow as tf

from aoi_pcb_ssd.model import build_custom_model

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
        # config.json enables mean subtraction and stddev division; swap_channels
        # exercises the remaining optional preprocessing lambda.
        model = build_custom_model(
            _IMAGE_SIZE,
            n_classes=_N_CLASSES,
            subtract_mean=127.5,
            divide_by_stddev=127.5,
            swap_channels=[2, 1, 0],
        )
        y = model(tf.random.normal((1, *_IMAGE_SIZE)), training=False)
        assert tuple(y.shape) == (1, *_OUTPUT_SHAPE)


class TestGradientFlow:
    def test_gradients_reach_trainable_weights(self, model: tf.keras.Model) -> None:
        x = tf.random.normal((2, *_IMAGE_SIZE))
        with tf.GradientTape() as tape:
            y = model(x, training=True)
            loss = tf.reduce_mean(tf.square(y))
        grads = tape.gradient(loss, model.trainable_weights)
        assert any(g is not None for g in grads)
