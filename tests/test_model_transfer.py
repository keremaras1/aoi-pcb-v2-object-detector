"""Tests for the MobileNetV2 transfer-learning architecture (Figure 4 of the paper)."""

import pytest
import tensorflow as tf

from aoi_pcb_ssd.model import build_transfer_model

_IMAGE_SIZE = (256, 256, 3)
_N_CLASSES = 1
_OUTPUT_SHAPE = (64, 12)


def _backbone(model: tf.keras.Model) -> tf.keras.Model:
    """Return the MobileNetV2 sub-model nested inside the SSD model."""
    return next(
        layer
        for layer in model.layers
        if isinstance(layer, tf.keras.Model) and "mobilenet" in layer.name.lower()
    )


@pytest.fixture(scope="module")
def model() -> tf.keras.Model:
    """Frozen-backbone transfer model with random weights (no ImageNet download)."""
    return build_transfer_model(_IMAGE_SIZE, n_classes=_N_CLASSES, weights=None)


class TestModelOutputShape:
    def test_output_shape(self, model: tf.keras.Model) -> None:
        y = model(tf.random.normal((2, *_IMAGE_SIZE)), training=False)
        assert tuple(y.shape) == (2, *_OUTPUT_SHAPE)

    def test_matches_custom_head_output(self, model: tf.keras.Model) -> None:
        # The detection head is identical to the custom architecture.
        y = model(tf.random.normal((1, *_IMAGE_SIZE)), training=False)
        assert tuple(y.shape) == (1, *_OUTPUT_SHAPE)

    def test_predictor_sizes_is_single_8x8_map(self) -> None:
        _, predictor_sizes = build_transfer_model(
            _IMAGE_SIZE, n_classes=_N_CLASSES, weights=None, return_predictor_sizes=True
        )
        assert predictor_sizes.tolist() == [[8, 8]]


class TestBackboneFreezing:
    def test_backbone_frozen_by_default(self, model: tf.keras.Model) -> None:
        backbone = _backbone(model)
        assert backbone.trainable is False
        assert len(backbone.trainable_weights) == 0
        assert len(backbone.non_trainable_weights) > 0

    def test_detection_head_remains_trainable(self, model: tf.keras.Model) -> None:
        assert len(model.trainable_weights) > 0

    def test_fine_tuned_variant_unfreezes_backbone(self) -> None:
        model = build_transfer_model(
            _IMAGE_SIZE, n_classes=_N_CLASSES, weights=None, trainable_backbone=True
        )
        assert _backbone(model).trainable is True


class TestAdapter:
    def test_adapter_maps_to_512_channels(self, model: tf.keras.Model) -> None:
        assert model.get_layer("conv1").filters == 512


class TestGradientFlow:
    def test_gradients_reach_trainable_weights(self, model: tf.keras.Model) -> None:
        x = tf.random.normal((2, *_IMAGE_SIZE))
        with tf.GradientTape() as tape:
            y = model(x, training=True)
            loss = tf.reduce_mean(tf.square(y))
        grads = tape.gradient(loss, model.trainable_weights)
        assert any(g is not None for g in grads)
