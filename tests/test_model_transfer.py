"""Tests for the MobileNetV2 transfer-learning architecture (Figure 4 of the paper)."""

import numpy as np
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
        # Given: the pre-built transfer model (frozen backbone, random weights).
        # When: a batch of 2 images is passed through.
        y = model(tf.random.normal((2, *_IMAGE_SIZE)), training=False)
        # Then: output is (batch, 64, 12) — same layout as the custom architecture.
        assert tuple(y.shape) == (2, *_OUTPUT_SHAPE)

    def test_matches_custom_head_output(self, model: tf.keras.Model) -> None:
        # Given: the transfer model.
        # When: a single image is passed through.
        y = model(tf.random.normal((1, *_IMAGE_SIZE)), training=False)
        # Then: the detection head (identical to Figure 3) produces (64, 12) per image.
        assert tuple(y.shape) == (1, *_OUTPUT_SHAPE)

    def test_predictor_sizes_is_single_8x8_map(self) -> None:
        # Given/When: the model is built with return_predictor_sizes=True.
        _, predictor_sizes = build_transfer_model(
            _IMAGE_SIZE, n_classes=_N_CLASSES, weights=None, return_predictor_sizes=True
        )
        # Then: one 8×8 predictor layer (MobileNetV2 output at 1/32 resolution).
        assert predictor_sizes.tolist() == [[8, 8]]


class TestBackboneFreezing:
    def test_backbone_frozen_by_default(self, model: tf.keras.Model) -> None:
        # Given: the default frozen-backbone transfer model.
        backbone = _backbone(model)
        # Then: the MobileNetV2 sub-model has no trainable weights.
        assert backbone.trainable is False
        assert len(backbone.trainable_weights) == 0
        assert len(backbone.non_trainable_weights) > 0

    def test_detection_head_remains_trainable(self, model: tf.keras.Model) -> None:
        # Given: a frozen-backbone model.
        # Then: the detection head adds trainable weights on top of the frozen backbone.
        assert len(model.trainable_weights) > 0

    def test_fine_tuned_variant_unfreezes_backbone(self) -> None:
        # Given: a model built with trainable_backbone=True.
        model = build_transfer_model(
            _IMAGE_SIZE, n_classes=_N_CLASSES, weights=None, trainable_backbone=True
        )
        # Then: the MobileNetV2 sub-model is trainable.
        assert _backbone(model).trainable is True


class TestAdapter:
    def test_adapter_maps_to_512_channels(self, model: tf.keras.Model) -> None:
        # Given: the transfer model.
        # Then: the adapter conv maps MobileNetV2's output to 512 channels
        # so the detection head matches the custom architecture.
        assert model.get_layer("conv1").filters == 512


class TestGradientFlow:
    def test_gradients_reach_trainable_weights(self, model: tf.keras.Model) -> None:
        # Given: the transfer model in training mode (head unfrozen, backbone frozen).
        x = tf.random.normal((2, *_IMAGE_SIZE))
        # When: a forward pass and mock loss are computed.
        with tf.GradientTape() as tape:
            y = model(x, training=True)
            loss = tf.reduce_mean(tf.square(y))
        grads = tape.gradient(loss, model.trainable_weights)
        # Then: at least one head weight receives a gradient.
        assert any(g is not None for g in grads)


class TestSaveLoadRoundTrip:
    def test_reloads_under_default_safe_mode(self, tmp_path) -> None:
        # Given: a transfer model with random weights (no ImageNet download).
        model = build_transfer_model((32, 32, 3), n_classes=1, weights=None)
        x = np.random.rand(2, 32, 32, 3).astype(np.float32)
        # When: the model is saved and reloaded under default safe_mode=True.
        path = str(tmp_path / "model.keras")
        model.save(path)
        loaded = tf.keras.models.load_model(path)
        # Then: predictions are bit-identical.
        np.testing.assert_array_equal(model.predict(x, verbose=0), loaded.predict(x, verbose=0))
