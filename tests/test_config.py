"""Tests for the Config loader."""

import json
from pathlib import Path

import pytest

from aoi_pcb_ssd.config_loader import Config

_REAL_CONFIG = Path(__file__).parent.parent / "config.json"


@pytest.fixture
def sample_config(tmp_path: Path) -> Path:
    """Write a minimal config JSON to a temp file and return its path."""
    data = {
        "training": {
            "learning_rate": 0.001,
            "epochs": 100,
            "early_stopping": {
                "patience": 10,
                "restore_best_weights": True,
            },
        },
        "loss": {
            "alpha": 3.0,
            "neg_pos_ratio": 3,
        },
        "flat_key": "flat_value",
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(data))
    return config_path


class TestConfigLoading:
    def test_loads_flat_value(self, sample_config: Path) -> None:
        config = Config(str(sample_config))
        assert config.flat_key == "flat_value"

    def test_loads_nested_attribute(self, sample_config: Path) -> None:
        config = Config(str(sample_config))
        assert config.training.learning_rate == 0.001
        assert config.training.epochs == 100

    def test_loads_deeply_nested_attribute(self, sample_config: Path) -> None:
        config = Config(str(sample_config))
        assert config.training.early_stopping.patience == 10
        assert config.training.early_stopping.restore_best_weights is True

    def test_from_dict(self) -> None:
        config = Config.from_dict({"a": 1, "b": {"c": 2}})
        assert config.a == 1
        assert config.b.c == 2

    def test_missing_attribute_raises(self, sample_config: Path) -> None:
        config = Config(str(sample_config))
        with pytest.raises(AttributeError, match="no attribute"):
            _ = config.nonexistent


class TestGetInitKwargs:
    def test_single_level_key(self, sample_config: Path) -> None:
        config = Config(str(sample_config))
        kwargs = config.get_init_kwargs("loss")
        assert kwargs["alpha"] == 3.0
        assert kwargs["neg_pos_ratio"] == 3

    def test_dot_notation_nested_key(self, sample_config: Path) -> None:
        config = Config(str(sample_config))
        kwargs = config.get_init_kwargs("training.early_stopping")
        assert kwargs["patience"] == 10
        assert kwargs["restore_best_weights"] is True

    def test_returns_plain_dict(self, sample_config: Path) -> None:
        config = Config(str(sample_config))
        kwargs = config.get_init_kwargs("training.early_stopping")
        assert isinstance(kwargs, dict)
        assert not isinstance(kwargs, Config)

    def test_raises_for_missing_key(self, sample_config: Path) -> None:
        config = Config(str(sample_config))
        with pytest.raises(ValueError, match="not found"):
            config.get_init_kwargs("nonexistent")

    def test_raises_for_missing_nested_key(self, sample_config: Path) -> None:
        config = Config(str(sample_config))
        with pytest.raises(ValueError, match="not found"):
            config.get_init_kwargs("training.nonexistent")

    def test_raises_when_key_points_to_scalar(self, sample_config: Path) -> None:
        config = Config(str(sample_config))
        with pytest.raises(ValueError, match="scalar"):
            config.get_init_kwargs("flat_key")


class TestToDict:
    def test_roundtrip_through_from_dict(self) -> None:
        original = {"a": 1, "b": {"c": 2, "d": {"e": 3}}}
        config = Config.from_dict(original)
        assert config.to_dict() == original

    def test_nested_sections_become_plain_dicts(self) -> None:
        config = Config.from_dict({"outer": {"inner": 1}})
        result = config.to_dict()
        assert isinstance(result["outer"], dict)
        assert not isinstance(result["outer"], Config)


class TestProjectConfig:
    """Verify the published config.json loads and exposes the structure the
    implementation depends on."""

    def test_real_config_loads(self) -> None:
        config = Config(str(_REAL_CONFIG))
        assert hasattr(config, "generator")
        assert hasattr(config, "model")
        assert hasattr(config, "training")
        assert hasattr(config, "augmentation")

    def test_model_section_is_complete(self) -> None:
        # Keys consumed by the model and encoder builders.
        config = Config(str(_REAL_CONFIG))
        kwargs = config.get_init_kwargs("model")
        for key in (
            "img_height",
            "img_width",
            "img_channels",
            "n_classes",
            "normalize_coords",
            "predictor_sizes",
        ):
            assert key in kwargs

    def test_predictor_sizes_shape(self) -> None:
        # predictor_sizes is a list of [height, width] pairs, one per predictor layer.
        config = Config(str(_REAL_CONFIG))
        predictor_sizes = config.model.predictor_sizes
        assert isinstance(predictor_sizes, list)
        assert all(len(pair) == 2 for pair in predictor_sizes)

    def test_square_input_geometry(self) -> None:
        # The custom feature extractor expects a square input image.
        config = Config(str(_REAL_CONFIG))
        assert config.model.img_height == config.model.img_width

    def test_training_section_is_complete(self) -> None:
        # Keys consumed by the training script and its callbacks.
        config = Config(str(_REAL_CONFIG))
        kwargs = config.get_init_kwargs("training")
        for key in ("batch_size", "epochs", "val_split", "optimizer", "loss"):
            assert key in kwargs

    def test_callback_sections_unpack_as_kwargs(self) -> None:
        # early_stopping and lr_schedule are unpacked into Keras callbacks.
        config = Config(str(_REAL_CONFIG))
        assert isinstance(config.get_init_kwargs("training.early_stopping"), dict)
        assert isinstance(config.get_init_kwargs("training.lr_schedule"), dict)

    def test_loss_section_provides_aoiloss_arguments(self) -> None:
        # AOILoss is constructed from this section.
        config = Config(str(_REAL_CONFIG))
        kwargs = config.get_init_kwargs("training.loss")
        assert "neg_pos_ratio" in kwargs
        assert "alpha" in kwargs
