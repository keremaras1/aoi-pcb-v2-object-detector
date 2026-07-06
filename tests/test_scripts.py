# SPDX-License-Identifier: Apache-2.0
"""Wiring tests for the train and evaluate CLI scripts.

These tests verify that each script's ``main()`` correctly wires together its
collaborators — argument parsing, config loading, data pipeline, model
building, compile/fit/save — without running real I/O or training.
All heavy collaborators are mocked.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from aoi_pcb_ssd.config_loader import Config
from aoi_pcb_ssd.model.loss import AOILoss
from aoi_pcb_ssd.model.metrics import class_map, mae

# ---------------------------------------------------------------------------
# Module-level fixtures
# ---------------------------------------------------------------------------

_CONFIG_DICT = {
    "generator": {
        "templates_dir": "templates",
        "train_data": {
            "dataset_size": 4,
            "rotation_range": 2,
            "placement_offset_x": 2,
            "placement_offset_y": 2,
            "img_size": 32,
            "save_dir": "datasets/generated",
            "crop_save_dir": "datasets/train",
            "seed": 42,
        },
        "val_data": {
            "dataset_size": 2,
            "rotation_range": 2,
            "placement_offset_x": 2,
            "placement_offset_y": 2,
            "img_size": 32,
            "save_dir": "datasets/generated",
            "crop_save_dir": "datasets/val",
            "seed": 7,
        },
    },
    "model": {
        "img_height": 32,
        "img_width": 32,
        "img_channels": 3,
        "n_classes": 1,
        "normalize_coords": True,
        "subtract_mean": 127.5,
        "divide_by_stddev": 127.5,
        "l2_regularization": 0.0,
        "predictor_sizes": [[8, 8]],
    },
    "training": {
        "batch_size": 2,
        "epochs": 1,
        "val_split": 0.25,
        "random_seed": 0,
        "optimizer": {"learning_rate": 0.001},
        "loss": {"neg_pos_ratio": 3, "alpha": 3.0},
        "early_stopping": {
            "monitor": "val_loss",
            "min_delta": 0.0,
            "patience": 1,
            "verbose": 0,
            "restore_best_weights": False,
        },
        "lr_schedule": {
            "monitor": "val_loss",
            "factor": 0.5,
            "patience": 1,
            "verbose": 0,
            "min_delta": 0.001,
            "cooldown": 0,
            "min_lr": 1e-7,
        },
        "performance": {"jit_compile": False, "mixed_float16": False},
    },
    "augmentation": {"enabled": False, "probability": 0.0},
}


@pytest.fixture
def fake_cfg() -> Config:
    return Config.from_dict(_CONFIG_DICT)


@pytest.fixture
def fake_data() -> tuple[np.ndarray, np.ndarray]:
    X = np.zeros((4, 32, 32, 3), dtype=np.uint8)
    y = np.zeros((4, 64, 12), dtype=np.float32)
    return X, y


@pytest.fixture
def mock_model() -> MagicMock:
    m = MagicMock()
    m.evaluate.return_value = {"loss": 0.5, "class_map": 0.9, "mae": 1.2}
    m.predict.return_value = np.zeros((1, 64, 12), dtype=np.float32)
    m.fit.return_value = MagicMock(history={"loss": [0.5], "val_loss": [0.6]})
    return m


# ---------------------------------------------------------------------------
# train.py — argument parsing
# ---------------------------------------------------------------------------


class TestTrainArgParsing:
    def test_architecture_required(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Given: sys.argv missing --architecture.
        monkeypatch.setattr(sys, "argv", ["train.py"])
        import scripts.train as train_script

        # When/Then: argparse exits with an error.
        with pytest.raises(SystemExit):
            train_script.parse_args()

    def test_custom_architecture_parsed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Given: --architecture custom on the command line.
        monkeypatch.setattr(sys, "argv", ["train.py", "--architecture", "custom"])
        import scripts.train as train_script

        # When: parsed.
        args = train_script.parse_args()
        # Then: architecture is "custom".
        assert args.architecture == "custom"

    def test_output_dir_defaults_to_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Given: no --output-dir argument.
        monkeypatch.setattr(sys, "argv", ["train.py", "--architecture", "custom"])
        import scripts.train as train_script

        args = train_script.parse_args()
        # Then: output_dir is None (resolved to a timestamped path inside main()).
        assert args.output_dir is None


# ---------------------------------------------------------------------------
# train.py — main() wiring
# ---------------------------------------------------------------------------


class TestTrainMainWiring:
    """All tests share a common patched environment set up in ``_setup``.

    ``_setup`` is pytest's equivalent of ``@BeforeEach``: it runs before every
    test method in this class, applies patches, and yields so that each test
    body runs while those patches are active.
    """

    @pytest.fixture(autouse=True)
    def _setup(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        fake_cfg: Config,
        fake_data: tuple,
        mock_model: MagicMock,
    ) -> None:
        import scripts.train as train_script

        self.train_script = train_script
        self.mock_model = mock_model
        X, y = fake_data
        self.X, self.y = X, y
        self.output_dir = tmp_path / "run"

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "train.py",
                "--architecture",
                "custom",
                "--config",
                "config.json",
                "--output-dir",
                str(self.output_dir),
            ],
        )

        with (
            patch.object(train_script, "Config", return_value=fake_cfg),
            patch("shutil.copy") as self._copy,
            patch.object(train_script, "DataGenerator") as self._data_gen,
            patch.object(train_script, "train_test_split", return_value=(X, X, y, y)),
            patch.object(
                train_script, "build_custom_model", return_value=mock_model
            ) as self._build_custom,
            patch.object(
                train_script, "build_transfer_model", return_value=mock_model
            ) as self._build_transfer,
        ):
            self._data_gen.return_value.get_data.return_value = (X, y)
            yield

    def test_output_dir_created(self) -> None:
        # Given: --output-dir points to a non-existent path.
        # When: main() runs.
        self.train_script.main()
        # Then: the directory is created.
        assert self.output_dir.is_dir()

    def test_config_copied_to_output_dir(self) -> None:
        # Given: a config path and output dir.
        # When: main() runs.
        self.train_script.main()
        # Then: shutil.copy is called with the config path and output dir destination.
        self._copy.assert_called_once_with("config.json", self.output_dir / "config.json")

    def test_compile_receives_aoi_loss_and_metrics(self) -> None:
        # Given: a mocked model.
        # When: main() runs.
        self.train_script.main()
        compile_kwargs = self.mock_model.compile.call_args.kwargs
        # Then: loss is an AOILoss bound method; metrics include class_map and mae.
        assert isinstance(compile_kwargs["loss"].__self__, AOILoss)
        assert class_map in compile_kwargs["metrics"]
        assert mae in compile_kwargs["metrics"]

    def test_model_saved_inside_output_dir(self) -> None:
        # Given: an output dir.
        # When: main() runs.
        self.train_script.main()
        (save_path,) = self.mock_model.save.call_args.args
        # Then: the model is saved as model.keras inside the output dir.
        assert Path(save_path).parent == self.output_dir
        assert save_path.endswith("model.keras")

    def test_transfer_architecture_calls_build_transfer_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Given: --architecture transfer (override the default "custom" argv set by _setup).
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "train.py",
                "--architecture",
                "transfer",
                "--config",
                "config.json",
                "--output-dir",
                str(self.output_dir),
            ],
        )
        # When: main() runs.
        self.train_script.main()
        # Then: only build_transfer_model is called.
        self._build_transfer.assert_called_once()
        self._build_custom.assert_not_called()

    def test_jit_compile_defaults_to_false(self) -> None:
        # Given: a config with performance.jit_compile = False.
        # When: main() runs.
        self.train_script.main()
        # Then: compile receives jit_compile=False (the Apple-Silicon-safe default).
        assert self.mock_model.compile.call_args.kwargs["jit_compile"] is False

    def test_jit_compile_forwarded_when_enabled(self, fake_cfg: Config) -> None:
        # Given: performance.jit_compile enabled in the config.
        fake_cfg.training.performance.jit_compile = True
        # When: main() runs.
        self.train_script.main()
        # Then: compile receives jit_compile=True.
        assert self.mock_model.compile.call_args.kwargs["jit_compile"] is True

    def test_data_generator_uses_cache_by_default(self) -> None:
        # Given: --no-cache not passed (default argv from _setup).
        # When: main() runs.
        self.train_script.main()
        # Then: the DataGenerator is constructed with caching enabled.
        assert self._data_gen.call_args.kwargs["use_cache"] is True

    def test_no_cache_flag_disables_dataset_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Given: --no-cache passed.
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "train.py",
                "--architecture",
                "custom",
                "--config",
                "config.json",
                "--output-dir",
                str(self.output_dir),
                "--no-cache",
            ],
        )
        # When: main() runs.
        self.train_script.main()
        # Then: the DataGenerator is constructed with caching disabled.
        assert self._data_gen.call_args.kwargs["use_cache"] is False

    def test_mixed_float16_sets_global_policy_when_enabled(
        self, fake_cfg: Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Given: performance.mixed_float16 enabled.
        fake_cfg.training.performance.mixed_float16 = True
        recorded: dict[str, str] = {}
        monkeypatch.setattr(
            self.train_script.tf.keras.mixed_precision,
            "set_global_policy",
            lambda policy: recorded.setdefault("policy", policy),
        )
        # When: main() runs.
        self.train_script.main()
        # Then: the mixed_float16 policy is activated.
        assert recorded.get("policy") == "mixed_float16"


# ---------------------------------------------------------------------------
# evaluate.py — argument parsing
# ---------------------------------------------------------------------------


class TestEvaluateArgParsing:
    def test_model_path_defaults_to_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Given: no --model-path argument.
        monkeypatch.setattr(sys, "argv", ["evaluate.py"])
        import scripts.evaluate as eval_script

        args = eval_script.parse_args()
        # Then: model_path is None (resolved to the latest run inside main()).
        assert args.model_path is None

    def test_save_visuals_defaults_to_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Given: --save-visuals not set.
        monkeypatch.setattr(sys, "argv", ["evaluate.py"])
        import scripts.evaluate as eval_script

        args = eval_script.parse_args()
        # Then: save_visuals is False.
        assert args.save_visuals is False

    def test_save_visuals_set_when_flag_passed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Given: --save-visuals passed.
        monkeypatch.setattr(sys, "argv", ["evaluate.py", "--save-visuals"])
        import scripts.evaluate as eval_script

        args = eval_script.parse_args()
        # Then: save_visuals is True.
        assert args.save_visuals is True


# ---------------------------------------------------------------------------
# evaluate.py — main() wiring
# ---------------------------------------------------------------------------


class TestEvaluateMainWiring:
    """All tests share a common patched environment set up in ``_setup``."""

    @pytest.fixture(autouse=True)
    def _setup(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        fake_cfg: Config,
        fake_data: tuple,
        mock_model: MagicMock,
    ) -> None:
        import scripts.evaluate as eval_script

        self.eval_script = eval_script
        self.mock_model = mock_model
        X, y = fake_data
        self.X, self.y = X, y

        model_file = tmp_path / "model.keras"
        model_file.touch()
        self.model_file = model_file

        monkeypatch.setattr(
            sys,
            "argv",
            ["evaluate.py", "--model-path", str(model_file), "--config", "config.json"],
        )

        X_val, y_val = X[:1], y[:1]

        with (
            patch.object(eval_script, "Config", return_value=fake_cfg),
            patch.object(eval_script, "DataGenerator") as self._data_gen,
            patch.object(
                eval_script,
                "train_test_split",
                return_value=(X[1:], X_val, y[1:], y_val),
            ),
            patch(
                "tensorflow.keras.models.load_model", return_value=mock_model
            ) as self._load_model,
        ):
            self._data_gen.return_value.get_data.return_value = (X, y)
            yield

    def test_load_model_called_with_compile_false(self) -> None:
        # Given: a saved model file.
        # When: main() runs.
        self.eval_script.main()
        # Then: load_model is called with compile=False so the custom loss is not deserialised.
        _, load_kwargs = self._load_model.call_args
        assert load_kwargs.get("compile") is False

    def test_compile_receives_aoi_loss_and_metrics(self) -> None:
        # Given: a loaded model.
        # When: main() runs.
        self.eval_script.main()
        compile_kwargs = self.mock_model.compile.call_args.kwargs
        # Then: AOILoss and both metrics are passed to compile.
        assert isinstance(compile_kwargs["loss"].__self__, AOILoss)
        assert class_map in compile_kwargs["metrics"]
        assert mae in compile_kwargs["metrics"]

    def test_data_generator_uses_cache_by_default(self) -> None:
        # Given: --no-cache not passed (default argv from _setup).
        # When: main() runs.
        self.eval_script.main()
        # Then: the DataGenerator is constructed with caching enabled.
        assert self._data_gen.call_args.kwargs["use_cache"] is True

    def test_no_cache_flag_disables_dataset_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Given: --no-cache passed.
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "evaluate.py",
                "--model-path",
                str(self.model_file),
                "--config",
                "config.json",
                "--no-cache",
            ],
        )
        # When: main() runs.
        self.eval_script.main()
        # Then: the DataGenerator is constructed with caching disabled.
        assert self._data_gen.call_args.kwargs["use_cache"] is False

    def test_save_visuals_calls_draw_keypoints(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Given: --save-visuals flag (override the default argv set by _setup).
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "evaluate.py",
                "--model-path",
                str(self.model_file),
                "--config",
                "config.json",
                "--save-visuals",
            ],
        )
        # When: main() runs with draw_keypoints and cv2.imwrite mocked out.
        with (
            patch.object(self.eval_script, "decode_detections", return_value=[np.zeros((0, 10))]),
            patch.object(
                self.eval_script,
                "draw_keypoints",
                return_value=np.zeros((32, 32, 3), dtype=np.uint8),
            ) as mock_draw,
            patch("cv2.imwrite"),
        ):
            self.eval_script.main()
        # Then: draw_keypoints is called once (1 validation sample, clamped to n_visuals=15).
        assert mock_draw.call_count == 1

    def test_no_save_visuals_skips_draw_keypoints(self) -> None:
        # Given: --save-visuals NOT passed (default argv from _setup).
        # When: main() runs.
        with patch.object(self.eval_script, "draw_keypoints") as mock_draw:
            self.eval_script.main()
        # Then: draw_keypoints is never called.
        mock_draw.assert_not_called()
