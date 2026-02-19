import textwrap

import pytest

from src.time_series_model.strategy_config.loader import StrategyConfigLoader
from pathlib import Path


def _write_yaml(path: Path, content: str) -> None:
    path.write_text(textwrap.dedent(content), encoding="utf-8")


def test_strategy_config_loader_success(tmp_path: Path):
    config_dir = tmp_path / "sr_reversal"
    config_dir.mkdir()

    _write_yaml(
        config_dir / "features.yaml",
        """
        name: sr_reversal
        feature_pipeline:
          exclude_columns: [atr]
          requested_features:
            - atr
          post_processors: []
        """,
    )
    _write_yaml(
        config_dir / "labels.yaml",
        """
        target_column: label
        label_generator:
          module: tests.sample_module
          function: fake_label
        """,
    )
    _write_yaml(
        config_dir / "model.yaml",
        """
        trainer:
          module: tests.sample_module
          function: fake_trainer
        """,
    )
    _write_yaml(
        config_dir / "evaluation.yaml",
        """
        evaluation:
          metrics:
            - name: test
              type: correlation
        """,
    )

    loader = StrategyConfigLoader(config_dir)
    config = loader.load()

    assert config.name == "sr_reversal"
    assert config.features.requested_features == ["atr"]
    assert config.features.exclude_columns == ["atr"]
    assert config.labels.target_column == "label"
    assert config.model.trainer.module == "tests.sample_module"
    assert config.evaluation.metrics[0]["name"] == "test"


def test_strategy_config_loader_missing_required(tmp_path: Path):
    config_dir = tmp_path / "bad_strategy"
    config_dir.mkdir()
    # Only write features file
    _write_yaml(
        config_dir / "features.yaml",
        """
        feature_pipeline:
          requested_features: []
        """,
    )

    with pytest.raises(FileNotFoundError):
        StrategyConfigLoader(config_dir).load()
