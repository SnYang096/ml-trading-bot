"""Loader for per-strategy configuration directories."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def _load_yaml_file(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data


@dataclass
class ModuleFunctionConfig:
    module: str
    function: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeaturePipelineConfig:
    requested_features: List[str] = field(default_factory=list)
    post_processors: List[ModuleFunctionConfig] = field(default_factory=list)
    selector: Optional[ModuleFunctionConfig] = None
    ensure_signal: Optional[Dict[str, Any]] = None


@dataclass
class LabelConfig:
    target_column: str
    generator: ModuleFunctionConfig
    filters: List[Dict[str, Any]] = field(default_factory=list)
    post_label_filters: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ModelConfig:
    trainer: ModuleFunctionConfig
    prediction: Dict[str, Any] = field(default_factory=dict)
    output: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationConfig:
    metrics: List[Dict[str, Any]] = field(default_factory=list)
    save_details: bool = False
    results_file: Optional[str] = None


@dataclass
class BacktestConfig:
    enabled: bool = False
    class_path: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyConfig:
    name: str
    path: Path
    features: FeaturePipelineConfig
    labels: LabelConfig
    model: ModelConfig
    evaluation: EvaluationConfig
    backtest: BacktestConfig
    meta: Dict[str, Any] = field(default_factory=dict)


class StrategyConfigLoader:
    """Load strategy configuration files from a directory."""

    REQUIRED_FILES = ("features.yaml", "labels.yaml", "model.yaml")
    OPTIONAL_FILES = ("evaluation.yaml", "backtest.yaml", "meta.yaml")

    def __init__(self, config_dir: Path | str) -> None:
        self.config_dir = Path(config_dir)
        if not self.config_dir.exists() or not self.config_dir.is_dir():
            raise FileNotFoundError(f"Config directory not found: {self.config_dir}")

    def load(self) -> StrategyConfig:
        missing_required = [
            fname
            for fname in self.REQUIRED_FILES
            if not (self.config_dir / fname).exists()
        ]
        if missing_required:
            raise FileNotFoundError(
                f"Config directory {self.config_dir} is missing required files: {missing_required}"
            )

        missing_optional = [
            fname
            for fname in self.OPTIONAL_FILES
            if not (self.config_dir / fname).exists()
        ]
        for fname in missing_optional:
            print(f"   ⚠️  Optional config '{fname}' not found in {self.config_dir}")

        features_data = _load_yaml_file(self.config_dir / "features.yaml")
        labels_data = _load_yaml_file(self.config_dir / "labels.yaml")
        model_data = _load_yaml_file(self.config_dir / "model.yaml")

        evaluation_data = (
            _load_yaml_file(self.config_dir / "evaluation.yaml")
            if (self.config_dir / "evaluation.yaml").exists()
            else {}
        )
        backtest_data = (
            _load_yaml_file(self.config_dir / "backtest.yaml")
            if (self.config_dir / "backtest.yaml").exists()
            else {}
        )
        meta_data = (
            _load_yaml_file(self.config_dir / "meta.yaml")
            if (self.config_dir / "meta.yaml").exists()
            else {}
        )

        name = (
            features_data.get("name")
            or labels_data.get("name")
            or model_data.get("name")
            or self.config_dir.name
        )

        feature_cfg = self._parse_feature_config(features_data)
        label_cfg = self._parse_label_config(labels_data)
        model_cfg = self._parse_model_config(model_data)
        evaluation_cfg = self._parse_evaluation_config(evaluation_data)
        backtest_cfg = self._parse_backtest_config(backtest_data)

        return StrategyConfig(
            name=name,
            path=self.config_dir,
            features=feature_cfg,
            labels=label_cfg,
            model=model_cfg,
            evaluation=evaluation_cfg,
            backtest=backtest_cfg,
            meta=meta_data.get("strategy", meta_data),
        )

    def _parse_feature_config(self, data: Dict[str, Any]) -> FeaturePipelineConfig:
        pipeline = data.get("feature_pipeline", {})
        requested = pipeline.get("requested_features", []) or []
        post_processors = [
            self._parse_module_function(entry)
            for entry in pipeline.get("post_processors", []) or []
        ]
        selector = (
            self._parse_module_function(pipeline["selector"])
            if pipeline.get("selector")
            else None
        )
        ensure_signal = pipeline.get("ensure_signal_column")
        return FeaturePipelineConfig(
            requested_features=requested,
            post_processors=post_processors,
            selector=selector,
            ensure_signal=ensure_signal,
        )

    def _parse_label_config(self, data: Dict[str, Any]) -> LabelConfig:
        target = data.get("target_column", "label")
        generator_cfg = data.get("label_generator") or data.get("generator") or {}
        generator = self._parse_module_function(generator_cfg)
        filters = data.get("filters", []) or []
        post_filters = data.get("post_label_filters", []) or []
        return LabelConfig(
            target_column=target,
            generator=generator,
            filters=filters,
            post_label_filters=post_filters,
        )

    def _parse_model_config(self, data: Dict[str, Any]) -> ModelConfig:
        trainer = self._parse_module_function(data.get("trainer", {}))
        prediction = data.get("prediction", {})
        output = data.get("output", {})
        return ModelConfig(trainer=trainer, prediction=prediction, output=output)

    def _parse_evaluation_config(self, data: Dict[str, Any]) -> EvaluationConfig:
        evaluation = data.get("evaluation", {})
        metrics = evaluation.get("metrics", []) or []
        save_details = evaluation.get("save_details", False)
        results_file = evaluation.get("results_file")
        return EvaluationConfig(
            metrics=metrics, save_details=save_details, results_file=results_file
        )

    def _parse_backtest_config(self, data: Dict[str, Any]) -> BacktestConfig:
        backtest = data.get("backtest", {})
        if not backtest:
            return BacktestConfig()
        return BacktestConfig(
            enabled=backtest.get("enabled", False),
            class_path=backtest.get("class"),
            params=backtest.get("params", {}),
        )

    def _parse_module_function(
        self, entry: Optional[Dict[str, Any]]
    ) -> ModuleFunctionConfig:
        if not entry:
            raise ValueError("Module/function configuration is missing")
        module = entry.get("module")
        function = entry.get("function")
        if not module or not function:
            raise ValueError(f"Incomplete module/function config: {entry}")
        params = entry.get("params", {}) or {}
        return ModuleFunctionConfig(module=module, function=function, params=params)
