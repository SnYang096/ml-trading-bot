"""测试 IncrementalFeatureComputer 的 second pass 逻辑。

覆盖 commit 1f81030 的修复:
1. second pass 准入检查只看 required_columns（不合并 column_mappings）
2. column_mappings 过滤：只保留 bars_tf 中实际存在的列
3. 3a-fix：skipped 节点的非-tick 依赖加入 first pass
"""

import inspect
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures: 模拟 feature_dependencies.yaml 中的节点配置
# ---------------------------------------------------------------------------


def _make_feats_cfg():
    """构造模拟的 feats_cfg，类似 bpc_soft_phase_f / me_soft_phase_f 场景。

    节点图:
      atr_f          (leaf, 非 tick)   → output: atr
      bb_width_f     (leaf, 非 tick)   → output: bb_width_normalized_pct
      cvd_change_f   (leaf, tick dep)  → output: cvd_change_5
      vpin_f         (leaf, tick dep)  → output: vpin
      ofci_pct_f     (leaf, tick dep)  → output: ofci_pct
      soft_phase_f   (依赖 atr_f + cvd_change_f) → has required_columns + column_mappings
    """
    return {
        "atr_f": {
            "compute_func": "compute_atr_from_series",
            "dependencies": [],
            "required_columns": ["high", "low", "close"],
            "output_columns": ["atr"],
        },
        "bb_width_f": {
            "compute_func": "compute_bb_width_from_series",
            "dependencies": [],
            "required_columns": ["close"],
            "output_columns": ["bb_width_normalized_pct"],
        },
        "cvd_change_f": {
            # tick-dependent (函数签名含 ticks_loader_json)
            "compute_func": "compute_cvd_change_from_series",
            "dependencies": [],
            "required_columns": [],
            "output_columns": ["cvd_change_5"],
        },
        "vpin_f": {
            "compute_func": "compute_vpin_from_series",
            "dependencies": [],
            "required_columns": [],
            "output_columns": ["vpin"],
        },
        "ofci_pct_f": {
            # tick-dependent
            "compute_func": "compute_ofci_pct_from_series",
            "dependencies": [],
            "required_columns": [],
            "output_columns": ["ofci_pct"],
        },
        "soft_phase_f": {
            "compute_func": "compute_soft_phase_from_series",
            "dependencies": ["atr_f", "cvd_change_f"],
            "required_columns": ["close", "high", "low", "atr", "volume"],
            "output_columns": [
                "soft_phase_signal",
                "phase_score",
                "energy_index",
            ],
            "column_mappings": {
                "ofci_pct": "ofci_pct",  # 可选参数 (函数默认 None)
                "bb_width_normalized": "bb_width_normalized_pct",
            },
        },
    }


def _make_bars_tf(n=50):
    """构造模拟的 bars_tf DataFrame（已重采样的大时间框架 bars）。"""
    idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    rng = np.random.default_rng(42)
    base = 50000 + rng.normal(0, 100, n).cumsum()
    return pd.DataFrame(
        {
            "open": base - 10,
            "high": base + 20,
            "low": base - 20,
            "close": base,
            "volume": rng.uniform(100, 1000, n),
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# Helper: 复制 IFC 内的 _has_tick_dependency 逻辑
# ---------------------------------------------------------------------------


def _has_tick_dependency(node_name, feats_cfg, tick_dependent_nodes, visited=None):
    if visited is None:
        visited = set()
    if node_name in visited:
        return False
    visited.add(node_name)
    if node_name in tick_dependent_nodes:
        return True
    info = feats_cfg.get(node_name)
    if isinstance(info, dict):
        for dep in info.get("dependencies") or []:
            if _has_tick_dependency(dep, feats_cfg, tick_dependent_nodes, visited):
                return True
    return False


# ============================================================================
# Tests
# ============================================================================


class TestSecondPassEligibility:
    """second pass 准入检查: 只看 required_columns, 不合并 column_mappings."""

    def test_node_enters_second_pass_when_required_columns_available(self):
        """
        soft_phase_f 的 required_columns 全部在 bar_cols_updated 中,
        即使 column_mappings 引用的 ofci_pct 缺失, 也应进入 second pass.
        """
        feats_cfg = _make_feats_cfg()
        tick_dependent_nodes = {"cvd_change_f", "vpin_f", "ofci_pct_f"}

        # 模拟 first pass 后的列 (atr 由 3a-fix 加入, ofci_pct 未计算)
        bar_cols_updated = {
            "open",
            "high",
            "low",
            "close",
            "volume",
            "atr",  # atr_f first pass 产出
            "bb_width_normalized_pct",  # bb_width_f first pass 产出
            "cvd_change_5",  # step 2 OF 计算
            "vpin",  # step 2 OF 计算
            # ofci_pct 不在 ← 这是关键: column_mappings 引用它但它缺失
        }

        skipped = ["soft_phase_f", "cvd_change_f", "vpin_f", "ofci_pct_f"]
        second_pass = []
        for n in skipped:
            if n in tick_dependent_nodes:
                continue
            info = feats_cfg.get(n)
            if not isinstance(info, dict):
                continue
            # ✅ 新逻辑: 只检查 required_columns
            req_cols = set(info.get("required_columns") or [])
            if req_cols.issubset(bar_cols_updated):
                second_pass.append(n)

        assert "soft_phase_f" in second_pass, (
            "soft_phase_f 应进入 second pass: required_columns 全满足, "
            "column_mappings 缺失不应阻止"
        )

    def test_node_blocked_when_required_columns_missing(self):
        """required_columns 缺失时, 节点不应进入 second pass."""
        feats_cfg = _make_feats_cfg()
        tick_dependent_nodes = {"cvd_change_f", "vpin_f", "ofci_pct_f"}

        # atr 未计算 → required_columns 不满足
        bar_cols_updated = {
            "open",
            "high",
            "low",
            "close",
            "volume",
            # 没有 atr
        }

        skipped = ["soft_phase_f"]
        second_pass = []
        for n in skipped:
            if n in tick_dependent_nodes:
                continue
            info = feats_cfg.get(n)
            if not isinstance(info, dict):
                continue
            req_cols = set(info.get("required_columns") or [])
            if req_cols.issubset(bar_cols_updated):
                second_pass.append(n)

        assert "soft_phase_f" not in second_pass, (
            "required_columns 包含 atr 但 atr 不在 bar_cols_updated, "
            "soft_phase_f 不应进入 second pass"
        )

    def test_old_logic_would_block_node(self):
        """验证旧逻辑 (合并 column_mappings + required_columns) 会错误阻止节点。"""
        feats_cfg = _make_feats_cfg()
        info = feats_cfg["soft_phase_f"]

        bar_cols_updated = {
            "open",
            "high",
            "low",
            "close",
            "volume",
            "atr",
            "bb_width_normalized_pct",
            "cvd_change_5",
            "vpin",
        }

        # 旧逻辑: 合并 column_mappings + required_columns
        col_mappings = info.get("column_mappings") or {}
        mapped_cols = set()
        for v in col_mappings.values():
            if isinstance(v, str):
                mapped_cols.add(v)
            elif isinstance(v, list):
                mapped_cols.update(v)
        req_cols = set(info.get("required_columns") or [])
        all_inputs = mapped_cols | req_cols

        # ofci_pct 在 all_inputs 中但不在 bar_cols_updated
        assert "ofci_pct" in all_inputs
        assert "ofci_pct" not in bar_cols_updated
        assert not all_inputs.issubset(
            bar_cols_updated
        ), "旧逻辑: all_inputs 包含 ofci_pct, 不是 bar_cols_updated 子集 → 会阻止节点"

        # 新逻辑: 只看 required_columns
        assert req_cols.issubset(
            bar_cols_updated
        ), "新逻辑: required_columns 是 bar_cols_updated 子集 → 允许节点进入"


class TestColumnMappingsFilter:
    """column_mappings 过滤: 只保留 bars_tf 中实际存在的列."""

    def test_filter_removes_missing_columns(self):
        """缺失的 ofci_pct 应从 column_mappings 中移除."""
        raw_mappings = {
            "ofci_pct": "ofci_pct",
            "bb_width_normalized": "bb_width_normalized_pct",
        }
        bar_cols_updated = {
            "open",
            "high",
            "low",
            "close",
            "volume",
            "atr",
            "bb_width_normalized_pct",
            # ofci_pct 缺失
        }

        avail_mappings = {}
        for param, src in raw_mappings.items():
            if isinstance(src, str) and src in bar_cols_updated:
                avail_mappings[param] = src
            elif isinstance(src, list) and all(s in bar_cols_updated for s in src):
                avail_mappings[param] = src

        assert "bb_width_normalized" in avail_mappings
        assert (
            "ofci_pct" not in avail_mappings
        ), "ofci_pct 不在 bar_cols_updated, 应从 avail_mappings 中移除"

    def test_filter_keeps_all_when_present(self):
        """所有映射列都存在时, 全部保留."""
        raw_mappings = {
            "ofci_pct": "ofci_pct",
            "bb_width_normalized": "bb_width_normalized_pct",
        }
        bar_cols_updated = {
            "open",
            "high",
            "low",
            "close",
            "volume",
            "atr",
            "bb_width_normalized_pct",
            "ofci_pct",
        }

        avail_mappings = {}
        for param, src in raw_mappings.items():
            if isinstance(src, str) and src in bar_cols_updated:
                avail_mappings[param] = src
            elif isinstance(src, list) and all(s in bar_cols_updated for s in src):
                avail_mappings[param] = src

        assert avail_mappings == raw_mappings

    def test_filter_handles_list_mappings(self):
        """list 映射只有全部列存在才保留."""
        raw_mappings = {
            "multi_col": ["col_a", "col_b"],
        }

        # 只有 col_a 存在
        bar_cols_updated = {"col_a"}
        avail_mappings = {}
        for param, src in raw_mappings.items():
            if isinstance(src, str) and src in bar_cols_updated:
                avail_mappings[param] = src
            elif isinstance(src, list) and all(s in bar_cols_updated for s in src):
                avail_mappings[param] = src

        assert "multi_col" not in avail_mappings

        # 两个列都存在
        bar_cols_updated = {"col_a", "col_b"}
        avail_mappings = {}
        for param, src in raw_mappings.items():
            if isinstance(src, str) and src in bar_cols_updated:
                avail_mappings[param] = src
            elif isinstance(src, list) and all(s in bar_cols_updated for s in src):
                avail_mappings[param] = src

        assert avail_mappings == raw_mappings

    def test_filtered_info_passed_to_build_call_args(self):
        """过滤后的 info_filtered 传给 _build_call_args 不会因缺失列报 KeyError."""
        from src.features.loader.feature_computer import _build_call_args

        bars_tf = _make_bars_tf()
        # 添加 atr 列
        bars_tf["atr"] = 100.0
        # 没有 ofci_pct 列

        info_filtered = {
            "compute_func": "compute_soft_phase_from_series",
            "required_columns": ["close", "high", "low", "atr", "volume"],
            "column_mappings": {
                # ofci_pct 已被过滤掉, 只保留 bars_tf 中存在的
                # (这里模拟: 没有任何映射能匹配)
            },
            "compute_params": {},
        }

        # 不应抛 KeyError
        call_args, call_kwargs = _build_call_args(
            info_filtered, bars_tf, "soft_phase_f"
        )
        # call_args 应包含 df（因为 required_columns 中的列存在于 df 中）
        assert call_args is not None

    def test_unfiltered_info_raises_key_error(self):
        """未过滤的 column_mappings 引用缺失列时 _build_call_args 抛 KeyError."""
        from src.features.loader.feature_computer import _build_call_args

        bars_tf = _make_bars_tf()
        bars_tf["atr"] = 100.0

        info_unfiltered = {
            "compute_func": "compute_soft_phase_from_series",
            "required_columns": ["close", "high", "low", "atr", "volume"],
            "column_mappings": {
                "ofci_pct": "ofci_pct",  # 不在 bars_tf 中
            },
            "compute_params": {},
        }

        with pytest.raises(KeyError, match="ofci_pct"):
            _build_call_args(info_unfiltered, bars_tf, "soft_phase_f")


class TestThreeAFix:
    """3a-fix: skipped 节点的非-tick 依赖加入 first pass filtered list."""

    def test_non_tick_deps_added_to_filtered(self):
        """
        soft_phase_f 被跳过 → 其依赖 atr_f (非 tick) 应加入 filtered,
        cvd_change_f (tick) 不加入.
        """
        feats_cfg = _make_feats_cfg()
        tick_dependent_nodes = {"cvd_change_f", "vpin_f", "ofci_pct_f"}

        live_feature_nodes = ["soft_phase_f", "bb_width_f"]
        bar_cols = {"open", "high", "low", "close", "volume"}

        # Step 3a: 分离 filtered / skipped
        filtered = []
        skipped = []
        for n in live_feature_nodes:
            if _has_tick_dependency(n, feats_cfg, tick_dependent_nodes):
                skipped.append(n)
            else:
                info = feats_cfg.get(n)
                if isinstance(info, dict):
                    req_cols = set(info.get("required_columns") or [])
                    deps = info.get("dependencies") or []
                    if req_cols and not req_cols.issubset(bar_cols) and not deps:
                        skipped.append(n)
                        continue
                filtered.append(n)

        assert "soft_phase_f" in skipped, "soft_phase_f 有 tick 依赖应被跳过"
        assert "bb_width_f" in filtered, "bb_width_f 无 tick 依赖应保留"

        # 3a-fix: 将 skipped 节点的非-tick 依赖加入 filtered
        filtered_set = set(filtered)
        for n in skipped:
            if n in tick_dependent_nodes:
                continue
            info = feats_cfg.get(n)
            if isinstance(info, dict):
                for dep in info.get("dependencies") or []:
                    if dep not in filtered_set and not _has_tick_dependency(
                        dep, feats_cfg, tick_dependent_nodes
                    ):
                        filtered.append(dep)
                        filtered_set.add(dep)

        assert (
            "atr_f" in filtered
        ), "atr_f 是 soft_phase_f 的非 tick 依赖, 应被 3a-fix 加入 filtered"
        assert (
            "cvd_change_f" not in filtered
        ), "cvd_change_f 是 tick 依赖, 不应加入 filtered"

    def test_leaf_tick_nodes_skipped_in_3a_fix(self):
        """tick 叶节点 (如 cvd_change_f) 不参与 3a-fix 依赖展开."""
        feats_cfg = _make_feats_cfg()
        tick_dependent_nodes = {"cvd_change_f", "vpin_f", "ofci_pct_f"}

        skipped = ["cvd_change_f", "soft_phase_f"]
        filtered = ["bb_width_f"]
        filtered_set = set(filtered)

        for n in skipped:
            if n in tick_dependent_nodes:
                continue  # 叶节点跳过
            info = feats_cfg.get(n)
            if isinstance(info, dict):
                for dep in info.get("dependencies") or []:
                    if dep not in filtered_set and not _has_tick_dependency(
                        dep, feats_cfg, tick_dependent_nodes
                    ):
                        filtered.append(dep)
                        filtered_set.add(dep)

        # cvd_change_f 是叶 tick 节点, 被 continue 跳过
        # 只有 soft_phase_f 的依赖被展开
        assert "atr_f" in filtered


class TestMESoftPhaseNeedsAtr:
    """验证 ME 策略的 me_soft_phase_f 需要 atr 作为必须参数."""

    def test_me_soft_phase_function_requires_atr(self):
        """compute_momentum_expansion_soft_phase_from_series 的 atr 是必须参数."""
        from src.features.time_series.momentum_expansion_features import (
            compute_momentum_expansion_soft_phase_from_series,
        )

        sig = inspect.signature(compute_momentum_expansion_soft_phase_from_series)
        atr_param = sig.parameters.get("atr")
        assert atr_param is not None, "函数应有 atr 参数"
        assert (
            atr_param.default is inspect.Parameter.empty
        ), "atr 应是必须参数 (无默认值), 不是 Optional"

    def test_me_soft_phase_config_has_atr_dependency(self):
        """me_soft_phase_f 在 feature_dependencies.yaml 中应依赖 atr_f."""
        import yaml
        from pathlib import Path

        deps_path = Path("config/feature_dependencies.yaml")
        if not deps_path.exists():
            pytest.skip("feature_dependencies.yaml not found")

        with open(deps_path) as f:
            deps = yaml.safe_load(f)

        feats = deps.get("features") or {}
        me_sp = feats.get("me_soft_phase_f")
        assert me_sp is not None, "me_soft_phase_f 应在 feature_dependencies.yaml 中"
        assert "atr_f" in (
            me_sp.get("dependencies") or []
        ), "me_soft_phase_f 应有 atr_f 依赖"
        assert "atr" in (
            me_sp.get("required_columns") or []
        ), "me_soft_phase_f 的 required_columns 应包含 atr"
        assert "bb_width_normalized_pct_f" in (
            me_sp.get("dependencies") or []
        ), "me_soft_phase_f 应依赖 bb_width_normalized_pct_f 以计算 me_semantic_chop"


class TestBpcSoftPhaseSecondPassIntegration:
    """集成测试: bpc_soft_phase_f 和 me_soft_phase_f 应能通过 second pass 计算."""

    def test_bpc_soft_phase_has_optional_ofci_pct(self):
        """bpc_soft_phase_f 的 ofci_pct 是可选参数 (默认 None)."""
        from src.features.time_series.bpc_features import (
            compute_bpc_soft_phase_from_series,
        )

        sig = inspect.signature(compute_bpc_soft_phase_from_series)
        ofci_param = sig.parameters.get("ofci_pct")
        assert ofci_param is not None, "函数应有 ofci_pct 参数"
        assert (
            ofci_param.default is None
        ), "ofci_pct 应有默认值 None (可选), 才能在实盘缺失 tick 数据时正常运行"

    def test_full_second_pass_scenario(self):
        """
        端到端模拟:
        1. live_feature_nodes 包含 soft_phase_f (有 tick 依赖)
        2. First pass 跳过 soft_phase_f, 但 3a-fix 加入 atr_f
        3. Second pass: required_columns 满足 → 进入
        4. column_mappings 过滤 → ofci_pct 移除
        5. 成功调用 compute_func
        """
        feats_cfg = _make_feats_cfg()
        tick_dependent_nodes = {"cvd_change_f", "vpin_f", "ofci_pct_f"}

        live_feature_nodes = ["bb_width_f", "soft_phase_f"]

        # Step 3a: 分离
        filtered, skipped = [], []
        for n in live_feature_nodes:
            if _has_tick_dependency(n, feats_cfg, tick_dependent_nodes):
                skipped.append(n)
            else:
                filtered.append(n)

        # 3a-fix
        filtered_set = set(filtered)
        for n in skipped:
            if n in tick_dependent_nodes:
                continue
            info = feats_cfg.get(n)
            if isinstance(info, dict):
                for dep in info.get("dependencies") or []:
                    if dep not in filtered_set and not _has_tick_dependency(
                        dep, feats_cfg, tick_dependent_nodes
                    ):
                        filtered.append(dep)
                        filtered_set.add(dep)

        assert set(filtered) == {
            "bb_width_f",
            "atr_f",
        }, f"filtered 应包含 bb_width_f + atr_f (3a-fix), got {filtered}"

        # 模拟 first pass 后可用列
        bar_cols_updated = {
            "open",
            "high",
            "low",
            "close",
            "volume",
            "atr",
            "bb_width_normalized_pct",
            "cvd_change_5",
            "vpin",  # step 2 OF
        }

        # Second pass 准入
        second_pass = []
        for n in skipped:
            if n in tick_dependent_nodes:
                continue
            info = feats_cfg.get(n)
            if not isinstance(info, dict):
                continue
            req_cols = set(info.get("required_columns") or [])
            if req_cols.issubset(bar_cols_updated):
                second_pass.append(n)

        assert second_pass == ["soft_phase_f"]

        # column_mappings 过滤
        info = feats_cfg["soft_phase_f"]
        raw_mappings = info.get("column_mappings") or {}
        avail_mappings = {}
        for param, src in raw_mappings.items():
            if isinstance(src, str) and src in bar_cols_updated:
                avail_mappings[param] = src
            elif isinstance(src, list) and all(s in bar_cols_updated for s in src):
                avail_mappings[param] = src

        assert "ofci_pct" not in avail_mappings, "ofci_pct 不在 bar_cols_updated"
        assert avail_mappings == {"bb_width_normalized": "bb_width_normalized_pct"}


class TestWhenClauseReservedKeys:
    """验证 _extract_features_from_when 正确跳过 all_of/any_of 等逻辑运算符。"""

    def test_all_of_not_treated_as_feature(self):
        """FER gate.yaml 使用 all_of 包裹条件，不应把 all_of 当作特征名。"""
        from src.time_series_model.live.live_feature_plan import (
            _extract_features_from_when,
        )

        when = {
            "all_of": [
                {"fer_trapped_longs_score": {"value_lt": 3.4965}},
                {"fer_trapped_shorts_score": {"value_lt": 3.7941}},
            ]
        }
        features = _extract_features_from_when(when)
        assert "all_of" not in features, "all_of 是逻辑运算符，不是特征名"
        assert "fer_trapped_longs_score" in features
        assert "fer_trapped_shorts_score" in features

    def test_any_of_not_treated_as_feature(self):
        """any_of 也不应被当作特征名。"""
        from src.time_series_model.live.live_feature_plan import (
            _extract_features_from_when,
        )

        when = {
            "any_of": [
                {"roc_20": {"value_gt": 1.0}},
                {"volatility_pct": {"value_lt": 0.5}},
            ]
        }
        features = _extract_features_from_when(when)
        assert "any_of" not in features
        assert features == {"roc_20", "volatility_pct"}

    def test_nested_all_of_and_simple_keys(self):
        """混合使用 all_of 和普通特征键。"""
        from src.time_series_model.live.live_feature_plan import (
            _extract_features_from_when,
        )

        when = {
            "and": [
                {
                    "all_of": [
                        {"feat_a": {"value_gt": 1}},
                        {"feat_b": {"value_lt": 2}},
                    ]
                },
                {"feat_c": {"value_gt": 0.5}},
            ]
        }
        features = _extract_features_from_when(when)
        assert features == {"feat_a", "feat_b", "feat_c"}
        assert "all_of" not in features
        assert "and" not in features

    def test_fer_archetypes_no_all_of_in_features(self):
        """FER archetypes 提取结果不应包含 all_of。"""
        from src.time_series_model.live.live_feature_plan import (
            extract_features_from_archetypes,
        )
        from pathlib import Path

        fer_dir = Path("config/strategies/fer/archetypes")
        if not fer_dir.exists():
            pytest.skip("FER archetypes not found")

        feat_set, nodes = extract_features_from_archetypes(fer_dir)
        assert (
            "all_of" not in feat_set
        ), "all_of 是 gate.yaml 的逻辑运算符，不应出现在 live_feature_set 中"

    def test_live_fer_archetypes_no_all_of_in_features(self):
        """live/highcap FER archetypes 同样不应包含 all_of。"""
        from src.time_series_model.live.live_feature_plan import (
            extract_features_from_archetypes,
        )
        from pathlib import Path

        fer_dir = Path("live/highcap/config/strategies/fer/archetypes")
        if not fer_dir.exists():
            pytest.skip("live FER archetypes not found")

        feat_set, nodes = extract_features_from_archetypes(fer_dir)
        assert "all_of" not in feat_set

    def test_pred_is_not_treated_as_live_computed_feature(self, tmp_path):
        """pred 是离线预测列，不应进入 Feature Bus 的 live_feature_set。"""
        from src.time_series_model.live.live_feature_plan import (
            extract_features_from_archetypes,
        )

        archetypes_dir = tmp_path / "archetypes"
        archetypes_dir.mkdir()
        (archetypes_dir / "evidence.yaml").write_text(
            """
evidence:
  - id: evidence_pred
    feature: pred
  - id: evidence_close
    feature: close
""",
            encoding="utf-8",
        )
        deps_path = tmp_path / "feature_dependencies.yaml"
        deps_path.write_text("features: {}\n", encoding="utf-8")

        feat_set, _nodes = extract_features_from_archetypes(
            archetypes_dir,
            feature_deps_path=deps_path,
        )
        assert "pred" not in feat_set
        assert "close" in feat_set

    def test_categorical_outputs_are_not_live_health_expected(self, tmp_path):
        """分类输出列不应进入数值 live_feature_set 的健康检查口径。"""
        from src.time_series_model.live.live_feature_plan import (
            extract_features_from_archetypes,
        )

        archetypes_dir = tmp_path / "archetypes"
        archetypes_dir.mkdir()
        (archetypes_dir / "gate.yaml").write_text(
            """
hard_gates:
  - id: gate_box_score
    enabled: true
    when:
      box_score:
        value_gt: 0.5
""",
            encoding="utf-8",
        )
        deps_path = tmp_path / "feature_dependencies.yaml"
        deps_path.write_text(
            """
features:
  box_structure_f:
    output_columns:
      - box_score
      - box_regime_label
""",
            encoding="utf-8",
        )

        feat_set, _nodes = extract_features_from_archetypes(
            archetypes_dir,
            feature_deps_path=deps_path,
        )
        assert "box_score" in feat_set
        assert "box_regime_label" not in feat_set


class TestSpotAccumStructuralExitFeatures:
    def test_spot_accum_includes_weekly_macro_exit_node(self):
        from src.time_series_model.live.live_feature_plan import (
            extract_features_from_archetypes,
        )
        from pathlib import Path

        arch = Path("config/strategies/bad-candidates/spot_accum/archetypes")
        if not arch.exists():
            pytest.skip("spot_accum archetypes not found")

        feat_set, nodes = extract_features_from_archetypes(arch)
        assert "weekly_macro_cycle_exit_f" in nodes
        assert "weekly_macro_cycle_exit_signal" in feat_set
        assert "rsi_f" in nodes
        assert "rsi" in feat_set


class TestAtrNodeAlwaysIncluded:
    """验证 extract_features_from_archetypes 始终包含 atr_f 节点。

    根因: load_features_from_requested 只返回 requested features 的输出列，
    atr_f 作为依赖被计算但输出被丢弃。必须显式请求。
    """

    def test_me_archetypes_include_atr_f(self):
        """ME 60T FC 的 live_feature_nodes 应包含 atr_f。"""
        from src.time_series_model.live.live_feature_plan import (
            extract_features_from_archetypes,
        )
        from pathlib import Path

        me_dir = Path("config/strategies/bad-candidates/me/archetypes")
        if not me_dir.exists():
            pytest.skip("ME archetypes not found")

        feat_set, nodes = extract_features_from_archetypes(me_dir)
        assert "atr" in feat_set, "atr 应在 live_feature_set 中"
        assert "atr_f" in nodes, (
            "atr_f 应在 live_feature_nodes 中，否则 "
            "load_features_from_requested 会丢弃 atr 列"
        )

    def test_bpc_archetypes_include_atr_f(self):
        """BPC 240T FC 的 live_feature_nodes 应包含 atr_f。"""
        from src.time_series_model.live.live_feature_plan import (
            extract_features_from_archetypes,
        )
        from pathlib import Path

        bpc_dir = Path("config/strategies/bad-candidates/bpc/archetypes")
        if not bpc_dir.exists():
            pytest.skip("BPC archetypes not found")

        feat_set, nodes = extract_features_from_archetypes(bpc_dir)
        assert "atr" in feat_set
        assert "atr_f" in nodes


# ---------------------------------------------------------------------------
# 6. 元数据驱动 warmup 校验 + 代码路径统一
# ---------------------------------------------------------------------------


class TestMetadataWarmupCheck:
    """_get_warmup_check_features 基于 feature_dependencies 元数据，
    而非后缀猜测。"""

    def test_pct_suffix_not_in_warmup_features(self):
        """_pct 后缀的特征 (fer_signed_efficiency_pct) 不应被纳入 warmup 检查。

        它们用 .rank(pct=True) 小窗口，不是大窗口 rolling percentile。
        """
        from src.time_series_model.live.incremental_feature_computer import (
            IncrementalFeatureComputer,
        )

        fc = IncrementalFeatureComputer.__new__(IncrementalFeatureComputer)
        fc.live_feature_set = {
            "fer_signed_efficiency_pct",
            "shd_pct",
            "bpc_volume_compression_pct",
            "atr_percentile",
        }
        fc._feature_deps = {
            "features": {
                # _pct 特征：无 percentile_window，compute_func 不含 percentile
                "fer_signals_f": {
                    "compute_func": "compute_fer_failure_signals_from_series",
                    "compute_params": {"efficiency_window": 20},
                    "output_columns": ["fer_signed_efficiency_pct"],
                },
                "shd_f": {
                    "compute_func": "compute_shd_from_series",
                    "compute_params": {},
                    "output_columns": ["shd_pct"],
                },
                # 真正的 percentile 特征：有 percentile_window 或 func 名含 percentile
                "atr_percentile_f": {
                    "compute_func": "compute_atr_percentile_from_series",
                    "compute_params": {"window": 540},
                    "output_columns": ["atr_percentile"],
                },
            }
        }
        result = fc._get_warmup_check_features()
        # atr_percentile 应在 (compute_func 含 percentile)
        assert "atr_percentile" in result
        # _pct 特征不应在
        assert "fer_signed_efficiency_pct" not in result
        assert "shd_pct" not in result
        assert "bpc_volume_compression_pct" not in result

    def test_percentile_window_feature_in_warmup_check(self):
        """compute_params 包含 percentile_window >= 100 的节点应被纳入。"""
        from src.time_series_model.live.incremental_feature_computer import (
            IncrementalFeatureComputer,
        )

        fc = IncrementalFeatureComputer.__new__(IncrementalFeatureComputer)
        fc.live_feature_set = {"jump_risk_pct", "path_length_pct"}
        fc._feature_deps = {
            "features": {
                "jump_risk_pct_f": {
                    "compute_func": "compute_jump_risk_pct_from_series",
                    "compute_params": {"percentile_window": 540},
                    "output_columns": ["jump_risk_pct"],
                },
                "path_length_pct_f": {
                    "compute_func": "compute_path_length_pct_from_series",
                    "compute_params": {"percentile_window": 540},
                    "output_columns": ["path_length_pct"],
                },
            }
        }
        result = fc._get_warmup_check_features()
        assert "jump_risk_pct" in result
        assert "path_length_pct" in result

    def test_small_percentile_window_excluded(self):
        """percentile_window < 100 不纳入 warmup 检查。"""
        from src.time_series_model.live.incremental_feature_computer import (
            IncrementalFeatureComputer,
        )

        fc = IncrementalFeatureComputer.__new__(IncrementalFeatureComputer)
        fc.live_feature_set = {"some_pct"}
        fc._feature_deps = {
            "features": {
                "some_f": {
                    "compute_func": "compute_some",
                    "compute_params": {"percentile_window": 20},
                    "output_columns": ["some_pct"],
                },
            }
        }
        result = fc._get_warmup_check_features()
        assert "some_pct" not in result

    def test_validate_warmup_raises_on_missing(self):
        """_validate_warmup 应在大窗口百分位特征缺失时报错。"""
        from src.time_series_model.live.incremental_feature_computer import (
            IncrementalFeatureComputer,
        )

        fc = IncrementalFeatureComputer.__new__(IncrementalFeatureComputer)
        fc.live_feature_set = {"atr_percentile"}
        fc._feature_deps = {
            "features": {
                "atr_percentile_f": {
                    "compute_func": "compute_atr_percentile_from_series",
                    "compute_params": {"window": 540},
                    "output_columns": ["atr_percentile"],
                },
            }
        }
        # 构造一个没有 atr_percentile 列的 DataFrame
        bars_tf = pd.DataFrame(
            {"close": [1.0, 2.0]},
            index=pd.date_range("2024-01-01", periods=2, freq="4h"),
        )
        with pytest.raises(RuntimeError, match="Warmup 不足"):
            fc._validate_warmup(bars_tf, features={"close": 1.0})

    def test_validate_warmup_passes_when_present(self):
        """_validate_warmup 不应在特征存在时报错。"""
        from src.time_series_model.live.incremental_feature_computer import (
            IncrementalFeatureComputer,
        )

        fc = IncrementalFeatureComputer.__new__(IncrementalFeatureComputer)
        fc.live_feature_set = {"atr_percentile"}
        fc._feature_deps = {
            "features": {
                "atr_percentile_f": {
                    "compute_func": "compute_atr_percentile_from_series",
                    "compute_params": {"window": 540},
                    "output_columns": ["atr_percentile"],
                },
            }
        }
        bars_tf = pd.DataFrame(
            {"atr_percentile": [0.5]},
            index=pd.date_range("2024-01-01", periods=1, freq="4h"),
        )
        fc._validate_warmup(bars_tf, features={"atr_percentile": 0.5})  # 不应报错

    def test_validate_warmup_skips_non_numeric_objects_in_last_row(self):
        """最后一行若有分类列 (如 box_regime_label=str)，不得以 float(str) 抛错。"""
        from src.time_series_model.live.incremental_feature_computer import (
            IncrementalFeatureComputer,
        )

        fc = IncrementalFeatureComputer.__new__(IncrementalFeatureComputer)
        fc.live_feature_set = {"atr_percentile"}
        fc._feature_deps = {
            "features": {
                "atr_percentile_f": {
                    "compute_func": "compute_atr_percentile_from_series",
                    "compute_params": {"window": 540},
                    "output_columns": ["atr_percentile"],
                },
            }
        }
        bars_tf = pd.DataFrame(
            {
                "atr_percentile": [0.55],
                "box_regime_label": ["mid"],
            },
            index=pd.date_range("2024-01-01", periods=1, freq="4h"),
        )
        fc._validate_warmup(bars_tf, features=None)


class TestCodePathUnification:
    """compute_features_batch 和 compute_features_dataframe 共享 _compute_features_core。"""

    def test_batch_calls_core(self):
        """compute_features_batch 内部应调用 _compute_features_core。"""
        from src.time_series_model.live.incremental_feature_computer import (
            IncrementalFeatureComputer,
        )
        import inspect

        src = inspect.getsource(IncrementalFeatureComputer.compute_features_batch)
        assert (
            "_compute_features_core" in src
        ), "compute_features_batch 应调用 _compute_features_core 而非重复实现"

    def test_dataframe_calls_core(self):
        """compute_features_dataframe 内部应调用 _compute_features_core。"""
        from src.time_series_model.live.incremental_feature_computer import (
            IncrementalFeatureComputer,
        )
        import inspect

        src = inspect.getsource(IncrementalFeatureComputer.compute_features_dataframe)
        assert (
            "_compute_features_core" in src
        ), "compute_features_dataframe 应调用 _compute_features_core 而非重复实现"

    def test_dataframe_also_validates_warmup(self):
        """compute_features_dataframe 应包含 _validate_warmup 调用。"""
        from src.time_series_model.live.incremental_feature_computer import (
            IncrementalFeatureComputer,
        )
        import inspect

        src = inspect.getsource(IncrementalFeatureComputer.compute_features_dataframe)
        assert (
            "_validate_warmup" in src
        ), "compute_features_dataframe 应包含 warmup 校验，与实盘路径一致"

    def test_no_duplicated_resample_logic(self):
        """compute_features_batch / compute_features_dataframe 不应包含 resample 逻辑。"""
        from src.time_series_model.live.incremental_feature_computer import (
            IncrementalFeatureComputer,
        )
        import inspect

        for method_name in ["compute_features_batch", "compute_features_dataframe"]:
            src = inspect.getsource(getattr(IncrementalFeatureComputer, method_name))
            assert (
                "resample" not in src
            ), f"{method_name} 不应包含 resample 逻辑，应委托给 _compute_features_core"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
