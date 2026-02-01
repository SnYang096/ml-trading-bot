"""
Failure-first Binary Label 模块

🟥 Failure-first 的核心思想：
- 不是问"能不能赚钱"，而是问"在哪些结构条件下会系统性失败"
- 树学习的目标是找到 failure probability 显著升高的区域
- 产出是"禁区地图"，不是交易信号

⚠️ 关键设计：
- failure 是 boolean，不是连续值
- failure 定义是"不可接受的失败"，不是简单的"亏钱"
- 包含 MAE（最大不利偏移）、recovery_time 等结构性判断

参考文档：docs/architecture/OUTCOME_BASED_TREE_LABELING.md
"""

from __future__ import annotations

from typing import Literal, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd


EPS = 1e-8


@dataclass
class FailureDefinition:
    """
    失败定义配置。

    ⚠️ 这是整个 Failure-first 系统的"分水岭"。
    失败不是"亏钱"，而是"不可接受的失败"。

    Attributes:
        rr_threshold: forward_rr 阈值（低于此值视为失败）
        mae_mult: MAE 乘数（相对于 expected_stop）
        mfe_mult: MFE 乘数（相对于 expected_target）
        recovery_time_mult: 恢复时间乘数（相对于 expected_holding）
        require_all: 是否要求所有条件同时满足（False = 任一满足即为失败）
    """

    rr_threshold: float = -0.8  # forward_rr < -0.8R 视为失败
    mae_mult: float = 1.2  # MAE > 1.2 * expected_stop
    mfe_mult: float = 0.3  # MFE < 0.3 * expected_target
    recovery_time_mult: float = 2.0  # recovery_time > 2 * expected
    require_all: bool = False  # 任一条件满足即为失败


def compute_failure_first_label(
    df: pd.DataFrame,
    direction: Literal["long", "short"] = "long",
    horizon: int = 50,
    price_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    atr_col: str = "atr",
    failure_def: Optional[FailureDefinition] = None,
    expected_stop_atr: float = 1.0,  # 预期止损 = 1 ATR
    expected_target_atr: float = 2.0,  # 预期目标 = 2 ATR
    **kwargs,
) -> pd.Series:
    """
    计算 Failure-first 二值标签。

    🟥 核心语义：
    - 1 = 不可接受的失败（unacceptable failure）
    - 0 = 可接受或正常

    ⚠️ 失败定义（默认，可通过 failure_def 自定义）：
    failure = (
        forward_rr < -0.8R
        OR (
            mae > 1.2 * expected_stop
            AND mfe < 0.3 * expected_target
        )
    )

    语义解释：
    - forward_rr < -0.8R: 路径极端不利，亏损严重
    - mae > 1.2 * expected_stop: 被打穿止损 1.2 倍
    - mfe < 0.3 * expected_target: 几乎没有达到目标（没给你赚钱的机会）

    Args:
        df: 价格数据，必须包含 OHLC 和 ATR
        direction: 交易方向，"long" 或 "short"
        horizon: 持仓窗口（bars）
        failure_def: 自定义失败定义
        expected_stop_atr: 预期止损（ATR 倍数）
        expected_target_atr: 预期目标（ATR 倍数）

    Returns:
        pd.Series: 二值失败标签（1=失败，0=正常）
    """
    if failure_def is None:
        failure_def = FailureDefinition()

    required_cols = [price_col, high_col, low_col, atr_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"缺少必需列: {missing}")

    close = df[price_col].values
    high = df[high_col].values
    low = df[low_col].values
    atr = df[atr_col].values
    n = len(df)

    # 输出数组
    failure_label = np.full(n, np.nan)

    # 用于诊断的中间变量
    forward_rr = np.full(n, np.nan)
    mae_atr = np.full(n, np.nan)
    mfe_atr = np.full(n, np.nan)

    for i in range(n - horizon):
        entry_price = close[i]
        current_atr = atr[i]

        if np.isnan(current_atr) or current_atr <= EPS:
            continue

        # 计算持仓窗口内的 MFE 和 MAE
        future_high = np.nanmax(high[i + 1 : i + horizon + 1])
        future_low = np.nanmin(low[i + 1 : i + horizon + 1])

        if direction == "long":
            mfe = future_high - entry_price  # 最大有利偏移
            mae = entry_price - future_low  # 最大不利偏移
        else:
            mfe = entry_price - future_low
            mae = future_high - entry_price

        # 归一化为 ATR 倍数
        mfe_norm = mfe / max(current_atr, EPS)
        mae_norm = mae / max(current_atr, EPS)
        rr = (mfe - mae) / max(current_atr, EPS)

        forward_rr[i] = rr
        mfe_atr[i] = mfe_norm
        mae_atr[i] = mae_norm

        # ========== 失败判断 ==========
        # 条件 1: forward_rr 极端负
        cond_rr_bad = rr < failure_def.rr_threshold

        # 条件 2: MAE 过大 且 MFE 过小
        cond_mae_bad = mae_norm > failure_def.mae_mult * expected_stop_atr
        cond_mfe_bad = mfe_norm < failure_def.mfe_mult * expected_target_atr
        cond_structural_bad = cond_mae_bad and cond_mfe_bad

        # 综合判断
        if failure_def.require_all:
            # 严格模式：所有条件都满足才算失败
            is_failure = cond_rr_bad and cond_structural_bad
        else:
            # 宽松模式：任一条件满足即为失败（推荐）
            is_failure = cond_rr_bad or cond_structural_bad

        failure_label[i] = 1.0 if is_failure else 0.0

    return pd.Series(failure_label, index=df.index, name="failure_label")


def compute_failure_components(
    df: pd.DataFrame,
    direction: Literal["long", "short"] = "long",
    horizon: int = 50,
    price_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    atr_col: str = "atr",
    expected_stop_atr: float = 1.0,
    expected_target_atr: float = 2.0,
    **kwargs,
) -> pd.DataFrame:
    """
    计算失败标签的各个组成部分（用于诊断和理解）。

    返回的 DataFrame 包含：
    - forward_rr: 路径极端收益风险比
    - mfe_atr: 最大有利偏移（ATR 倍数）
    - mae_atr: 最大不利偏移（ATR 倍数）
    - is_rr_failure: forward_rr < threshold
    - is_structural_failure: MAE 大 且 MFE 小
    - failure_label: 最终失败标签

    用于：
    - 理解失败的具体原因
    - 调整失败定义参数
    - 诊断 failure_rate 分布
    """
    required_cols = [price_col, high_col, low_col, atr_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"缺少必需列: {missing}")

    close = df[price_col].values
    high = df[high_col].values
    low = df[low_col].values
    atr = df[atr_col].values
    n = len(df)

    results = {
        "forward_rr": np.full(n, np.nan),
        "mfe_atr": np.full(n, np.nan),
        "mae_atr": np.full(n, np.nan),
        "is_rr_failure": np.full(n, np.nan),
        "is_structural_failure": np.full(n, np.nan),
        "failure_label": np.full(n, np.nan),
    }

    failure_def = FailureDefinition()

    for i in range(n - horizon):
        entry_price = close[i]
        current_atr = atr[i]

        if np.isnan(current_atr) or current_atr <= EPS:
            continue

        future_high = np.nanmax(high[i + 1 : i + horizon + 1])
        future_low = np.nanmin(low[i + 1 : i + horizon + 1])

        if direction == "long":
            mfe = future_high - entry_price
            mae = entry_price - future_low
        else:
            mfe = entry_price - future_low
            mae = future_high - entry_price

        mfe_norm = mfe / max(current_atr, EPS)
        mae_norm = mae / max(current_atr, EPS)
        rr = (mfe - mae) / max(current_atr, EPS)

        results["forward_rr"][i] = rr
        results["mfe_atr"][i] = mfe_norm
        results["mae_atr"][i] = mae_norm

        # 失败条件
        cond_rr_bad = rr < failure_def.rr_threshold
        cond_mae_bad = mae_norm > failure_def.mae_mult * expected_stop_atr
        cond_mfe_bad = mfe_norm < failure_def.mfe_mult * expected_target_atr
        cond_structural_bad = cond_mae_bad and cond_mfe_bad

        results["is_rr_failure"][i] = 1.0 if cond_rr_bad else 0.0
        results["is_structural_failure"][i] = 1.0 if cond_structural_bad else 0.0
        results["failure_label"][i] = (
            1.0 if (cond_rr_bad or cond_structural_bad) else 0.0
        )

    return pd.DataFrame(results, index=df.index)


def get_failure_stats(
    df: pd.DataFrame,
    failure_col: str = "failure_label",
    feature_cols: Optional[list] = None,
    n_quantiles: int = 5,
) -> dict:
    """
    计算失败统计信息。

    用于理解：
    - 全局 failure_rate（baseline）
    - 各特征分位数的 failure_rate
    - 哪些特征区域 failure_rate 显著升高

    Returns:
        dict: {
            "baseline_failure_rate": float,
            "total_samples": int,
            "failure_samples": int,
            "feature_failure_rates": {feature: {quantile: failure_rate, lift}}
        }
    """
    valid_mask = df[failure_col].notna()
    df_valid = df[valid_mask]

    total = len(df_valid)
    failures = (df_valid[failure_col] == 1).sum()
    baseline = failures / total if total > 0 else 0

    stats = {
        "baseline_failure_rate": float(baseline),
        "total_samples": int(total),
        "failure_samples": int(failures),
        "feature_failure_rates": {},
    }

    if feature_cols is None:
        return stats

    for col in feature_cols:
        if col not in df_valid.columns:
            continue

        try:
            quantiles = pd.qcut(
                df_valid[col], n_quantiles, labels=False, duplicates="drop"
            )
            col_stats = {}

            for q in range(quantiles.nunique()):
                q_mask = quantiles == q
                q_total = q_mask.sum()
                q_failures = (df_valid.loc[q_mask, failure_col] == 1).sum()
                q_rate = q_failures / q_total if q_total > 0 else 0
                lift = q_rate / baseline if baseline > 0 else 0

                col_stats[f"q{q}"] = {
                    "failure_rate": float(q_rate),
                    "lift": float(lift),
                    "samples": int(q_total),
                }

            stats["feature_failure_rates"][col] = col_stats
        except Exception:
            continue

    return stats


# ============================================================
# Failure 子标签（用于分析，不进模型）
# ============================================================


def compute_failure_subtypes(
    df: pd.DataFrame,
    direction: Literal["long", "short"] = "long",
    horizon: int = 50,
    price_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    atr_col: str = "atr",
    expected_stop_atr: float = 1.0,
    expected_target_atr: float = 2.0,
    **kwargs,
) -> pd.DataFrame:
    """
    计算 Failure 子标签（只用于分析，不进模型）。

    🟥 两种 failure 子类型：

    1. failure_rr_extreme: forward_rr < -0.8R
       - 路径极端不利，亏损严重
       - 通常意味着"踩了大坑"

    2. failure_no_opportunity: MAE > 1.2*stop AND MFE < 0.3*target
       - 被打穿止损，且没给赚钱机会
       - 意味着"入场后立刻反向，没有任何转戴空间"

    Args:
        df: 价格数据，必须包含 OHLC 和 ATR
        direction: 交易方向
        horizon: 持仓窗口（bars）

    Returns:
        pd.DataFrame: 包含以下列
        - failure_rr_extreme: 1=极端RR失败, 0=否
        - failure_no_opportunity: 1=无机会失败, 0=否
        - failure_any: 1=任一失败, 0=否
        - forward_rr: 路径RR
        - mfe_atr: 最大有利偏移
        - mae_atr: 最大不利偏移
    """
    required_cols = [price_col, high_col, low_col, atr_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"缺少必需列: {missing}")

    close = df[price_col].values
    high = df[high_col].values
    low = df[low_col].values
    atr = df[atr_col].values
    n = len(df)

    failure_def = FailureDefinition()

    results = {
        "failure_rr_extreme": np.full(n, np.nan),
        "failure_no_opportunity": np.full(n, np.nan),
        "failure_any": np.full(n, np.nan),
        "forward_rr": np.full(n, np.nan),
        "mfe_atr": np.full(n, np.nan),
        "mae_atr": np.full(n, np.nan),
    }

    for i in range(n - horizon):
        entry_price = close[i]
        current_atr = atr[i]

        if np.isnan(current_atr) or current_atr <= EPS:
            continue

        future_high = np.nanmax(high[i + 1 : i + horizon + 1])
        future_low = np.nanmin(low[i + 1 : i + horizon + 1])

        if direction == "long":
            mfe = future_high - entry_price
            mae = entry_price - future_low
        else:
            mfe = entry_price - future_low
            mae = future_high - entry_price

        mfe_norm = mfe / max(current_atr, EPS)
        mae_norm = mae / max(current_atr, EPS)
        rr = (mfe - mae) / max(current_atr, EPS)

        results["forward_rr"][i] = rr
        results["mfe_atr"][i] = mfe_norm
        results["mae_atr"][i] = mae_norm

        # 失败子类型 1: RR 极端失败
        cond_rr_bad = rr < failure_def.rr_threshold
        results["failure_rr_extreme"][i] = 1.0 if cond_rr_bad else 0.0

        # 失败子类型 2: 无机会失败
        cond_mae_bad = mae_norm > failure_def.mae_mult * expected_stop_atr
        cond_mfe_bad = mfe_norm < failure_def.mfe_mult * expected_target_atr
        cond_no_opp = cond_mae_bad and cond_mfe_bad
        results["failure_no_opportunity"][i] = 1.0 if cond_no_opp else 0.0

        # 任一失败
        results["failure_any"][i] = 1.0 if (cond_rr_bad or cond_no_opp) else 0.0

    return pd.DataFrame(results, index=df.index)


# ============================================================
# 拆分 Failure 子标签训练函数
# ============================================================


def compute_bpc_failure_rr_extreme_label(
    df: pd.DataFrame,
    direction: Literal["long", "short"] = "long",
    horizon: int = 50,
    invert: bool = True,
    **kwargs,
) -> pd.Series:
    """
    只预测 "failure_rr_extreme"（踩大坑）的二值标签。

    🟥 核心语义：
    - failure_rr_extreme = forward_rr < -0.8R
    - 路径极端不利，亏损严重，通常意味着"踩了大坑"

    ❗ invert=True 时（默认）：
    - 输出: 1=好机会（不会踩坑），0=踩坑
    - 适配回测代码的 `preds >= threshold` 逻辑

    Args:
        df: 价格数据
        direction: 交易方向
        horizon: 持仓窗口
        invert: 是否反转标签

    Returns:
        pd.Series: 二值标签
    """
    subtypes = compute_failure_subtypes(
        df=df,
        direction=direction,
        horizon=horizon,
        price_col=kwargs.get("price_col", "close"),
        high_col=kwargs.get("high_col", "high"),
        low_col=kwargs.get("low_col", "low"),
        atr_col=kwargs.get("atr_col", "atr"),
        expected_stop_atr=kwargs.get("expected_stop_atr", 1.0),
        expected_target_atr=kwargs.get("expected_target_atr", 2.0),
    )

    failure_label = subtypes["failure_rr_extreme"]

    if invert:
        success_label = 1.0 - failure_label
        success_label.name = "success_no_rr_extreme"
        return success_label

    failure_label.name = "failure_rr_extreme"
    return failure_label


def compute_bpc_failure_no_opportunity_label(
    df: pd.DataFrame,
    direction: Literal["long", "short"] = "long",
    horizon: int = 50,
    invert: bool = True,
    **kwargs,
) -> pd.Series:
    """
    只预测 "failure_no_opportunity"（入场即反）的二值标签。

    🟥 核心语义：
    - failure_no_opportunity = (MAE > 1.2*stop) AND (MFE < 0.3*target)
    - 被打穿止损，且没给赚钱机会
    - 意味着"入场后立刻反向，没有任何转戴空间"

    ❗ invert=True 时（默认）：
    - 输出: 1=好机会（不会入场即反），0=入场即反
    - 适配回测代码的 `preds >= threshold` 逻辑

    Args:
        df: 价格数据
        direction: 交易方向
        horizon: 持仓窗口
        invert: 是否反转标签

    Returns:
        pd.Series: 二值标签
    """
    subtypes = compute_failure_subtypes(
        df=df,
        direction=direction,
        horizon=horizon,
        price_col=kwargs.get("price_col", "close"),
        high_col=kwargs.get("high_col", "high"),
        low_col=kwargs.get("low_col", "low"),
        atr_col=kwargs.get("atr_col", "atr"),
        expected_stop_atr=kwargs.get("expected_stop_atr", 1.0),
        expected_target_atr=kwargs.get("expected_target_atr", 2.0),
    )

    failure_label = subtypes["failure_no_opportunity"]

    if invert:
        success_label = 1.0 - failure_label
        success_label.name = "success_has_opportunity"
        return success_label

    failure_label.name = "failure_no_opportunity"
    return failure_label


# ============================================================
# 训练流水线适配函数
# ============================================================


def compute_bpc_failure_label(
    df: pd.DataFrame,
    archetype: str = "bpc",
    direction: Literal["long", "short"] = "long",
    horizon: int = 50,
    invert: bool = True,  # 默认输出 success_label，方便回测
    **kwargs,
) -> pd.Series:
    """
    BPC 策略的 Failure-first 标签（训练流水线适配）。

    这是 compute_failure_first_label 的包装函数，
    为 BPC 策略提供合适的默认参数。

    ❗ invert=True 时（默认）：
    - 输出 success_label: 1=好机会，0=失败
    - 适配回测代码的 `preds >= threshold` 逻辑

    ❗ invert=False 时：
    - 输出 failure_label: 1=失败，0=正常
    - 适合用于分析“哪里会失败”

    Example labels.yaml:
    ```yaml
    name: bpc_failure_first
    target_column: success_label  # invert=True 时

    label_generator:
      module: src.time_series_model.strategies.labels.failure_first_label
      function: compute_bpc_failure_label
      params:
        direction: long
        horizon: 50
        invert: true  # 1=好机会，0=失败
    ```
    """
    failure_label = compute_failure_first_label(
        df=df,
        direction=direction,
        horizon=horizon,
        price_col=kwargs.get("price_col", "close"),
        high_col=kwargs.get("high_col", "high"),
        low_col=kwargs.get("low_col", "low"),
        atr_col=kwargs.get("atr_col", "atr"),
        expected_stop_atr=kwargs.get("expected_stop_atr", 1.0),
        expected_target_atr=kwargs.get("expected_target_atr", 2.0),
    )

    if invert:
        # 反转为 success_label: 1=好机会，0=失败
        success_label = 1.0 - failure_label
        success_label.name = "success_label"
        return success_label
    return failure_label


# ============================================================
# Return Tree 标签（用于 GOOD 样本空间）
# ============================================================


def compute_return_tree_label(
    df: pd.DataFrame,
    direction: Literal["long", "short"] = "long",
    horizon: int = 50,
    filter_good_only: bool = True,
    **kwargs,
) -> pd.Series:
    """
    计算 Return Tree 标签：GOOD 样本的 forward_rr。

    🟢 用途：在已经排除 failure 的样本中，学习“如何让 RR 更大”

    Args:
        df: 价格数据，必须包含 OHLC 和 ATR
        direction: 交易方向
        horizon: 持仓窗口（bars）
        filter_good_only: 是否只返回 GOOD 样本（默认 True）

    Returns:
        pd.Series: forward_rr 值，失败样本为 NaN
    """
    # 计算 failure 子标签
    failure_df = compute_failure_subtypes(
        df=df,
        direction=direction,
        horizon=horizon,
        **kwargs,
    )

    # 提取 forward_rr
    forward_rr = failure_df["forward_rr"].copy()
    forward_rr.name = "forward_rr"

    if filter_good_only:
        # GOOD 样本 = ~failure_any
        failure_any = failure_df["failure_any"]
        # 将 failure 样本的 forward_rr 设为 NaN（会被过滤掉）
        forward_rr = forward_rr.where(failure_any == 0)

    return forward_rr


def compute_bpc_return_tree_label(
    df: pd.DataFrame,
    direction: Literal["long", "short"] = "long",
    horizon: int = 50,
    filter_good_only: bool = True,
    **kwargs,
) -> pd.Series:
    """
    BPC 策略的 Return Tree 标签（训练流水线适配）。

    🟢 Phase 2 核心任务：在 GOOD 样本中学习“如何让 RR 更大”

    输出：
    - forward_rr: 路径 RR 值
    - 失败样本自动过滤（failure_any = 1 的样本 forward_rr 为 NaN）

    GOOD 样本空间：
    - ~failure_rr_extreme: forward_rr >= -0.8R
    - ~failure_no_opportunity: 有机会的交易
    """
    return compute_return_tree_label(
        df=df,
        direction=direction,
        horizon=horizon,
        filter_good_only=filter_good_only,
        **kwargs,
    )
