"""
改进的方向判断方法

当前问题：
1. 使用预测值分位数判断方向，但预测值本身可能不准确
2. 阈值太极端（0.9/0.1），导致信号太少
3. Win Rate 只有 13.5%，说明方向判断不准确

改进方案：
1. 直接使用预测值符号（如果预测值 > 0 → Long，< 0 → Short）
2. 动态阈值优化（基于历史表现优化阈值）
3. 结合预测值和分位数
4. 校准预测值
"""

from __future__ import annotations

from typing import Optional, Literal
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import accuracy_score


def generate_trading_signals_improved(
    predictions: pd.Series,
    pred_quantile: pd.Series,
    confidence_score: pd.Series,
    true_returns: Optional[pd.Series] = None,
    method: Literal["quantile", "sign", "hybrid", "optimized"] = "hybrid",
    confidence_threshold: float = 0.85,
    long_threshold: float = 0.9,
    short_threshold: float = 0.1,
    optimize_on_train: bool = True,
) -> pd.Series:
    """
    改进的交易信号生成方法

    Args:
        predictions: 原始预测值（可以是任何值，不一定是 0-1）
        pred_quantile: 预测值分位数（0-1）
        confidence_score: 置信度分数（0-1）
        true_returns: 真实收益（用于优化阈值，可选）
        method: 信号生成方法
            - "quantile": 使用分位数（当前方法）
            - "sign": 直接使用预测值符号
            - "hybrid": 结合预测值符号和分位数
            - "optimized": 基于历史表现优化阈值
        confidence_threshold: 最小置信度阈值
        long_threshold: 做多分位数阈值
        short_threshold: 做空分位数阈值
        optimize_on_train: 是否在训练数据上优化阈值

    Returns:
        交易信号：1 (Long), -1 (Short), 0 (Hold)
    """
    signals = pd.Series(0, index=predictions.index, dtype=int)

    if method == "quantile":
        # 当前方法：使用分位数
        long_mask = (pred_quantile >= long_threshold) & (
            confidence_score >= confidence_threshold
        )
        short_mask = (pred_quantile <= short_threshold) & (
            confidence_score >= confidence_threshold
        )
        signals.loc[long_mask] = 1
        signals.loc[short_mask] = -1

    elif method == "sign":
        # 方法1：直接使用预测值符号
        # 如果预测值 > 0 → Long，< 0 → Short
        # 但需要置信度过滤
        high_confidence = confidence_score >= confidence_threshold

        # 使用预测值符号
        pred_sign = np.sign(predictions)

        # 结合置信度
        long_mask = (pred_sign > 0) & high_confidence
        short_mask = (pred_sign < 0) & high_confidence

        signals.loc[long_mask] = 1
        signals.loc[short_mask] = -1

    elif method == "hybrid":
        # 方法2：结合预测值符号和分位数
        # 只有当预测值符号和分位数方向一致时才交易
        high_confidence = confidence_score >= confidence_threshold

        # 预测值符号
        pred_sign = np.sign(predictions)

        # 分位数方向
        quantile_long = pred_quantile >= long_threshold
        quantile_short = pred_quantile <= short_threshold

        # 只有当符号和分位数方向一致时才交易
        long_mask = (pred_sign > 0) & quantile_long & high_confidence
        short_mask = (pred_sign < 0) & quantile_short & high_confidence

        signals.loc[long_mask] = 1
        signals.loc[short_mask] = -1

    elif method == "optimized":
        # 方法3：基于历史表现优化阈值
        if true_returns is not None and optimize_on_train:
            # 优化预测值阈值
            def objective(threshold):
                """优化目标：最大化方向准确率"""
                pred_dir = (predictions > threshold).astype(int)
                true_dir = (true_returns > 0).astype(int)

                # 只考虑高置信度的样本
                high_conf_mask = confidence_score >= confidence_threshold
                if high_conf_mask.sum() == 0:
                    return 1.0  # 如果没有高置信度样本，返回最差分数

                pred_dir_conf = pred_dir[high_conf_mask]
                true_dir_conf = true_dir[high_conf_mask]

                if len(pred_dir_conf) == 0:
                    return 1.0

                acc = accuracy_score(true_dir_conf, pred_dir_conf)
                return 1.0 - acc  # 最小化错误率

            # 优化阈值
            result = minimize_scalar(
                objective,
                bounds=(predictions.min(), predictions.max()),
                method="bounded",
            )
            optimal_threshold = result.x

            # 使用优化后的阈值
            high_confidence = confidence_score >= confidence_threshold
            long_mask = (predictions > optimal_threshold) & high_confidence
            short_mask = (predictions < optimal_threshold) & high_confidence

            signals.loc[long_mask] = 1
            signals.loc[short_mask] = -1
        else:
            # 如果没有真实收益，回退到 sign 方法
            return generate_trading_signals_improved(
                predictions,
                pred_quantile,
                confidence_score,
                method="sign",
                confidence_threshold=confidence_threshold,
            )

    return signals


def calibrate_predictions(
    predictions: pd.Series,
    true_returns: pd.Series,
    method: Literal["isotonic", "platt", "sigmoid"] = "sigmoid",
) -> pd.Series:
    """
    校准预测值，使其更准确地反映真实收益

    Args:
        predictions: 原始预测值
        true_returns: 真实收益
        method: 校准方法
            - "isotonic": 等渗回归（需要 sklearn）
            - "platt": Platt scaling（逻辑回归）
            - "sigmoid": 简单 sigmoid 缩放

    Returns:
        校准后的预测值
    """
    if method == "sigmoid":
        # 简单 sigmoid 缩放：将预测值映射到合理范围
        # 使用历史数据的均值和标准差
        pred_mean = predictions.mean()
        pred_std = predictions.std()

        # 标准化
        pred_normalized = (predictions - pred_mean) / (pred_std + 1e-8)

        # Sigmoid 映射到 [-1, 1]
        pred_calibrated = np.tanh(pred_normalized)

        # 缩放到真实收益的尺度
        return_mean = true_returns.mean()
        return_std = true_returns.std()
        pred_scaled = pred_calibrated * return_std + return_mean

        return pd.Series(pred_scaled, index=predictions.index)

    elif method == "platt":
        # Platt scaling: 使用逻辑回归校准
        try:
            from sklearn.linear_model import LogisticRegression

            # 准备数据
            X = predictions.values.reshape(-1, 1)
            y = (true_returns > 0).astype(int)

            # 训练逻辑回归
            lr = LogisticRegression()
            lr.fit(X, y)

            # 预测概率
            prob = lr.predict_proba(X)[:, 1]

            # 转换为预测值（概率 - 0.5）* 2，映射到 [-1, 1]
            pred_calibrated = (prob - 0.5) * 2

            # 缩放到真实收益的尺度
            return_mean = true_returns.mean()
            return_std = true_returns.std()
            pred_scaled = pred_calibrated * return_std + return_mean

            return pd.Series(pred_scaled, index=predictions.index)
        except ImportError:
            # 如果没有 sklearn，回退到 sigmoid
            return calibrate_predictions(predictions, true_returns, method="sigmoid")

    else:
        # 默认使用 sigmoid
        return calibrate_predictions(predictions, true_returns, method="sigmoid")


def evaluate_direction_accuracy(
    predictions: pd.Series,
    true_returns: pd.Series,
    method: str = "sign",
    threshold: Optional[float] = None,
) -> dict:
    """
    评估方向预测准确性

    Args:
        predictions: 预测值
        true_returns: 真实收益
        method: 判断方法（"sign" 或 "threshold"）
        threshold: 阈值（如果 method="threshold"）

    Returns:
        包含准确率、精确率、召回率等的字典
    """
    # 真实方向
    true_dir = (true_returns > 0).astype(int)

    # 预测方向
    if method == "sign":
        pred_dir = (predictions > 0).astype(int)
    elif method == "threshold" and threshold is not None:
        pred_dir = (predictions > threshold).astype(int)
    else:
        pred_dir = (predictions > predictions.median()).astype(int)

    # 计算指标
    accuracy = accuracy_score(true_dir, pred_dir)

    # 计算混淆矩阵
    tp = ((pred_dir == 1) & (true_dir == 1)).sum()
    fp = ((pred_dir == 1) & (true_dir == 0)).sum()
    tn = ((pred_dir == 0) & (true_dir == 0)).sum()
    fn = ((pred_dir == 0) & (true_dir == 1)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }
