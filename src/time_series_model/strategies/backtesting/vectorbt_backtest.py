"""
VectorBT 通用回测类

将现有的 run_vectorbt_backtest 封装为类，继承 BaseBacktest。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .base_backtest import BaseBacktest
from src.time_series_model.pipeline.training.label_utils import simulate_rr_exits


class VectorBTBacktest(BaseBacktest):
    """使用 vectorbt 的通用回测实现。"""

    def run(
        self,
        df: pd.DataFrame,
        predictions: np.ndarray,
        task_type: str = "binary",
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        params = kwargs or {}
        enabled = params.get("enabled", True)
        if not enabled:
            return None

        try:
            import vectorbt as vbt  # type: ignore
        except ImportError:
            print("   ⚠️  vectorbt not installed. Skipping backtest.")
            return None

        price_col = params.get("price_col", "close")
        if price_col not in df.columns:
            print(f"   ⚠️  Price column '{price_col}' not found. Skipping backtest.")
            return None

        price = df[price_col].astype(float)
        fee = params.get("fee", 0.0004)
        slippage = params.get("slippage", 0.0)
        init_cash = params.get("initial_cash", 10000.0)

        index = df.index

        debug = bool(params.get("debug", False))
        use_signal_direction = bool(params.get("use_signal_direction", False))
        signal_col = params.get("signal_col", "signal")
        use_rr_exit = bool(params.get("use_rr_exit", False))

        if task_type == "multiclass" and predictions.ndim == 2:
            class_preds = np.argmax(predictions, axis=1)
            multi_cfg = params.get("multiclass", {})
            long_class = multi_cfg.get("long_class", 2)
            short_class = multi_cfg.get("short_class", 0)
            neutral_class = multi_cfg.get("neutral_class", 1)
            long_entries = pd.Series(class_preds == long_class, index=index)
            long_exits = pd.Series(class_preds == neutral_class, index=index)
            short_entries = pd.Series(class_preds == short_class, index=index)
            short_exits = pd.Series(class_preds == neutral_class, index=index)
        else:
            long_entry = params.get("long_entry_threshold", 0.6)
            long_exit = params.get("long_exit_threshold", 0.4)
            short_entry = params.get("short_entry_threshold", 0.4)
            short_exit = params.get("short_exit_threshold", 0.6)

            preds_series = pd.Series(predictions, index=index)

            if use_signal_direction and signal_col in df.columns:
                # SR reversal 等策略：方向由 signal 决定，preds 只控制是否入场
                signal_series = df[signal_col].fillna(0).astype(float)

                base_long_entries = preds_series >= long_entry
                base_short_entries = preds_series <= short_entry

                long_entries = (signal_series > 0) & base_long_entries
                short_entries = (signal_series < 0) & base_short_entries

                # 初始的概率退出，后续可能被 RR 覆盖
                long_exits = preds_series <= long_exit
                short_exits = preds_series >= short_exit
            else:
                # 仅根据预测得分构造多空信号
                long_entries = preds_series >= long_entry
                long_exits = preds_series <= long_exit
                short_entries = preds_series <= short_entry
                short_exits = preds_series >= short_exit

            if debug:
                debug_signals = pd.DataFrame(
                    {
                        "price": price,
                        "pred": preds_series,
                        "long_entry": long_entries,
                        "long_exit": long_exits,
                        "short_entry": short_entries,
                        "short_exit": short_exits,
                    }
                )

        # 动态仓位（可选，基于 ATR 控制每笔风险）
        long_size, short_size = self._compute_position_sizes(
            df=df,
            price=price,
            params=params,
            rr_params=params.get("rr", {}),
            init_cash=init_cash,
            index=index,
        )

        # RR 退出逻辑：使用与标签一致的 simulate_rr_exits
        if use_rr_exit:
            if not use_signal_direction:
                raise ValueError(
                    "use_rr_exit=True requires use_signal_direction=True so direction is defined by signal"
                )

            rr_params = params.get("rr", {})
            rr_max_holding_bars = int(rr_params.get("max_holding_bars", 24))
            rr_stop_loss_r = float(rr_params.get("stop_loss_r", 1.0))
            rr_take_profit_r = float(rr_params.get("take_profit_r", 2.0))
            rr_atr_window = int(rr_params.get("atr_window", 14))
            rr_entry_offset = int(rr_params.get("entry_offset", 1))
            rr_entry_price_col = rr_params.get("entry_price_col", None)

            # 构造仅包含被模型选中的信号方向列：1=多，-1=空
            rr_signal = pd.Series(0.0, index=index)
            rr_signal[long_entries] = 1.0
            rr_signal[short_entries] = -1.0

            df_rr = df.copy()
            df_rr[signal_col] = rr_signal

            long_exits_rr, short_exits_rr = simulate_rr_exits(
                df_rr,
                signal_col=signal_col,
                price_col=price_col,
                atr_col=params.get("atr_col", "atr"),
                atr_window=rr_atr_window,
                max_holding_bars=rr_max_holding_bars,
                stop_loss_r=rr_stop_loss_r,
                take_profit_r=rr_take_profit_r,
                entry_price_col=rr_entry_price_col,
                entry_offset=rr_entry_offset,
            )

            long_exits = long_exits_rr.reindex(index).fillna(False)
            short_exits = short_exits_rr.reindex(index).fillna(False)

        # 频率：用于 vectorbt 计算年化指标
        freq = params.get("freq", None)
        if freq is None:
            if isinstance(index, pd.DatetimeIndex):
                inferred_freq = index.inferred_freq
                if inferred_freq:
                    freq = inferred_freq
                elif len(index) > 1:
                    time_diff = index[1] - index[0]
                    sec = time_diff.total_seconds()
                    if sec == 900:
                        freq = "15T"
                    elif sec == 3600:
                        freq = "1H"
                    elif sec == 14400:
                        freq = "4H"
                    elif sec == 86400:
                        freq = "1D"
            if freq is None:
                raise ValueError(
                    "❌ 'freq' must be configured in backtest params for vectorbt metrics."
                )

        try:
            portfolio = vbt.Portfolio.from_signals(
                price,
                entries=long_entries,
                exits=long_exits,
                short_entries=short_entries,
                short_exits=short_exits,
                init_cash=init_cash,
                fees=fee,
                slippage=slippage,
                freq=freq,
                size=long_size,
                short_size=short_size,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"   ⚠️  Backtest failed: {exc}")
            return None

        stats = portfolio.stats()

        result: Dict[str, Any] = {
            "total_return_pct": float(stats.get("Total Return [%]", 0.0)),
            "sharpe": float(stats.get("Sharpe Ratio", 0.0)),
            "max_drawdown_pct": float(stats.get("Max Drawdown [%]", 0.0)),
            "win_rate": float(stats.get("Win Rate [%]", 0.0)),
            "total_trades": int(stats.get("Total Trades", 0)),
        }

        if debug:
            debug_payload: Dict[str, Any] = {}
            try:
                trades = portfolio.trades.records_readable
            except Exception:
                trades = None

            if trades is not None and not trades.empty:
                n_trades = int(len(trades))
                n_win = int((trades["PnL"] > 0).sum())
                win_rate_manual = 100.0 * n_win / n_trades if n_trades > 0 else 0.0
                trades_sample = (
                    trades.sort_values("Entry Timestamp")
                    .head(200)
                    .reset_index(drop=True)
                )
                debug_payload["trades"] = trades_sample.to_dict(orient="records")
                debug_payload["trades_meta"] = {
                    "n_trades": n_trades,
                    "n_win": n_win,
                    "win_rate_manual": win_rate_manual,
                }

            if "debug_signals" in locals():
                entry_mask = long_entries | short_entries
                signals_sample = (
                    debug_signals[entry_mask]
                    .head(200)
                    .reset_index()
                    .rename(columns={"index": "timestamp"})
                )
                debug_payload["signals"] = signals_sample.to_dict(orient="records")

            try:
                returns = portfolio.returns()
                mean_ret = float(returns.mean())
                std_ret = float(returns.std())
                debug_payload["returns_stats"] = {"mean": mean_ret, "std": std_ret}
            except Exception:
                pass

            result["debug"] = debug_payload

        return result

    def _compute_position_sizes(
        self,
        df: pd.DataFrame,
        price: pd.Series,
        params: Dict[str, Any],
        rr_params: Dict[str, Any],
        init_cash: float,
        index: pd.Index,
    ) -> tuple[pd.Series, pd.Series]:
        """
        基于 ATR 的动态仓位（可选）。

        逻辑：
        - 若未配置 position_sizing 或 type=none，则返回全1仓位（等权）。
        - type="atr_risk": 以每笔风险占初始资金的比例控制 size。
          size = (init_cash * risk_pct) / (stop_loss_r * atr * price)
          并可通过 max_size_cap 限制最大 size。
        """
        sizing = params.get("position_sizing", {}) or {}
        sizing_type = sizing.get("type", "none")

        # 默认等权
        default_size = pd.Series(1.0, index=index)
        if sizing_type == "none":
            return default_size, default_size

        if sizing_type == "atr_risk":
            atr_col = sizing.get("atr_col", "atr")
            atr_window = int(sizing.get("atr_window", rr_params.get("atr_window", 14)))
            risk_pct = float(sizing.get("risk_pct", 0.01))  # 1% 风险
            max_size_cap = float(sizing.get("max_size_cap", 10.0))

            # 确保 ATR 存在
            if atr_col not in df.columns:
                high_col = params.get("high_col", "high")
                low_col = params.get("low_col", "low")
                close_col = params.get("price_col", "close")
                if all(col in df.columns for col in (high_col, low_col, close_col)):
                    high = df[high_col]
                    low = df[low_col]
                    close = df[close_col]
                    tr = pd.concat(
                        [
                            high - low,
                            (high - close.shift(1)).abs(),
                            (low - close.shift(1)).abs(),
                        ],
                        axis=1,
                    ).max(axis=1)
                    atr = tr.rolling(window=atr_window, min_periods=1).mean()
                else:
                    atr = pd.Series(1.0, index=index)
            else:
                atr = df[atr_col]

            stop_loss_r = float(rr_params.get("stop_loss_r", 1.0))
            # size 按价格和 ATR 反比，限制上限
            raw_size = (init_cash * risk_pct) / (stop_loss_r * atr * price)
            raw_size = raw_size.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            raw_size = raw_size.clip(upper=max_size_cap)
            return raw_size, raw_size

        # 其他类型未实现，回退等权
        return default_size, default_size
