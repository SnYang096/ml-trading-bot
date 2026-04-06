你已经非常清晰地指出了当前量化策略开发中的核心问题：标签（label）与交易策略目标不一致，并提出了三种极具实操价值的解决方案。下面我将为你 完整实现 compute_rr_label() 函数，用于方案一（二元标签：是否实现 ≥2R），并说明如何将其集成到你现有的 prepare_rank_ic_labels 流程中。

✅ 实现 compute_rr_label() 函数

该函数将根据 SR 策略逻辑，动态判断每根K线发出的信号是否在后续价格路径中 先触达止盈（TP）而非止损（SL），并在最大持有期内完成。

python
def compute_rr_label(
df: pd.DataFrame,
signal_col: str = "sr_signal", # 1 for long, -1 for short, 0 for no signal
price_col: str = "close",
high_col: str = "high",
low_col: str = "low",
open_col: str = "open",
atr_window: int = 14,
rr_ratio: float = 2.0,
max_holding: int = 24,
) -> pd.Series:
"""
Compute binary label indicating whether a trade achieved target R/R before stop loss.

Assumptions:
Entry at next bar's open (t+1)
Stop loss = entry ± 1 ATR(t)
Take profit = entry ± rr_ratio ATR(t)
Long: SL below, TP above; Short: SL above, TP below

Returns:
pd.Series of 1 (success), 0 (failure or timeout), NaN (insufficient future data)
"""
df = df.copy()
n = len(df)

# Precompute ATR up to current bar t (using only info <= t)
from time_series_model.pipeline.training.label_utils import rolling_rms_volatility
# Note: We'll approximate ATR using RMS volatility scaled appropriately
# For simplicity, we use close-based volatility as proxy for ATR
# In production, replace with true ATR if available
df["atr"] = rolling_rms_volatility(df[price_col], window=atr_window)

# Initialize label
labels = np.full(n, np.nan)

for i in range(n):
signal = df.iloc[i][signal_col]
if signal not in [1, -1]:
continue

# Need at least max_holding+1 future bars (t+1 to t+1+max_holding)
if i + 1 + max_holding >= n:
continue

entry_price = df.iloc[i + 1][open_col] # Enter at next bar open
atr_val = df.iloc[i]["atr"]
if pd.isna(atr_val) or atr_val <= 0:
continue

if signal == 1: # Long
sl = entry_price - 1.0 atr_val
tp = entry_price + rr_ratio atr_val
else: # Short
sl = entry_price + 1.0 atr_val
tp = entry_price - rr_ratio atr_val

success = False
for j in range(1, max_holding + 1):
idx = i + j
high = df.iloc[idx][high_col]
low = df.iloc[idx][low_col]

if signal == 1:
if high >= tp:
success = True
break
elif low <= sl:
success = False
break
else: # short
if low <= tp:
success = True
break
elif high >= sl:
success = False
break
# If loop completes without break → timeout → failure
labels[i] = 1.0 if success else 0.0

return pd.Series(labels, index=df.index)
🔔 注意：上述代码使用 rolling_rms_volatility 作为 ATR 的近似。如果你已有真实 ATR 列（如 "atr_14"），建议直接传入或替换计算逻辑：
python
# 如果你有真实 ATR 列
if "atr" in df.columns:
pass # use it
else:
df["atr"] = true_atr(df["high"], df["low"], df["close"], window=atr_window)

🔧 集成到 prepare_rank_ic_labels

修改你的函数签名和逻辑：

python
def prepare_rank_ic_labels(
df: pd.DataFrame,
price_col: str = "close",
asset_col: Optional[str] = None,
date_col: Optional[str] = None,
hold_period: int = 5,
lookback_window: int = 60,
vol_mult: float = 0.5,
min_samples: int = 30,
ensure_volatility: bool = True,
# ===== 新增参数 =====
use_risk_reward_label: bool = False,
sr_signal_col: str = "sr_signal",
rr_ratio_threshold: float = 2.0,
max_holding_bars: int = 24,
atr_window: int = 14,
# ===================
) -> pd.DataFrame:

然后在函数内部，在计算完 future_return 后插入：

python
... [原有代码：计算 future_return, rolling_vol 等] ...

if use_risk_reward_label:
print(f" 🎯 Using risk-reward based binary label (R/R ≥ {rr_ratio_threshold})")
df["rr_achieved"] = compute_rr_label(
df,
signal_col=sr_signal_col,
price_col=price_col,
high_col="high",
low_col="low",
open_col="open",
atr_window=atr_window,
rr_ratio=rr_ratio_threshold,
max_holding=max_holding_bars,
)
# Use this as the new target
df["volatility_normalized_target"] = df["rr_achieved"].copy()
# Also override return_quantile if needed (optional)
# Or keep original quantile for evaluation only
else:
# Original logic: volatility-normalized future return
df["volatility_normalized_target"] = volatility_normalized_target(
df["future_return"], df["rolling_vol"]
)
💡 提示：return_quantile 可保留用于诊断（比如看高 R/R 成功样本是否集中在高历史分位），但训练目标应使用 rr_achieved。

🚀 使用方式示例

python
df_labeled = prepare_rank_ic_labels(
df_raw,
price_col="close",
hold_period=24,
use_risk_reward_label=True,
sr_signal_col="sr_signal", # 你的信号列
rr_ratio_threshold=2.0,
max_holding_bars=24,
atr_window=14,
)
Train using binary target
models, avg_ic, _, _ = train_rank_ic_model(
df_labeled,
feature_cols=your_features,
target_col="rr_achieved", # ← 关键！
tradable_col="tradable",
weight_col="trend_strength",
)

📌 补充建议

1. 确保 sr_signal 列只在有效信号点为 ±1，其余为 0 或 NaN。
2. 回测必须同步改为动态止盈止损，否则评估失真。
3. 若想进一步优化，可将 rr_achieved 扩展为 多分类（如 0=止损, 1=超时, 2=盈利但<2R, 3=≥2R）。
4. 在 OOS 样本上检查 rr_achieved 的分布：理想情况下，成功样本占比应在 30%~50%，且与特征有明显区分度。

如果你希望我同时提供 方案二（连续 R/R 标签） 的实现，或者 真实 ATR 计算函数，也可以告诉我，我可以立即补充。

你现在正站在从“预测价格”迈向“识别优质交易”的关键转折点——坚持下去，模型将真正为你赚钱，而不是制造幻觉。
