非常好 👍
那我们就一步步来构建一个完整、**可落地的“从规则 → 特征 → LightGBM 信号生成”系统骨架**。
整个系统分为四层：

---

## 🧠 体系结构总览

```text
┌──────────────────────────────────────────┐
│  1️⃣ 特征层（Feature Layer）             │ ← 负责计算压缩、突破、体积、结构指标
│  └→ BreakoutQualityScorer + 其它指标输出 │
│                                          │
│  2️⃣ 数据集生成层（Dataset Builder）      │ ← 从历史Bar生成特征+标签
│  └→ 构造训练样本，保存到CSV/Parquet       │
│                                          │
│  3️⃣ 模型训练层（Model Trainer）           │ ← LightGBM 训练 + 校验 + 特征重要度
│                                          │
│  4️⃣ 实盘推理层（Signal Inference）        │ ← 实时计算特征 + 模型预测信号
└──────────────────────────────────────────┘
```

---

## 1️⃣ 特征层（Feature Layer）

我们先把你当前已有的指标（如 `BreakoutQualityScorer`、压缩指标、CVD 等）改造成输出多维特征的模块。

```python
# yin_bot/features/breakout_features.py
from typing import Dict
from nautilus_trader.model.data import Bar
from yin_bot.intraday_sniper.indicators.cvd import CVD
from yin_bot.intraday_sniper.indicators.compression import AdaptiveMultiDimCompressionIndicator
from nautilus_trader.indicators import BollingerBands, AverageTrueRange

class FeatureExtractor:
    def __init__(self):
        self.cvd = CVD()
        self.compression = AdaptiveMultiDimCompressionIndicator()
        self.bb = BollingerBands(period=20, std_dev=2)
        self.atr = AverageTrueRange(period=14)
    
    def on_bar(self, bar: Bar) -> Dict[str, float]:
        """每个bar更新指标并输出特征"""
        self.cvd.update(bar)
        self.compression.update(bar)
        self.bb.update(bar)
        self.atr.update(bar)
        
        bb_width = (self.bb.upper.value - self.bb.lower.value) / bar.close
        atr_norm = self.atr.value / bar.close
        
        features = {
            "cvd_slope": self.cvd.slope,
            "compression_score": self.compression.get_compression_score(),
            "breakout_quality": getattr(self.compression, "breakout_quality", 0.0),
            "bb_width": bb_width,
            "atr_norm": atr_norm,
            "bar_range": (bar.high - bar.low) / bar.close,
            "close_position": (bar.close - self.bb.middle.value) / (self.bb.upper.value - self.bb.lower.value + 1e-9),
        }
        return features
```

> 💡 目标：你所有的自定义指标，都最终汇总为一个 `Dict[str, float]`。

---

## 2️⃣ 数据集生成层（Dataset Builder）

我们把这些特征在历史数据上滑动计算，并生成未来收益的标签（比如未来5根K线的收益）。

```python
# yin_bot/training/dataset_builder.py
import pandas as pd
import numpy as np
from yin_bot.features.breakout_features import FeatureExtractor
from nautilus_trader.model.data import Bar

def build_dataset(bars: list[Bar], horizon: int = 5, return_threshold: float = 0.002):
    extractor = FeatureExtractor()
    rows = []

    for i in range(len(bars) - horizon):
        bar = bars[i]
        feats = extractor.on_bar(bar)
        # 未来收益率标签
        future_close = bars[i + horizon].close
        ret = (future_close - bar.close) / bar.close
        label = 1 if ret > return_threshold else 0  # 成功突破
        feats["label"] = label
        rows.append(feats)

    df = pd.DataFrame(rows)
    return df
```

---

## 3️⃣ 模型训练层（Model Trainer）

使用 LightGBM 自动学习哪些特征组合最能预测突破成功。

```python
# yin_bot/training/model_trainer.py
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import pandas as pd

def train_lightgbm_model(df: pd.DataFrame, output_path="models/breakout_model.txt"):
    X = df.drop(columns=["label"])
    y = df["label"]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

    params = {
        "objective": "binary",
        "learning_rate": 0.03,
        "num_leaves": 64,
        "max_depth": -1,
        "metric": ["auc", "binary_logloss"],
        "verbosity": -1,
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        early_stopping_rounds=50,
    )

    model.save_model(output_path)
    print(f"✅ Model saved to {output_path}")
    print("Feature importance:")
    print(pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importance()
    }).sort_values("importance", ascending=False))
```

---

## 4️⃣ 实盘推理层（Signal Inference）

在实盘运行时，只需在每个 `on_bar` 事件调用：

```python
# yin_bot/live/signal_inference.py
import lightgbm as lgb
import numpy as np
from yin_bot.features.breakout_features import FeatureExtractor

class SignalGenerator:
    def __init__(self, model_path="models/breakout_model.txt", threshold=0.6):
        self.extractor = FeatureExtractor()
        self.model = lgb.Booster(model_file=model_path)
        self.threshold = threshold

    def on_bar(self, bar):
        feats = self.extractor.on_bar(bar)
        X = np.array([list(feats.values())])
        prob = self.model.predict(X)[0]
        if prob > self.threshold:
            print(f"🔥 Breakout signal detected, prob={prob:.3f}")
            return True, prob
        return False, prob
```

---

## 5️⃣ 优化建议

| 模块               | 重点                        | 可拓展方向                            |
| ---------------- | ------------------------- | -------------------------------- |
| FeatureExtractor | 输出的每个特征必须稳定、无未来函数         | 用 TDigest / Wavelet / PCA 提取隐含特征 |
| DatasetBuilder   | Label 设计决定模型的“任务”         | 可用未来5/10/20 bar 的最大涨幅或方向         |
| ModelTrainer     | 可用 Optuna 自动调 LightGBM 超参 | 可加入特征重要度分析或 SHAP 解释              |
| SignalInference  | 与交易逻辑分离，便于替换模型            | 阈值可动态调整（基于历史信号质量）                |

---

## 🔍 接下来你可以决定两件事：

1. **你希望我帮你把这整套结构和你现有的指标类（BreakoutQualityScorer + Compression + CVD）整合起来生成完整代码吗？**
   这样你直接可以跑训练。

2. **你的目标标签定义想用哪种？**

   * ✅ 二分类（未来收益 > 某阈值）
   * 📈 连续预测（未来收益率）

这两个决定我才能帮你生成完整 pipeline（含特征 → 训练 → 实盘推理可用）。
你希望我帮你生成哪一种？

非常好的问题，这个是“**信号模型的核心选择点**”。
不同任务定义（分类 vs 回归）会直接影响信号的**表达方式、训练效果和实盘控制逻辑**。
我给你拆开讲清楚它们的逻辑、适用场景、优缺点和实盘使用方式👇

---

## 🧩 一、两种任务定义的根本区别

| 模式                            | 任务类型       | 输出含义      | 模型目标              | 示例                    |
| ----------------------------- | ---------- | --------- | ----------------- | --------------------- |
| **分类（Binary Classification）** | 判断突破是否“成功” | 概率（0~1）   | 最大化突破成功率（准确率/AUC） | “未来5根K线收益 > +0.3% 吗？” |
| **回归（Continuous Regression）** | 预测未来收益率    | 连续值（可能为负） | 精确预测未来收益期望        | “未来5根K线收益是多少？”        |

---

## 🧠 二、在量化信号系统里的不同意义

| 维度         | 二分类（收益>阈值）          | 连续预测（收益率）                      |
| ---------- | ------------------- | ------------------------------ |
| **输出形式**   | 概率（置信度）             | 预测收益（含方向）                      |
| **实盘信号解释** | `prob > 0.6 → 多头信号` | `pred > 0 → 多头, pred < 0 → 空头` |
| **决策方式**   | 触发/不触发（离散）          | 方向 + 强度（连续）                    |
| **阈值敏感性**  | 依赖收益阈值设定（如0.002）    | 依赖收益波动尺度                       |
| **训练稳定性**  | 更稳健，收敛快             | 更难，但表达力更强                      |
| **过拟合风险**  | 较低                  | 较高（尤其在噪音行情）                    |
| **可解释性**   | 简单直观（突破成功率）         | 适合组合打分（预期收益）                   |
| **常见用途**   | 突破检测、进场确认           | 目标价预测、仓位大小控制                   |

---

## 🧩 三、实盘中的应用方式对比

### 🟩 **(1) 二分类模型**

👉 输出为突破成功概率 `p ∈ [0,1]`

```python
prob = model.predict(features)
if prob > 0.6:
    signal = "BUY"
elif prob < 0.4:
    signal = "SELL"
else:
    signal = "HOLD"
```

* 你可以直接把 `p` 用作“信号置信度”；
* 可以作为 **过滤层**：

  * 只有当突破置信度 > 0.6 且结构确认时才开仓；
  * 非突破状态则不动；
* 可以方便地与规则叠加（规则过滤 + 模型确认）。

**优点：**

* 训练快速；
* 鲁棒性强；
* 容易理解；
* 适合“要不要做”的决策。

**缺点：**

* 无法区分强弱突破；
* 阈值(例如未来收益0.002)设错会导致模型失效。

---

### 🟦 **(2) 连续收益预测模型**

👉 输出为 `predicted_return ∈ [-0.01, +0.01]` 这样的连续值。

```python
pred_ret = model.predict(features)[0]
if pred_ret > 0.001:
    size = min(1.0, pred_ret / 0.01)  # 动态仓位
    signal = ("BUY", size)
elif pred_ret < -0.001:
    signal = ("SELL", abs(pred_ret) / 0.01)
else:
    signal = ("HOLD", 0)
```

* 预测未来收益的**方向 + 强度**；
* 可用于：

  * 多空方向决策；
  * 动态仓位控制；
  * 风控模型（预测波动率、Sharpe等）；
* 适合组合信号系统（可与多因子线性/非线性融合）。

**优点：**

* 表达力强；
* 可直接用于仓位决策；
* 可扩展为多任务模型（收益 + 波动率）。

**缺点：**

* 训练更敏感；
* 容易受极端样本干扰；
* 需要更稳定的特征工程。

---

## 🧩 四、推荐选择策略（结合你的系统特征）

你当前系统特征：

* 指标特征很多（压缩、CVD、BreakoutQuality、POC方向等）；
* 主要目标是**识别结构性突破信号**；
* 交易周期在 5min 以上（中低频）；
* 想提高置信度、减少虚假突破。

➡️ **推荐主模型：二分类（未来收益 > 阈值）**

原因：

* 更稳健；
* 易调试；
* 便于与结构规则结合；
* 可先优化突破成功率，再在上层加入收益预测或回归增强。

然后你可以：

* 在分类模型稳定后，再增加一个回归头（预测收益幅度）；
* 构成“突破概率 × 预期收益”的混合打分。

---

## 🧩 五、最优组合方案（推荐生产级做法）

**多任务模型结构：**

| 模型输出                                     | 含义        | 用途          |
| ---------------------------------------- | --------- | ----------- |
| `p_breakout`                             | 是否突破成功的概率 | 作为过滤信号      |
| `pred_return`                            | 未来收益的期望   | 用于仓位大小控制    |
| `final_score = p_breakout * pred_return` | 综合信号强度    | 作为排序指标或执行阈值 |

这种组合的逻辑非常像：

> “先判断它是不是好机会 → 再估计它能赚多少 → 决定要不要重仓。”

LightGBM 支持多输出（或训练两模型叠加）。

---

## ✅ 建议总结

| 目标             | 建议使用          | 备注        |
| -------------- | ------------- | --------- |
| 判断是否突破、过滤虚信号   | 二分类模型         | 稳健、解释性强   |
| 判断方向 + 强度、动态仓位 | 连续预测模型        | 对冲仓位、回测优化 |
| 想兼顾两者          | 混合方案（分类 + 回归） | 最优生产实践    |

---

如果你希望，我可以帮你：

1. 基于前面那套结构生成一个**二分类版本 pipeline**；
2. 然后再给你一个可选的 **连续预测版本**（共享特征层）；
3. 最后附带一个 **混合信号融合模块**（final_score = p × r）。

你希望我直接生成哪一个版本的完整实现骨架？

* 🟩 A. 二分类（未来收益 > 阈值）
* 🟦 B. 连续预测（收益率）
* 🟨 C. 混合版（分类 + 回归，最终综合打分）

# 先判断它是不是好机会 → 再估计它能赚多少 → 决定要不要重仓。

非常棒 ✅ ——你选的是**行业最先进、实盘最稳定的信号生成逻辑**，也就是：

> **多阶段决策模型（Two-Stage Signal Model）**
>
> 👉 第一阶段：判断是否为高质量机会（分类）
> 👉 第二阶段：估计潜在收益幅度（回归）
> 👉 最终信号：
> [
> \text{final_score} = p_{\text{breakout}} \times \max(pred_return, 0)
> ]
> （再根据打分决定是否建仓、加仓或跳过）

---

## 🧩 整体结构

我们来搭一套完整、可落地的工程骨架：

```text
yin_bot/
│
├── features/
│   └── feature_extractor.py      # 所有特征计算逻辑（CVD, Compression, BB, ATR 等）
│
├── training/
│   ├── dataset_builder.py        # 构造特征 + label
│   ├── train_classifier.py       # 训练突破成功概率模型
│   ├── train_regressor.py        # 训练收益幅度预测模型
│
├── live/
│   └── signal_inference.py       # 实盘信号推理模块
│
└── models/
    ├── breakout_classifier.txt
    └── return_regressor.txt
```

---

## 1️⃣ 特征层：FeatureExtractor

（与之前类似，只是输出结构化特征）

```python
# yin_bot/features/feature_extractor.py
from typing import Dict
from nautilus_trader.model.data import Bar
from yin_bot.intraday_sniper.indicators.cvd import CVD
from yin_bot.intraday_sniper.indicators.compression import AdaptiveMultiDimCompressionIndicator
from nautilus_trader.indicators import BollingerBands, AverageTrueRange

class FeatureExtractor:
    def __init__(self):
        self.cvd = CVD()
        self.compression = AdaptiveMultiDimCompressionIndicator()
        self.bb = BollingerBands(period=20, std_dev=2)
        self.atr = AverageTrueRange(period=14)

    def on_bar(self, bar: Bar) -> Dict[str, float]:
        self.cvd.update(bar)
        self.compression.update(bar)
        self.bb.update(bar)
        self.atr.update(bar)

        bb_width = (self.bb.upper.value - self.bb.lower.value) / bar.close
        atr_norm = self.atr.value / bar.close

        return {
            "cvd_slope": self.cvd.slope,
            "compression_score": self.compression.get_compression_score(),
            "breakout_quality": getattr(self.compression, "breakout_quality", 0.0),
            "bb_width": bb_width,
            "atr_norm": atr_norm,
            "bar_range": (bar.high - bar.low) / bar.close,
            "close_pos": (bar.close - self.bb.middle.value) / (self.bb.upper.value - self.bb.lower.value + 1e-9),
        }
```

---

## 2️⃣ 数据集生成层：DatasetBuilder

输出两个 label：

* `label_class` → 是否突破成功（收益 > 阈值）
* `label_return` → 未来收益率（回归用）

```python
# yin_bot/training/dataset_builder.py
import pandas as pd
from yin_bot.features.feature_extractor import FeatureExtractor
from nautilus_trader.model.data import Bar

def build_dataset(bars: list[Bar], horizon: int = 5, return_threshold: float = 0.002):
    extractor = FeatureExtractor()
    rows = []

    for i in range(len(bars) - horizon):
        bar = bars[i]
        feats = extractor.on_bar(bar)
        future_close = bars[i + horizon].close
        ret = (future_close - bar.close) / bar.close
        feats["label_class"] = 1 if ret > return_threshold else 0
        feats["label_return"] = ret
        rows.append(feats)

    return pd.DataFrame(rows)
```

---

## 3️⃣ 分类模型：BreakoutClassifier（判断是不是好机会）

```python
# yin_bot/training/train_classifier.py
import lightgbm as lgb
from sklearn.model_selection import train_test_split

def train_breakout_classifier(df, output_path="models/breakout_classifier.txt"):
    X = df.drop(columns=["label_class", "label_return"])
    y = df["label_class"]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

    params = {
        "objective": "binary",
        "learning_rate": 0.03,
        "num_leaves": 64,
        "metric": ["auc"],
        "verbosity": -1,
    }

    model = lgb.train(
        params,
        lgb.Dataset(X_train, label=y_train),
        valid_sets=[lgb.Dataset(X_val, label=y_val)],
        num_boost_round=1000,
        early_stopping_rounds=50,
    )
    model.save_model(output_path)
    print(f"✅ Classifier saved to {output_path}")
```

---

## 4️⃣ 回归模型：ReturnRegressor（预测能赚多少）

```python
# yin_bot/training/train_regressor.py
import lightgbm as lgb
from sklearn.model_selection import train_test_split

def train_return_regressor(df, output_path="models/return_regressor.txt"):
    X = df.drop(columns=["label_class", "label_return"])
    y = df["label_return"]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

    params = {
        "objective": "regression",
        "learning_rate": 0.03,
        "num_leaves": 64,
        "metric": ["l2", "mae"],
        "verbosity": -1,
    }

    model = lgb.train(
        params,
        lgb.Dataset(X_train, label=y_train),
        valid_sets=[lgb.Dataset(X_val, label=y_val)],
        num_boost_round=1000,
        early_stopping_rounds=50,
    )
    model.save_model(output_path)
    print(f"✅ Regressor saved to {output_path}")
```

---

## 5️⃣ 实盘信号融合模块：SignalInference

```python
# yin_bot/live/signal_inference.py
import lightgbm as lgb
import numpy as np
from yin_bot.features.feature_extractor import FeatureExtractor

class SignalGenerator:
    def __init__(
        self,
        cls_path="models/breakout_classifier.txt",
        reg_path="models/return_regressor.txt",
        prob_threshold=0.6,
        score_threshold=0.0008,
    ):
        self.extractor = FeatureExtractor()
        self.cls_model = lgb.Booster(model_file=cls_path)
        self.reg_model = lgb.Booster(model_file=reg_path)
        self.prob_th = prob_threshold
        self.score_th = score_threshold

    def on_bar(self, bar):
        feats = self.extractor.on_bar(bar)
        X = np.array([list(feats.values())])

        prob = self.cls_model.predict(X)[0]        # 突破成功概率
        pred_ret = self.reg_model.predict(X)[0]    # 预期收益率
        score = prob * max(pred_ret, 0)

        signal = None
        if prob > self.prob_th and score > self.score_th:
            signal = ("BUY", score)
        elif prob > self.prob_th and pred_ret < -self.score_th:
            signal = ("SELL", score)

        return {
            "prob": prob,
            "pred_ret": pred_ret,
            "score": score,
            "signal": signal,
        }
```

---

## 6️⃣ 实盘逻辑（执行侧）

可以根据最终 `score` 决定仓位强度或跳过交易：

```python
result = signal_gen.on_bar(bar)
if result["signal"]:
    side, score = result["signal"]
    size = min(1.0, score / 0.01)  # 动态仓位
    execute_trade(side=side, size=size)
```

---

## ⚙️ 七、改进方向建议

| 模块   | 可增强点                                        | 效果      |
| ---- | ------------------------------------------- | ------- |
| 特征层  | 增加波动压缩分位、成交量爆发分位、POC方向、微结构特征                | 更高信号可分性 |
| 数据集  | 使用未来 N 根最高价/最低价，定义更稳健的 `label_return`       | 减少噪音    |
| 模型   | 用 Optuna 调整 `num_leaves`、`min_data_in_leaf` | 提升性能    |
| 推理层  | 加滑动平滑（EMA）或置信度累积机制                          | 减少假突破触发 |
| 训练评估 | 同时输出特征重要度 + SHAP值                           | 理解模型逻辑  |

---

## ✅ 结论总结

| 阶段      | 模型                          | 输出                   | 实盘用途         |
| ------- | --------------------------- | -------------------- | ------------ |
| Stage 1 | BreakoutClassifier          | `p_breakout` (成功概率)  | 是否做          |
| Stage 2 | ReturnRegressor             | `pred_return` (预期收益) | 做多/做空 + 动态仓位 |
| Fusion  | Final Score = p × max(r, 0) | 综合打分                 | 排序 / 策略决策    |

---

是否希望我帮你：

1. **整合为一个完整可运行的训练+预测 pipeline（含 CSV 输入输出与回测用例）**？
2. 还是先帮你在你已有的 `BreakoutQualityScorer` 框架中接入这个多阶段信号结构？

这两种方向略有不同：

* ✅ **pipeline版本**：可以独立训练和预测，用于实验和模型对比；
* ⚙️ **整合版本**：直接融入你现有策略系统，可实时出信号。

你想要我先帮你做哪一个版本？

# pipeline

非常好 👍 你这个需求属于**“分层决策 Pipeline”结构化建模**，常见于量化 Alpha 策略系统（尤其是强化学习或多阶段模型）。
你的逻辑是典型的三阶段：

> **Stage 1：机会识别（分类）** → **Stage 2：收益预估（回归）** → **Stage 3：仓位决策（权重映射）**

下面是完整 pipeline 的设计与实现示例（可直接跑 optuna 或 sklearn pipeline 优化）。

---

## 🧩 一、结构设计图

```
┌───────────────────────────────┐
│  Feature Engineering (X)     │  ← 技术指标、压缩因子、tdigest特征、小波特征等
└───────────────┬──────────────┘
                │
                ▼
┌───────────────────────────────┐
│  Stage 1: Opportunity Classifier│ → 输出 P(good_opportunity)
│  (LightGBM / XGBoost / CatBoost)│
└───────────────┬──────────────┘
                │
         if P > threshold:
                ▼
┌───────────────────────────────┐
│  Stage 2: Return Estimator (Regressor)│ → 预测未来收益率
│  (LightGBMRegressor / XGBRegressor)   │
└───────────────┬──────────────┘
                │
                ▼
┌───────────────────────────────┐
│  Stage 3: Position Sizer │ → 根据期望收益/风险调整仓位
│  (position = f(predicted_return, vol)) │
└───────────────────────────────┘
```

---

## 🧠 二、代码示例（Sklearn + LightGBM Pipeline）

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ==== 示例数据 ====
# X: 你的特征（技术指标、压缩因子、小波能量等）
# y_class: 二分类标签 (未来收益 > 阈值)
# y_reg: 连续目标 (未来收益率)
X = np.random.randn(10000, 20)
y_reg = np.random.randn(10000)
y_class = (y_reg > 0.002).astype(int)

X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
    X, y_class, y_reg, test_size=0.2, random_state=42
)

# ==== Stage 1: 分类模型 ====
clf = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    ))
])

clf.fit(X_train, y_class_train)
p_opportunity = clf.predict_proba(X_test)[:, 1]

# ==== Stage 2: 回归模型 ====
mask_good = p_opportunity > 0.6  # 只对“高置信度机会”样本进行回归预测
X_good = X_test[mask_good]
y_good = y_reg_test[mask_good]

reg = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LGBMRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    ))
])

reg.fit(X_good, y_good)
pred_return = reg.predict(X_good)

# ==== Stage 3: 仓位决策 ====
# 例：动态仓位分配函数
def position_size(pred, vol):
    """根据预期收益和波动率动态调整仓位"""
    risk_adjusted = pred / (vol + 1e-6)
    return np.clip(risk_adjusted * 5, 0, 1)  # 最多满仓，最少0

vol = np.std(pred_return)
positions = position_size(pred_return, vol)

# 输出结果
result = pd.DataFrame({
    "p_opportunity": p_opportunity[mask_good],
    "pred_return": pred_return,
    "position": positions
})
print(result.head(10))
```

---

## ⚙️ 三、核心逻辑总结

| 阶段      | 模型类型 | 输入         | 输出       | 决策逻辑      |
| ------- | ---- | ---------- | -------- | --------- |
| Stage 1 | 分类   | 全部样本       | 是否“值得交易” | 过滤掉低置信度区域 |
| Stage 2 | 回归   | 高置信度样本     | 未来收益率预测  | 用于仓位分配    |
| Stage 3 | 规则   | 回归输出 + 波动率 | 仓位权重     | 动态风险控制    |

---

## 🔮 四、增强版本建议

1. **Stacking / Meta-learning**

   * Stage1 + Stage2 输出一起喂入强化学习或 MLP 再决定仓位。
2. **Optuna 自动化调参**

   * 你可以单独对 Stage1、Stage2 调参，也可以联合 pipeline。
3. **实时更新机制**

   * 随着市场结构变化（压缩区 regime shift），定期重训模型。
4. **标签自适应**

   * 阈值 `0.002` 可以改为波动率自适应收益率阈值，比如 `> 0.5 * ATR`。

---

是否希望我帮你把这个 pipeline 封装成一个**可直接用于回测/实盘预测**的类版本（含 fit/predict/position）？
那样就能直接接入你现在的因子生成模块（压缩特征、POC方向、CVD等）。

太好了 👍 我来帮你把刚才的 pipeline 封装成一个**可直接用于回测/实盘预测**的完整类版本，
你可以直接接入自己当前的特征生成模块（如 `AdaptiveMultiDimCompressionIndicator`, `BreakoutQualityScorer`, `CVD`, `POC` 等）。

---

## 🧱 完整版：`OpportunityPipeline`

这个类结构清晰、可回测、可预测、可与 Optuna 调优整合。
你只需喂入特征矩阵 `X`、收益标签 `y_reg` 即可。

```python
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


class OpportunityPipeline:
    """
    三阶段量化决策模型：
    Stage 1: 判断是否好机会（二分类）
    Stage 2: 估计未来收益（回归）
    Stage 3: 根据预期收益与波动率分配仓位
    """

    def __init__(
        self,
        class_threshold: float = 0.6,
        max_position: float = 1.0,
        min_position: float = 0.0,
        return_clip: float = 0.05,
        random_state: int = 42,
    ):
        self.class_threshold = class_threshold
        self.max_position = max_position
        self.min_position = min_position
        self.return_clip = return_clip
        self.random_state = random_state

        self.classifier = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LGBMClassifier(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state
            ))
        ])

        self.regressor = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LGBMRegressor(
                n_estimators=400,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state
            ))
        ])

    # === Stage 1: 分类 ===
    def fit_classifier(self, X: np.ndarray, y_reg: np.ndarray, threshold: float = 0.001):
        y_class = (y_reg > threshold).astype(int)
        self.classifier.fit(X, y_class)

    # === Stage 2: 回归 ===
    def fit_regressor(self, X: np.ndarray, y_reg: np.ndarray):
        self.regressor.fit(X, y_reg)

    # === 联合训练 ===
    def fit(self, X: np.ndarray, y_reg: np.ndarray, threshold: float = 0.001):
        self.fit_classifier(X, y_reg, threshold)
        y_prob = self.classifier.predict_proba(X)[:, 1]
        mask = y_prob > self.class_threshold
        if mask.sum() > 10:
            self.fit_regressor(X[mask], y_reg[mask])
        else:
            print("[Warning] Not enough high-confidence samples for regressor training.")

    # === 预测 ===
    def predict(self, X: np.ndarray):
        prob = self.classifier.predict_proba(X)[:, 1]
        mask_good = prob > self.class_threshold

        pred_return = np.zeros(len(prob))
        if mask_good.any():
            pred_return[mask_good] = self.regressor.predict(X[mask_good])

        positions = self._compute_position(pred_return)
        return pd.DataFrame({
            "p_good": prob,
            "pred_return": pred_return,
            "position": positions
        })

    # === 仓位函数 ===
    def _compute_position(self, pred_return: np.ndarray):
        pred_return = np.clip(pred_return, -self.return_clip, self.return_clip)
        mean_abs = np.mean(np.abs(pred_return)) + 1e-6
        weights = pred_return / mean_abs
        pos = np.clip(weights, self.min_position, self.max_position)
        return pos


# ==== 示例用法 ====
if __name__ == "__main__":
    X = np.random.randn(10000, 20)
    y_reg = np.random.randn(10000) * 0.005  # 模拟未来收益率

    X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)

    pipe = OpportunityPipeline(class_threshold=0.6)
    pipe.fit(X_train, y_train)

    result = pipe.predict(X_test)
    print(result.head(10))
```

---

## 🚀 模型解释

| 阶段      | 模型                       | 输出            | 用途             |
| ------- | ------------------------ | ------------- | -------------- |
| Stage 1 | LightGBMClassifier       | `p_good`      | 是否是好机会         |
| Stage 2 | LightGBMRegressor        | `pred_return` | 未来收益率估计        |
| Stage 3 | 内部函数 `_compute_position` | `position`    | 仓位控制（重仓/轻仓/空仓） |

---

## ⚙️ 接入你现有系统的方法

```python
features = your_feature_builder.generate(bar_data)
returns = your_label_maker.future_return(bar_data, horizon=10)

pipeline = OpportunityPipeline(class_threshold=0.65)
pipeline.fit(features, returns)
pred_df = pipeline.predict(features_latest)

# pred_df 包含：
# - p_good：机会置信度
# - pred_return：预期收益
# - position：建议仓位
```

---

## 🔮 下一步可增强版本

1. ✅ **Optuna集成**：同时调参 `class_threshold`, `max_depth`, `learning_rate`。
2. ✅ **在线更新**：`partial_fit` 或定期重训。
3. ✅ **多时间尺度模型**：5m, 15m, 1h 各训练一套 pipeline。
4. ✅ **风险校正目标**：用 *Sharpe-adjusted returns* 替代原始收益作为 `y_reg`。

---

是否希望我给你扩展一个 **Optuna 自动调参版本**（自动寻找分类/回归两个模型的最优结构 + 阈值）？
可以直接一键跑出最优 pipeline 参数配置用于实盘。
