## 特征计算复杂性分层（让你先跑通主流程，再逐步“解锁”重特征）

你现在的痛点很典型：订单流 + 数学特征（DTW / 频谱 / WPT / Hilbert）一起算，**既慢、又难定位瓶颈、还很难快速迭代**，导致主流程（训练→预测→Router→E2E）一直跑不通。

这份文档给你一个**可执行的“分层策略”**：
- 先用“便宜特征层”把流程跑通、验证 primitives 是否有信息
- 再用 ABC（A/B/C 三阶段）逐层引入更贵的特征
- 每层只做 **一个变量**：把“贵特征”当作 *unlockable modules*，而不是默认全量开

---

## 0) 核心原则（强烈建议遵守）

- **原则 0：先跑通主流程**  
  主流程能稳定产出 report，比一次性追求“全特征最强”更重要。
- **原则 1：特征按计算复杂性分层**  
  你在做的是“工程可维护性优先”的研究：先便宜、后昂贵。
- **原则 2：每次只解锁一层（或一个 block）**  
  这样你才能回答：到底是“信号变好”还是“纯粹算得更慢”。
- **原则 3：先踢掉明显没用的，再考虑昂贵特征**  
  用 A/B 快筛把明显无效的候选踢掉，再给昂贵特征上算力。

---

## 1) 分层定义（Tier 0 → Tier 3）

下面是一个工程导向的复杂性分层（不追求学术完美，只追求“能落地迭代”）。

### Tier 0（最便宜）：OHLCV + 轻量 rolling（默认先从这里开始）

**特征类型**：
- 仅依赖 bar 级 OHLCV（close/high/low/volume），轻量 rolling
- 不依赖 ticks
- 不依赖 DTW / Spectrum / WPT / Hilbert

**代表性 feature nodes（举例）**：
- `atr_f`, `rsi_f`, `macd_f`, `bb_width_f`
- `roc_5_f`, `acceleration_3_f`
- `range_ratio_5bar_f`, `wick_ratios_f`, `price_range_symmetry_f`
- `trend_r2_20_f`, `trend_r2_50_f`
- `volume_anomaly_f`

**建议用途**：
- 先做 nnmultihead 的 baseline 训练/评估
- 先跑通 Router / E2E（不追求最强，只追求稳定和速度）

---

### Tier 1（中等）：bar 级“语义/结构/压缩/SR质量/分布特征”（推荐第二层）

**特征类型**：
- 仍然是 bar 级（可以更复杂：压缩、SR质量、语义评分、VPVR 等）
- 可能更重，但通常仍可控
- 不强依赖 ticks（有些可能会依赖派生列，如 `cvd`，需看你的数据管线）

**代表性 feature nodes（举例）**：
- 压缩：`compression_duration_f`, `compression_to_breakout_prob_f`, `compression_score_f`
- SR质量：`sr_strength_*`, `dist_to_nearest_sr*`, `direction_to_nearest_sr*`（以你 repo 实际节点为准）
- 成交量分布：`volume_profile_vpvr_f`, `volume_profile_volatility_features_f`
- 语义：`*_scene_semantic_scores_f`（如 wick/volume_profile/funding 等）

**建议用途**：
- 作为“你原始策略语义”的主要承载层：压缩 + SR质量 + 结构评分
- 用 ABC 先筛、再固化到 `features.yaml`（required/optional_blocks）

---

### Tier 2（重）：数学/信号处理类（DTW / Spectrum / WPT / Hilbert）

**特征类型**：
- 计算复杂度高（滚动频谱、DTW 模板距离、WPT 分解等）
- 很容易成为“跑不动”的主因

**代表性 feature nodes（举例）**：
- `dtw_features_*`
- `spectrum_*`（包含 price/volume/cvd 的频谱特征）
- `wpt_*`
- `hilbert_*`

**建议用途**：
- 先不要默认全开
- 只有在 Tier0/1 已经能跑通且有信息时，再逐个 block 解锁
- 搜索阶段优先用 Fast Mode（见下）

**可用加速开关（只影响 DTW/Spectrum）**：
- `FEATURE_FAST_MODE=1`：DTW 关闭 random templates；Spectrum 低频计算 + forward fill  
  你也可以通过 CLI：`mlbot nnmultihead build-feature-store --fast-features ...`

---

### Tier 3（最重）：tick/orderflow 类（VPIN / trade_cluster / footprint / order_flow）

**特征类型**：
- 依赖 ticks 或高频聚合
- IO+CPU 都重，最容易把整个 pipeline 拖死

**代表性 feature nodes（举例）**：
- `vpin_*`
- `trade_cluster_*`
- `footprint_basic_f`
- `order_flow_all_features_f`, `ofi_short_f`, `market_cap_normalized_orderflow_f`
- `funding_rate_features_f`（取决于实现，通常也较重）

**建议用途**：
- 作为“最后解锁层”：先验证 primitives 框架有效，再叠加 orderflow
- 开月度并行（见下），但 worker 不要太大（先 2-4）

---

## 2) 推荐迭代流程（把“跑通”变成硬约束）

### Step A：只跑 Tier0（把主链路跑通）

目标：训练/预测/E2E 都能稳定跑，形成“可复盘产物”。

### Step B：解锁 Tier1（压缩 + SR质量 + bar语义）

目标：用 ABC 找到“稳定改善”的组合，再固化到 `features.yaml`。

### Step C：解锁 Tier2（数学特征）——一次只加一个 block

目标：回答“数学特征是否真的对 primitives 有用”，避免一次加一堆导致不可归因。

### Step D：解锁 Tier3（orderflow/ticks）——最后再上

目标：让 orderflow 变成“增益模块”，而不是“全局拖慢系统的默认项”。

---

## 3) 加速与工程建议（你现在立刻能用）

### 3.1 月度并行（opt-in）：只并行 monthly cache miss

你现在的特征栈支持 **按月并行**（不是特征级 DAG 并行）：
- 并行粒度：单月切片（更稳定）
- 适用：tick-heavy 特征非常吃这个

建议从 2-4 开始：

```bash
mlbot nnmultihead build-feature-store --no-docker \
  --feature-monthly-workers 4 \
  --feature-monthly-backend process \
  ...
```

### 3.2 Fast Mode（用于 DTW/Spectrum 快速试错）

```bash
mlbot nnmultihead build-feature-store --no-docker \
  --fast-features \
  ...
```

### 3.3 把昂贵特征放进 optional_blocks（而不是 required）

你的 `features.yaml` 支持 `required/optional_blocks`：
- **Tier0/1 放 required**（保证主链路稳定）
- **Tier2/3 放 optional_blocks**（可按阶段解锁、可做 mask 鲁棒性）

---

## 4) 你现在最应该做的（最短路径）

- 先固定一个 **Tier0/1-only** 的 `features.yaml`，把 nn 主流程跑通（训练→预测→Router→E2E）
- 再用 **ABC**（A/B/C 三阶段）只在 Tier1 候选里找“最稳的一套”
- 然后再逐个解锁：`DTW`（fast mode）→ `Spectrum`（拆分 price/volume/cvd）→ `Orderflow ticks`

