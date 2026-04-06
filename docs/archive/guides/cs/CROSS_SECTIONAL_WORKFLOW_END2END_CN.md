# CS 全流程（End‑to‑End）：FeatureStore → 因子评估 → 筛选 → 训练 → Walk‑Forward → 回测审计

本文是 `src/cross_sectional/` 的“一张图说明书”，把你关心的 **selected_factors.txt、CS Pipeline Report、回测与交易审计** 串在一条线上。

---

## 0) 最小可跑命令（建议照抄）

你只需要两步：先把特征算进 FeatureStore（缓存），再跑 workflow（会生成报告与审计产物）。

### Step A：构建 CS FeatureStore（Alpha101-CS）

> 注意：symbols 请用逗号分隔；timeframe 4H 用 `240T`；日期按你的研究窗调整。

```bash
mlbot cross-section build-store --no-docker \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT \
  --timeframe 240T \
  --start-date 2025-07-01 \
  --end-date 2025-12-31 \
  --factor-set-yaml config/cross_sectional/cs_factor_sets_crypto.yaml \
  --factor-set crypto_alpha101_cs_rank \
  --features-store-layer cs_alpha101_cs_rank_4h_v1 \
  --warmup-bars 600
```

### Step B：一键跑 CS workflow（推荐）

> 这个命令会把 pipeline 串起来：panel(feature_store) → factor_eval → select → report → train → walk-forward 回测 + 审计日志 → index.html

```bash
mlbot cross-section workflow --no-docker \
  --config config/cross_sectional/pipeline_alpha101_cs_rank_4h_feature_store.yaml
```

### Step C：打开最终报告 + 审计日志

- 最终报告（总入口）：`results/cross_sectional/pipeline_alpha101_cs_rank_4h/index.html`
- 因子筛选结果：`results/cross_sectional/pipeline_alpha101_cs_rank_4h/selected_factors.txt`
- OOS 模型回测指标：`results/cross_sectional/pipeline_alpha101_cs_rank_4h/train/model_bt_metrics__<mode>.json`
- **按 rebalance 一笔的审计日志**：
  - `.../train/model_rebalance_log__<mode>.csv`
  - `.../train/factor_combo_rebalance_log__<mode>.csv`

其中 `<mode>`：
- `long_only`
- `market_neutral`

---

## 1) 你会得到什么（产物清单）

以 `output_root=results/cross_sectional/<run_name>/` 为例：

- `factor_eval/summary.csv`：每个因子的 IC/IR + long/short 组合指标（含 turnover/fee/Sharpe）
- `selected_factors.txt`：自动筛选出来的因子列表（运行时产物）
- `fama_macbeth_report.md`：可选的回归报告（更偏统计解释）
- `train/`：可选的模型训练与回测产物
  - `metrics.json`：模型在 OOS 的 IC / rank‑IC
  - `model_bt_metrics__<mode>.json`：模型信号的 OOS 组合回测结果（Sharpe/回撤等）
  - `factor_combo_bt_metrics__<mode>.json`：因子组合 baseline 的 OOS 组合回测结果
  - `*_rebalance_log__<mode>.csv`：**按 rebalance 一笔**的审计日志（持仓/换手/成本/多空名单）
- `index.html`：CS Pipeline Report（上述所有产物的总入口）

其中 `<mode>` 目前支持：
- `long_only`
- `market_neutral`（多空净暴露≈0）

---

## 2) selected_factors.txt 是什么？为什么它是“运行时产物”

`selected_factors.txt` 的来源：`src/cross_sectional/scripts/auto_select_factors.py`

它是 **基于某次运行的数据窗口 + universe + horizon + 成本假设** 自动筛出来的结果。

这就是你看到它不是“静态 YAML 配置”的原因：
- **静态 YAML**：你提前规定“用哪些因子”
- **selected_factors.txt**：系统在当前窗口里评估后告诉你“这一轮更可能有效/更可交易的因子”

> 换句话说：它更像 TS 的 `factor-eval` 之后导出的候选池，而不是模型输入配置本身。

如果你希望统一格式，我们可以额外导出 `selected_factors.yaml`（但它依然是“输出产物”）。

---

## 3) CS Pipeline Report（index.html）是干嘛的？和 ts-factor-eval 对应吗？

它的定位：**CS 版本的 factor-eval + selection +（可选）train/backtest 的总报告页**。

和 `ts-factor-eval` 的关系：
- 都是“因子有效性筛选”
- CS 里更强调“排序→建仓→换仓”，所以天然需要：
  - long/short spread
  - turnover + fee
  - Sharpe（gross/net）

生成代码：`src/cross_sectional/scripts/pipeline.py`

---

## 4) 两种回测：factor-eval vs 模型回测

### 3.1 因子回测（factor-eval 内置）
代码：
- `src/cross_sectional/factor_backtest.py`
- 调用者：`src/cross_sectional/scripts/factor_eval.py`

逻辑（每根 bar）：按因子值排序 → top/bottom 等权 → 计算 turnover → 扣 fee。

### 3.2 模型回测（更像实盘）
代码：
- 执行引擎：`src/cross_sectional/model_portfolio_backtest.py`
- 调用者：`src/cross_sectional/scripts/train_cross_sectional_model.py`

核心执行假设（可 YAML 配置）：
- `holding_period_bars = H`（每 H 根 bar 才换仓一次）
- `execution_lag_bars = 1`（信号延迟一根 bar 才执行）
- `long_only` / `market_neutral`
- `top_k/bottom_k`、`gross_leverage`、`max_weight`、`cash_buffer`
- 成本：`fee_bps + slippage_bps`（按换手计）
- 做空更真实：`funding_bps_per_bar/borrow_bps_per_bar`
- equity：`compound/log/simple` + `max_drawdown`

---

## 5) “Sharpe(net) model vs factor-combo” 是什么意思？factor-combo 是什么？

为了回答“到底是因子没效果，还是模型映射有问题”，我们在 OOS 回测里做了一个 baseline：

- **model**：用模型预测值作为信号（`predictions`）
- **factor-combo**：把 `selected_factors.txt` 里的因子做截面 z-score 后取均值，作为一个“多因子合成信号”

同一套执行假设下对比两者的 Sharpe：
- 如果 model 明显优于 factor-combo：模型确实学到了非线性/交互
- 如果 model 不如 factor-combo：可能是模型过拟合或映射方式不合适

---

## 6) 回测的“交易审计”：rebalance log（按 rebalance 一笔）

每次换仓会输出一行 CSV（足够审计）：
- `rebalance_ts`、`signal_ts`
- `long_symbols_json`、`short_symbols_json`
- `gross_exposure/net_exposure/turnover/trade_cost/funding_cost`

产物路径：
- `train/model_rebalance_log__<mode>.csv`
- `train/factor_combo_rebalance_log__<mode>.csv`

---

## 7) 一键运行（建议）

```bash
mlbot cross-section workflow --no-docker \
  --config config/cross_sectional/pipeline_alpha101_cs_rank_4h_feature_store.yaml
```

打开报告：
- `results/cross_sectional/pipeline_alpha101_cs_rank_4h/index.html`

---

## 8) 常见改参入口（都在 YAML 里）

编辑：`config/cross_sectional/pipeline_alpha101_cs_rank_4h_feature_store.yaml`

- **改时间窗/币种**：`panel.feature_store.symbols / start_date / end_date`
- **改因子集合**：`factor_eval.factor_set_yaml + factor_eval.factor_set`
- **改筛选阈值**：`select.*`
- **改 walk-forward**：`train.walk_forward.folds / embargo_bars`
- **改“像实盘”的执行假设**：`train.backtest_cfg.*`
  - `holding_period_bars`（持有期=H）
  - `execution_lag_bars`（延迟执行）
  - `mode`（long_only / market_neutral）
  - `top_k/bottom_k`、`gross_leverage`、`max_weight`、`cash_buffer`
  - `fee_bps/slippage_bps/funding_bps_per_bar/borrow_bps_per_bar`

---

## 9) `config/cross_sectional/` 目录里的文件分别是干嘛的？

目前目录里主要是 3 个“入口配置”：

### 9.1 `cs_factor_sets_crypto.yaml`：CS 因子集合清单（命名集合）

- **作用**：定义一组组可复用的因子列名（`factor_sets.*`），供 `factor-eval / pipeline / build-store` 选择。
- **典型用法**：
  - 在 pipeline 里指定：
    - `factor_eval.factor_set_yaml: config/cross_sectional/cs_factor_sets_crypto.yaml`
    - `factor_eval.factor_set: crypto_alpha101_cs_rank`（或 `crypto_cs_core`、`crypto_ts_compatible_core`）
  - 在 build-store 里指定：
    - `--factor-set-yaml config/cross_sectional/cs_factor_sets_crypto.yaml --factor-set crypto_alpha101_cs_rank`

### 9.2 （已移除）`pipeline_example_crypto_4h.yaml` / build-panel 示例

为了减少分支与维护成本，我们移除了 `mlbot cross-section build-panel`，并删除了 `pipeline_example_crypto_4h.yaml`。

如果你想固定某次拼出来的面板做复现/调试：
- 用 FeatureStore 跑 `workflow/pipeline` 时，默认会落：`output_root/panel_from_feature_store.parquet`
- 后续把 `panel.source` 切到 `parquet` 直接读这个快照即可（与 FeatureStore 后续增量/补月解耦）

### 9.3 `pipeline_alpha101_cs_rank_4h_feature_store.yaml`：Alpha101‑CS 完整 workflow（panel=feature_store）

- **作用**：这是我们当前“可直接跑”的主配置：
  - panel 从 FeatureStore 加载（可复用、可增量）
  - 做 `factor_eval → select → report → train → walk-forward → OOS backtest → audit logs → index.html`
- **适用场景**：你希望跑全币种/大 universe，且希望“第一次慢、第二次复用”。
- **典型命令**：

```bash
mlbot cross-section build-store --no-docker \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT \
  --timeframe 240T \
  --start-date 2025-07-01 --end-date 2025-12-31 \
  --factor-set-yaml config/cross_sectional/cs_factor_sets_crypto.yaml \
  --factor-set crypto_alpha101_cs_rank \
  --features-store-layer cs_alpha101_cs_rank_4h_v1 \
  --warmup-bars 600

mlbot cross-section workflow --no-docker \
  --config config/cross_sectional/pipeline_alpha101_cs_rank_4h_feature_store.yaml
```

> 小贴士：如果你经常遇到 “我改了 start_date/end_date 但 report 没变”，通常是因为 FeatureStore layer 里缺失对应月份分区文件。
> 现在 pipeline 支持可选的自动补齐：在 YAML 里加上（默认不启用）：
>
> ```yaml
> panel:
>   source: feature_store
>   feature_store: { ... }
>   auto_build_store:
>     enabled: true
>     data_path: data/parquet_data
>     # 若不填，会默认使用 factor_eval.factor_set_yaml / factor_eval.factor_set
>     # factor_set_yaml: config/cross_sectional/cs_factor_sets_crypto.yaml
>     # factor_set: crypto_alpha101_cs_rank
>     feature_deps: config/feature_dependencies.yaml
>     warmup_bars: 600
>     include_ohlcv: true
>     overwrite: false
> ```

---

## 10) select 的算法细节：per_category_top vs global_top（以及为什么默认用 ic/ir）

这部分对应 `src/cross_sectional/scripts/auto_select_factors.py`，选择逻辑是“两阶段筛选”：

### 10.1 第一步：按类别筛选（per_category_top）
1) **列分组**：把 panel 的列名按启发式规则分组（例如 `cs_crypto_*`、其它列会归到 `other`；Alpha101-CS 通常在 `other`）。
2) **计算指标**：对每个类别内的候选因子，计算截面 IC/IR（target 默认是 `future_return_<horizon>`）。
3) **类别内筛选**：
   - 先按阈值过滤：`ic_threshold`/`ir_threshold`（可以只设一个）
   - 再按 `ranking_stat` 排序：`ic` 或 `ir`
   - 取前 `per_category_top` 个进入“候选池”

目的：避免所有名额被某一类因子垄断（例如全是某一组 alpha 或某一组 cs_crypto）。

### 10.2 第二步：全局再筛一次（global_top）
把所有类别入围因子合并为一个候选集合，然后：
1) **重新统一口径计算** 候选集合的 IC/IR（同一 target、同一 `min_assets`）
2) 再次应用阈值与排序（同上）
3) 取前 `global_top` 个作为最终 `selected_factors.txt`

目的：控制最终因子数量，并做一次“总决赛”。

### 10.3 ranking_stat 为什么默认用 ic/ir，而不是 sharpe？
当前 `select` 阶段默认用 **IC/IR**（`ranking_stat: ic|ir`），原因：
- IC/IR 更稳定、且不依赖组合构建细节（topK、holding、lag、成本）
- Sharpe 更像“交易层指标”，应在 **OOS backtest** 中评估（我们已在 `index.html` 做 model vs factor-combo + 审计 log）

如果你希望“按 Sharpe 选”：
- 可以作为下一步扩展：让 `select` 读取 `factor_eval/summary.csv` 的 `sharpe_net` 排序筛选（更交易导向，但也更依赖执行假设）。

---

## 11) factor_set 是否应该默认“全选”？

不建议默认全选，原因：
- CS 面板列可能非常多（尤其混入其它特征/中间列），全选会引入大量噪声与过拟合风险
- 你的研究目标通常是“某一类因子族”（例如 Alpha101-CS、cs_crypto、或一组 TS 兼容指标）

推荐做法：
- 在 `factor_eval` 里显式指定 `factor_set_yaml + factor_set`（例如 `crypto_alpha101_cs_rank`）
- 如果确实想“全选”：建议用 **多个 factor_set 合并**（更可控、可复现），或者用 `factors_file` 显式列出候选列名（最严格）。

另外：`factor_set` 现在支持“多个集合合并”，两种写法等价：

```yaml
factor_eval:
  factor_set_yaml: config/cross_sectional/cs_factor_sets_crypto.yaml
  factor_set: crypto_alpha101_cs_rank,crypto_cs_core
```

或：

```yaml
factor_eval:
  factor_set_yaml: config/cross_sectional/cs_factor_sets_crypto.yaml
  factor_set:
    - crypto_alpha101_cs_rank
    - crypto_cs_core
```

含义：把两个集合的列名 **做 union**，作为候选池，再进入 `select` 阶段筛选出 `selected_factors.txt`。

---

## 12) “FeatureStore 增量/缺失月会影响什么？”以及为什么 parquet 仍有价值

你说“一个数据源更容易调试”是对的；我们现在的主路径就是 **FeatureStore 一个数据源**。

我说的“增量/缺失月影响”主要指两件事：
- **缺失月/缺失 symbol**：FeatureStore 是按 `YYYY-MM` 分区落盘的。如果某些 symbol 在某些月没有数据/没有生成分区文件，拼出来的 panel 会在那段时间里“少一些资产”（assets_per_timestamp 变动），这会影响 IC/回测结果。
- **增量构建带来的样本变化**：你后续补算了缺失月/新增了 symbol/新增了特征列，同一个 date range 从 FeatureStore 拼出来的 panel 可能会变“更完整”，从而导致评估结果变化。

为什么 parquet 仍有价值：**用于固定住一个“快照 panel”做复现/对比**。

另外：即便你用 feature_store，pipeline 也会把当次拼出来的面板保存为：
- `output_root/panel_from_feature_store.parquet`

这就是一个“快照”，你可以用它反复复现/调参，而不受 FeatureStore 后续增量变化影响。


