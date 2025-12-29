# ML Trading Bot

本仓库包含因子研究、降维、模型训练和实盘回测堆栈的生产就绪组件。`src/time_series_model/` 下的代码包含可复用的 Python 包；`scripts/` 目录仅暴露最小的一组命令行入口点，用于包装包 API。

## 快速开始

1. 创建虚拟环境（conda、venv 等）并激活它。
2. 以可编辑模式安装项目：
   ```bash
   pip install -e .[dev]
   ```
3. 安装 Git pre-commit 钩子（可选但推荐）：
   ```bash
   make install-hooks
   ```
   这会在每次提交前自动运行 `mlbot dev format` 和 `mlbot dev lint` 以确保代码质量。
4. 运行帮助命令验证安装：
   ```bash
   mlbot --help
   ```
   或查看所有可用命令：
   ```bash
   mlbot analyze --help
   mlbot train --help
   mlbot diagnose --help
   mlbot optimize --help
   ```

## 推荐使用流程

### 核心工作流（配置驱动架构）

推荐的工作流使用**配置驱动架构**，具有特定策略的配置：

1. **特征分析** (`mlbot analyze factor-eval`): 评估因子并选择最优特征
2. **策略训练** (`mlbot train sr-reversal-long`, `mlbot train sr-reversal-short`, `mlbot train rolling`): 使用策略配置训练模型
3. **消融研究** (`mlbot analyze strategy-feature-compare`): 比较特征配置（可选）
4. **回测** (`mlbot backtest vectorbot`): 验证策略性能

**📖 详细工作流请参见 [完整流程指南](docs/时序模型/完整流程指南.md)。**

### 架构入口（建议先读）

- **工业化 Experiment Loop（Layer A/B/C、TaskSpec、Filter→Wrapper、稳定性证据口径）**：`docs/architecture/EXPERIMENT_LOOP_ARCHITECTURE.md`
- **NN 多头 Path Primitives + Router→Execution（NO/MEAN/TREND）**：`docs/时序模型/架构：NN多头路径原语（Path Primitives）+Router解耦升级.md`

快速心智模型：
- **PolicyTask（直接开仓）**：模型直接输出开仓信号/分数，研究闭环最快（常见于树模型）。
- **PrimitivesTask（路径原语→执行）**：先训练共享 Router（dir/mfe/mae/t），再由 Execution 在强 safety 约束下映射到 NO/MEAN/TREND；复用性更强、长期更稳。

### 分步工作流

#### 为什么 SR Reversal 要拆成 Long/Short（以及为什么移除了 `sr_reversal/`）

我们现在刻意使用 **两个方向固定的策略**：
- `sr_reversal_long`：只做多，二元标签 = “在 \(t+1\) 进多单后，是否先到 +2R 再到 -1R”
- `sr_reversal_short`：只做空，二元标签 = “在 \(t+1\) 进空单后，是否先到 +2R 再到 -1R”

这替代了旧的 **双向** 配置（`sr_reversal/`, `combine_mode: any_success`），后者把做多/做空的结果混在同一个标签里。

**标签语义的关键差异**：
- **旧（`any_success`）**：对每根 K 线同时假设做多和做空两种入场；只要任意一个方向“成功”，标签就可能为 1。这会让 `pred` 很难解释成“某个具体方向的成功概率”，并且往往需要依赖额外的“方向来源”（例如 `signal`）来决定到底做多还是做空。
- **新（`long_only` / `short_only`）**：标签只对应一个明确动作。`pred` 就是该方向在 RR 定义下的**成功概率**，阈值更稳定，也更容易保证回测/实盘一致。

**拆分的好处**：
- **目标更干净**：`predict_proba` 直接代表“该方向交易成功的概率”。
- **执行更简单**：方向来自策略本身（不需要 `use_signal_direction`，也不需要 `signal` 列）。
- **控制更细**：多空可以分别设阈值/风控。
- **一致性更强**：离线回测与生产推理语义一致。

#### SR Reversal 的“标签语义 ↔ 执行语义”约定（非常重要）

SR Reversal（`sr_reversal_long/short`）的核心不是“预测下一根 K 涨跌”，而是：
**预测一次入场后，在 R/R 规则下能否先到 TP（如 +2R）再到 SL（如 -1R）**。
因此，我们在代码与配置里刻意做了下面这些约定，保证训练/回测/实盘语义一致：

- **`pred` 的含义**：`pred = P(该方向交易在 max_holding_bars 内按 R/R 规则成功)`  
  它是“达到 RR 目标的成功概率”，不是“退出概率”。
- **入场（entry）**：用 `entry_threshold` 做概率门控（例如 `pred >= 0.35` 才允许开仓）。  
  建议默认用 `entry_mode: cross`，避免 `pred >= threshold` 在连续多根 bar 为 True 时只开一单或造成信号黏连。
- **出场（exit）**：默认只由 **R/R 执行规则** 产生（TP / SL / 超时 / 期末强平），而不是由 `pred` 触发。  
  这点很关键：因为 `pred` 与“何时退出”在该标签定义下没有直接关系。
- **可选：概率阈值退出（仅用于其他策略/实验）**：如果你明确想把 `pred` 当作“趋势/风险状态”，可以显式启用 `exit_mode: threshold`。  
  但对 SR Reversal 的 RR 标签语义，默认应保持 `exit_mode: none`。

#### 负向因子为什么要“取反”（统一方向）

在因子库长期维护中，我们约定：**特征值越大越偏向看涨（正类）**。  
对于在 `factor-eval` 里识别出的“强负因子（Strong Negative Factors）”，推荐在入模前取反（乘以 -1），以获得：
- 更一致的解释与组合（尤其是后续做人为加权/因子打分时）
- 更少的“方向记忆成本”（不用记哪些因子是反的）

实现方式（单文件、配置驱动）：
- `mlbot analyze factor-eval` 导出的 `features_suggested.yaml` 会包含 `feature_pipeline.invert_features`
- 训练入口会读取该列表并在训练/推理时对这些列乘以 -1（保证一致性）

我们仍保留一个轻量的**保险丝**（可选），防止 OOD/噪声 regime 下过度交易：
`dist_to_nearest_sr / ATR > K  =>  不交易`。

#### 研究 → 实盘 Playbook（时间周期、数据长度、执行“性格”）

这一节的目标是：让你在不同策略之间做 **稳定、可对比** 的研究，然后顺滑推进到 **rolling 训练** 与实盘。

##### 1) 每个策略更推荐的 timeframe

- **SR Reversal（支撑阻力反转/均值回归）**：优先 **4H**（结构更干净、换手更低）。1H 也可以，但噪声更大，通常需要更强过滤器。
- **SR Breakout（SR 突破）**：**1H–4H**。4H 更稳，1H 样本更多但假突破更多。
- **Compression Breakout（压缩区突破）**：**1H–4H**。1H 更容易有形态，但确认要更严格。
- **Trend Following（趋势跟随）**：**4H–1D**。低频更容易在成本后保持稳健。

##### 2) 每个策略需要多长数据？（经验法则）

主要受两点约束：
- **市场状态**：需要覆盖多个 regime（趋势/震荡、高波动/低波动）。
- **交易样本**：需要足够多“完成的交易”才能让指标有意义。

以加密市场 **4H** 为例，建议：

| 策略 | 最少建议历史 | 更好 | 原因 |
|---|---:|---:|---|
| `sr_reversal_long/short` | 12–18 个月 | 24–36 个月 | 反转强烈依赖 regime，需要足够多“干净/失败”的 SR 触发样本 |
| `sr_breakout` | 18–24 个月 | 36+ 个月 | 突破在不同波动周期下风格变化大 |
| `compression_breakout` | 18–24 个月 | 36+ 个月 | 需要足够多“压缩→扩张”事件覆盖多种环境 |
| `trend_following` | 24–36 个月 | 4–6 年 | 趋势需要长历史覆盖长趋势年与震荡年 |

**切换 timeframe 时，要把“按 bars 的参数”换算成“按时间等价”。**
例如 4H → 1H，很多以 bars 表示的参数需要约 ×4 才能代表同样的时间长度：
- `rr.max_holding_bars`、标签持有期、rolling 窗口 bars 等。

##### 3) 多标的训练（推荐，但要可控）

你可以同时扩展时间与标的，但要保持可解释性：
- **先从 3–8 个高流动性标的开始**（大盘币优先）。
- **按策略配置训练**，评估时要看 **分标的表现**（不要只看汇总均值）。
- **成本要按标的真实设定**（费率/滑点），高频方案尤其容易“成本前好看”。
- 更推荐用 **rolling 训练** 做月度稳定性验证。

##### 4) 策略“性格”：是否加仓 / 多仓位

不同策略不应共享同一套仓位管理规则。

- **SR Reversal（`sr_reversal_*`）**
  - **一般不加仓**（反转边际脆弱，加仓更容易放大回撤）。
  - **通常每个标的同一时刻最多 1 笔**（方向固定 long-only/short-only）。
  - 目标是“更少、更高质量”的交易，避免过度交易。

- **趋势跟随（Trend Following）**
  - **经常适合加仓**，但必须有严格规则（例如走出 +1R 且信号仍强才允许加一次）。
  - **通常不使用固定 TP**，更偏向跟踪止损/趋势失效退出。

- **SR Breakout**
  - **可选加仓**：只在突破确认后加一次（例如站稳/回踩确认），不建议在第一根尖刺上加。

- **Compression Breakout**
  - **建议先不加仓**：先把 baseline 做稳；如果要加仓，先做“仅一次加仓 + 严格确认”，再重新验证稳定性。

##### 5) 止损止盈：研究阶段先稳定模板

为了让消融/参数对比有意义，执行层先稳定下来：
- **止损**：倾向 **ATR 止损**（跨波动 regime 更一致）。
- **止盈**：
  - **反转/突破类**：RR 风格的固定/部分止盈（例如 +2R）通常合理。
  - **趋势类**：往往不设硬 TP，使用 trailing / 趋势失效退出。
- **持有期**：`max_holding_bars` 最好与 **标签定义** 一致。
  - 如果你改了标签持有期（例如 50→20 bars），特征选择/最佳滞后可能会变化。

##### 6) 研究阶段哪些变量必须固定（保证可复现）

做特征/模型对比时，不要同时改太多变量。建议固定：
- **timeframe**
- **标签定义**（RR 参数、持有期）
- **回测执行语义**（RR 出场 vs 概率出场、成本、滑点）
- **目标交易频率**（例如“4H SR reversal ~20 笔/年”，然后调阈值去命中它）
- **标的集合**与评估窗口
- **rolling 协议**（训练月数、测试月、步长）

一旦 baseline 稳定，再按顺序推进：
1) `factor-eval`（筛特征）
2) `strategy-feature-compare`（消融）
3) `model-comparison`（验证 ML 优于规则）
4) `train rolling`（生产训练）

##### 7) K 线内部开仓 vs 等 K 线收盘开仓

默认建议：**研究与实盘都用“收盘/下一根 K 开仓”**。
原因是更容易保证一致性与可复现，也更不容易踩到“回测假设过于乐观/潜在穿越”的坑。

什么时候 K 线内部开仓有意义：
- **突破/动量**（SR breakout、部分 compression breakout）有时“早入场”会显著影响 R 倍数。
- **更短周期**（例如 1H 及以下）并且你能更真实地建模执行。

如何让 intrabar 更安全（避免回测过于乐观）：
- 最好使用 **更低级别的执行数据**（如 1m）或更严格的“bar 内执行模型”，不要假设能拿到 K 线内最优价格。
- 用保守假设：**next-tick/next-bar 成交**、真实滑点、明确限价/市价规则。
- 保持与标签一致：如果标签假设 \(t+1\) 进场，不要在回测里偷偷改成“当根 K 内进场”而不重新验证。

按策略的建议：
- **SR Reversal**：通常 **等收盘确认** 更稳（intrabar 更容易被噪声来回打脸）。
- **SR Breakout**：在确认逻辑与执行假设足够严格时，可以尝试 intrabar；否则收盘更安全。
- **Compression Breakout**：先用收盘把 baseline 做稳，再考虑 intrabar。
- **趋势类**：多数情况下收盘/下一根 K 足够，不一定需要 intrabar。

#### 步骤 0: 验证特征正确性（推荐）

在开始特征评估之前，运行测试以验证特征计算正确且不使用未来数据：

**快速测试**（仅关键特征）：
```bash
make test-key-features-all
```

**综合测试**（所有特征）：
```bash
make test-all-features-comprehensive
```

**这些测试验证的内容**：
- ✅ 无未来数据泄露（时刻 t 的特征仅使用 ≤ t 的数据）
- ✅ 多资产归一化（特征在不同资产间可比较）
- ✅ 流式与批量一致性（生产推理与训练匹配）
- ✅ 无全局归一化（防止前瞻偏差）

**注意**：在进行特征评估之前，这些测试应该通过。如果测试失败，在训练模型之前修复问题。

**📖 更多详细信息请参见 [测试运行说明](docs/测试运行说明.md)。**

#### 步骤 1: 特征分析

**1.1 评估单个因子**：

**基本用法**：
```bash
mlbot analyze factor-eval \
  --strategy-config config/strategies/sr_reversal_long/features_all.yaml \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 \
  --end-date 2025-10-31 \
```

**评估特定因子**：
```bash
mlbot analyze factor-eval \
  --strategy-config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --factors atr sqs_hal_high \
  --timeframe 240T
```

**高级选项**（去除相关特征，按最佳滞后过滤）：
```bash
mlbot analyze factor-eval \
  --strategy-config config/strategies/sr_reversal_long/features_all.yaml \
  --export-yaml config/strategies/sr_reversal_long/features_suggested.yaml \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2025-01-01 \
  --end-date 2025-10-31 \
  --remove-correlated \
  --correlation-threshold 0.9 \
  --target-lag 20 \
  --lag-tolerance 5 \
  --filter-by-best-lag \
  --open-browser
```

#### 步骤 2: 特征消融研究（必需）

比较不同的特征配置以验证特征选择：

```bash
mlbot analyze strategy-feature-compare \
  --strategy-config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 \
  --end-date 2025-10-31 \
  --test-size 0.5 \
  --feature-overrides "original=features_all.yaml selected=features_suggested.yaml"


mlbot analyze strategy-feature-compare --strategy-config config/strategies/sr_reversal_long --symbol BTCUSDT --timeframe 240T --start-date 2024-01-01 --end-date 2025-10-31 --feature-overrides "all=features_all.yaml suggested_noticks=features_suggested_noticks.yaml suggested=features_suggested.yaml" --test-size 0.5 --output-dir results/strategy_compare_4variants --rolling-max-windows 0
```

**使用滚动窗口**（更稳健）：
```bash
mlbot analyze strategy-feature-compare \
  --strategy-config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 \
  --end-date 2025-10-31 \
  --feature-overrides "original=features_all.yaml selected=features_suggested.yaml" \
  --run-rolling \
  --rolling-train-bars 5000 \
  --rolling-test-bars 1000 \
  --rolling-max-windows 10 \
   --test-size 0.5 
```

**此步骤验证的内容**：
- 选定的特征比所有特征表现更好
- 特征选择提高模型泛化能力
- 特征消融显示出有意义的差异

**注意**：在进行滚动训练之前，此步骤是**必需的**。

#### 步骤 3: 模型对比（必需）

验证 ML 模型优于基于规则的策略，并比较不同策略配置的差异：

**推荐先读实验协议（强烈建议）**：
- `docs/strategies/SR_REVERSAL_EXPERIMENT_PROTOCOL.md`（为什么要做可复现 + 多 seed 统计；如何从 MVP 逐步回填特征，再回到 SR/权重/ticks 主线）
  - ✅ 最新主线配置：`config/strategies/sr_reversal_rr_reg_long_mainline`（回归：SR-filter=1.5 + weights(lw=0.05, boost=1.5)）

**基本用法（单个策略配置）**：
```bash
mlbot diagnose model-comparison \
  --strategy-config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 \
  --end-date 2025-10-31
```

**多策略配置对比（推荐）**：
比较不同策略配置（标签、回测、止损止盈、特征差异）：

```bash
mlbot diagnose model-comparison \
  --strategy-config sr_reversal_long,sr_reversal_long_vol,sr_reversal_rr_reg_long \
  --rule-based-entry src.time_series_model.diagnostics.sr_reversal_model_comparison.evaluate_rule_based \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2023-01-01 \
  --end-date 2025-10-31
```

**参数说明**：
- `--strategy-config`: 逗号分隔的策略配置列表（支持相对路径，如 `sr_reversal_long`，或绝对路径）
- `--rule-based-entry`: （可选）指定 rule-based 策略的代码入口点，用于生成规则基线对比

**此步骤对比的内容**：
- 基于规则的基线（纯规则策略，如果提供了 `--rule-based-entry`）
- ML 模型（XGBoost/LightGBM）
- ML + 波动率模型（如果策略配置中启用了波动率模型）

**对比报告包含**：
- 性能指标：交易数、胜率、保本率、Total R、Sharpe 比率
- 配置差异：标签生成器、任务类型、止损止盈参数、特征数量
- 多策略横向对比表格

**与 `strategy-feature-compare` 的区别**：
- `model-comparison`: 需要复制配置（比较不同策略配置，如不同的标签、回测、止损止盈设置）
- `strategy-feature-compare`: 不需要复制配置（同一目录不同特征配置，用于特征消融研究）

**此步骤验证的内容**：
- ML 模型显著优于规则
- ML 模型提供稳定的收益
- ML 模型具有合理的交易频率
- 不同策略配置的性能差异

**注意**：在进行滚动训练之前，此步骤是**必需的**。

#### 步骤 4: 策略训练（可选 - 用于调试）

**⚠️ 可选**：单次训练仅用于调试或快速配置测试。对于生产环境，在完成步骤 2 和 3 后，直接进行步骤 5（滚动训练）。

**4.1 快速验证**（单次训练）：

**SR Reversal 仅做多**：
```bash
mlbot train sr-reversal-long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2025-01-01 \
  --end-date 2025-10-31 \
```

**SR Reversal 仅做空**：
```bash
mlbot train sr-reversal-short \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2025-01-01 \
  --end-date 2025-10-31 \
```

#### 步骤 5: 生产训练（滚动窗口 - 推荐）
```bash
# 扩展窗口训练：每个测试月份使用所有之前的月份
mlbot train rolling \
  --config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --initial-train-months 6 \
  --min-train-months 3

mlbot train rolling \
  --config config/strategies/sr_reversal_short \
  --symbol BTCUSDT \
  --timeframe 240T \
  --initial-train-months 6 \
  --min-train-months 3
```

**指定日期范围**：
```bash
mlbot train rolling \
  --config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start 2024-01-01 \
  --end 2025-10-31 \
  --initial-train-months 6 \
  --min-train-months 3
```

**仅更新（增量）**：
```bash
mlbot train rolling \
  --config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --update-only
```

**输出**：
- `results/rolling/{strategy}/{month}/model.pkl` - 每个月的模型
- `results/rolling/{strategy}/monthly_results.json` - 聚合结果

**注意**：只有在完成步骤 2（特征消融研究）和步骤 3（模型对比）后，才进行滚动训练，以确保特征和模型架构已验证。

**使用选定特征**（如果您有特征选择结果）：
```bash
# 如果您有来自 factor-eval 的 features_suggested.yaml，请更新您的策略配置以使用它
# 编辑策略的 features.yaml 以使用选定的特征，然后：
mlbot train rolling \
  --config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --initial-train-months 6
```

**输出**：
- `results/auto_rolling_*/monthly_results.csv` - 所有月份的详细结果
- `results/auto_rolling_*/summary.json` - 摘要信息
- `results/auto_rolling_*/monthly_rolling_report.html` - HTML 报告
- `results/auto_rolling_*/model_YYYY-MM.txt` - 每个月的模型

#### 在 Dev Container（Cursor / VS Code）中查看 HTML 报告

如果你在 Dev Container 里运行，`--open-browser` 可能无法自动打开报告。推荐用下面方式：

**方式 A：本地静态服务器 + 端口转发**（最稳定）

```bash
# 示例：把 results 目录作为静态站点暴露出来
mlbot serve-results
# 或（手动）
# python3 -m http.server 8008 --directory results
```

- 在 Cursor/VS Code 打开 **Ports** 面板，转发/打开端口 `8008`
- 然后在本机浏览器打开对应报告，例如：
  - `results/auto_rolling_*/monthly_rolling_report.html`
  - `results/strategy_compare/strategy_feature_compare_report.html`
  - `results/rule_optimization/optimization_report.html`

如果端口 `8008` 已被占用，可以在容器内强制结束占用端口的进程，然后重启服务：

```bash
mlbot serve-results --force
```

#### 步骤 6: 定期更新（每周/每月）

从最后训练的月份进行增量更新：

```bash
# 仅更新新月份（从最后位置开始）- 使用 --update-only 标志
mlbot train rolling \
  --config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --update-only
```

这将自动检测最后训练的月份并从那里继续。

### 完整工作流管道

从特征评估到滚动训练的完整工作流，按顺序执行命令：

```bash
# 步骤 0: 验证特征正确性（推荐）
make test-key-features-all

# 步骤 1: 特征评估和选择
mlbot analyze factor-eval \
  --strategy-config config/strategies/sr_reversal_long/features_all.yaml \
  --symbol BTCUSDT \
  --timeframe 15T \
  --start-date 2025-01-01 \
  --end-date 2025-04-30 \
  --remove-correlated \
  --target-lag 5

# 步骤 2: 特征消融研究（验证特征选择）
mlbot analyze strategy-feature-compare \
  --strategy-config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 \
  --end-date 2025-10-31 \
  --feature-overrides "original=features_all.yaml selected=features_suggested.yaml"

# 步骤 3: 模型对比（验证 ML 优于规则）
# 单个策略配置
mlbot diagnose model-comparison \
  --strategy-config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 15T \
  --start-date 2025-01-01 \
  --end-date 2025-04-30

# 多策略配置对比（推荐）
mlbot diagnose model-comparison \
  --strategy-config sr_reversal_long,sr_reversal_long_vol,sr_reversal_rr_reg_long \
  --rule-based-entry src.time_series_model.diagnostics.sr_reversal_model_comparison.evaluate_rule_based \
  --symbol BTCUSDT \
  --timeframe 15T \
  --start-date 2025-01-01 \
  --end-date 2025-04-30

# 步骤 5: 滚动窗口训练（主要生产工作流）
mlbot train rolling \
  --config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 15T \
  --start 2025-01-01 \
  --end 2025-04-30 \
  --initial-train-months 3 \
  --min-train-months 3
```

> **提示**  
> - 使用 `--docker` 标志（默认）以在可用时启用 GPU 训练  
> - 训练后查看 `results/rolling_*/summary.json` 中的 `monthly_results`  
> - 如果结果中出现 `drift detected`，考虑重新评估特征或调整参数  


## 数据管道

在训练之前，确保您有数据：

```bash
# 下载 Binance 月度聚合交易数据
mlbot data download \
  --symbols BTCUSDT,ETHUSDT \
  --start-year 2021 \
  --start-month 1

# 将 ZIP 转换为 Parquet（5 分钟 OHLC + 订单流）
mlbot data convert

# 或一次性运行两者（完整管道）
mlbot data pipeline \
  --symbols BTCUSDT,ETHUSDT
```

**更多数据管道示例**：

```bash
# 下载特定日期范围
mlbot data download \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT \
  --start-year 2024 \
  --start-month 1 \
  --end-year 2025 \
  --end-month 10

# 转换并清理（转换后删除 ZIP 文件）
mlbot data convert --cleanup
```

## 核心原则

**所有生产训练都应使用降维特征**（Top-K + Autoencoder），而不是原始的 482 个特征。

### 为什么？

1. **更好的性能**：降维特征通常表现更好（如研究所示）
2. **更快的训练**：更少的特征 = 更快的训练
3. **减少过拟合**：降低过拟合风险
4. **一致性**：与研究阶段使用相同的特征集

## 命令对比

| 命令                     | 用途                           | 何时使用                               |
| --------------------------- | --------------------------------- | ----------------------------------------- |
| `mlbot analyze factor-eval` | 因子评估与选择     | **推荐**：主要特征选择方法 |
| `mlbot train sr-reversal-long/short` | 训练单个模型（方向固定） | **可选**：仅用于调试/快速验证 |
| `mlbot train rolling`       | 滚动窗口训练           | **推荐**：主要生产工作流 |

### 关键点

- **工作流顺序**：始终按步骤 0 → 1 → 2 → 3 → 5 的顺序进行
  - 步骤 0: 验证特征正确性（推荐）
  - 步骤 1: 特征评估 (`factor-eval`)
  - 步骤 2: 特征消融研究 (`strategy-feature-compare`) - **必需**
  - 步骤 3: 模型对比 (`model-comparison`) - **必需**
  - 步骤 5: 滚动训练 (`rolling`) - **仅在验证后**

- `mlbot train sr-reversal-long/short`: 为单个时间段训练**一个**模型（方向固定）
  - **不推荐**用于生产评估
  - 仅用于调试或快速配置测试

- `mlbot train rolling`: 以滚动/扩展窗口方式训练**多个**模型（每月一个）
  - **必需**用于生产部署
  - 通过扩展窗口提供更好的评估
  - 只有在验证特征（步骤 2）和模型性能（步骤 3）后使用
  - 步骤 4（单次训练）是可选的，仅用于调试


## 工作流摘要

### 最小工作流（5 步，推荐）

```bash
# 步骤 0: 验证特征正确性（开始前推荐）
make test-key-features-all

# 步骤 1: 特征评估和选择
mlbot analyze factor-eval \
  --strategy-config config/strategies/sr_reversal_long/features_all.yaml \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 \
  --end-date 2025-10-31 \
  --remove-correlated \
  --filter-by-best-lag

# 步骤 2: 特征消融研究（验证特征选择）
mlbot analyze strategy-feature-compare \
  --strategy-config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 \
  --end-date 2025-10-31 \
  --output-dir results/strategy_compare \
  --feature-overrides "original=features_all.yaml selected=features_suggested.yaml"

# 步骤 3: 模型对比（验证 ML 优于规则）
# 单个策略配置
mlbot diagnose model-comparison \
  --strategy-config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 \
  --end-date 2025-10-31

# 多策略配置对比（推荐）
mlbot diagnose model-comparison \
  --strategy-config sr_reversal_long,sr_reversal_long_vol,sr_reversal_rr_reg_long \
  --rule-based-entry src.time_series_model.diagnostics.sr_reversal_model_comparison.evaluate_rule_based \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 \
  --end-date 2025-10-31

# 步骤 4: 滚动训练（仅在验证后）
# 注意：如果您有来自 factor-eval 的 features_suggested.yaml，请更新您的策略配置以使用它
mlbot train rolling \
  --config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start 2024-01-01 \
  --end 2025-10-31 \
  --initial-train-months 6 \
  --min-train-months 3
```

### 完整工作流（5 步，包含所有选项）

```bash
# 步骤 0: 验证特征正确性（推荐）
make test-all-features-comprehensive

# 步骤 1: 特征评估（生成包含选定特征的 features_suggested.yaml）
mlbot analyze factor-eval \
  --strategy-config config/strategies/sr_reversal_long/features_all.yaml \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 \
  --end-date 2025-10-31 \
  --remove-correlated \
  --target-lag 20 \
  --filter-by-best-lag

# 步骤 2: 特征消融研究（比较原始特征与选定特征）
mlbot analyze strategy-feature-compare \
  --strategy-config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 \
  --end-date 2025-10-31 \
  --feature-overrides "original=features_all.yaml selected=features_suggested.yaml" \
  --run-rolling \
  --rolling-train-bars 5000 \
  --rolling-test-bars 1000 \
  --rolling-max-windows 10

# 步骤 3: 模型对比（验证 ML 优于规则）
# 单个策略配置
mlbot diagnose model-comparison \
  --strategy-config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 \
  --end-date 2025-10-31

# 多策略配置对比（推荐）
mlbot diagnose model-comparison \
  --strategy-config sr_reversal_long,sr_reversal_long_vol,sr_reversal_rr_reg_long \
  --rule-based-entry src.time_series_model.diagnostics.sr_reversal_model_comparison.evaluate_rule_based \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2023-01-01 \
  --end-date 2025-10-31

# 步骤 4: 滚动训练（主要生产工作流 - 仅在验证后）
mlbot train rolling \
  --config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start 2024-01-01 \
  --end 2025-10-31 \
  --initial-train-months 6 \
  --min-train-months 3
```

### 高级工作流示例

**诊断和优化工作流**：

```bash
# 1. 规则基线（测试纯基于规则的策略）
mlbot diagnose rule-baseline \
  --strategy-config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2024-01-01 \
  --end-date 2025-10-31

# 2. 规则优化（找到最优参数）
mlbot optimize rule \
  --strategy-config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --search-type random \
  --n-trials 100

# 3. 生成规则平台图表
mlbot optimize rule-plateau-charts \
  --results-csv results/rule_optimization/optimization_results.csv \
  --report-html results/rule_optimization/optimization_report.html

# 4. ML 参数扫描
mlbot optimize ml-param-sweep \
  --strategy-config config/strategies/sr_reversal_long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --start-date 2025-01-01 \
  --end-date 2025-07-31

# 5. 生成 ML 平台图表
mlbot optimize ml-plateau-charts \
  --timeframe 240T
```

**横截面分析工作流**：

```bash
# 1. 构建横截面面板
mlbot cross-section build-panel \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT \
  --start-date 2024-01-01 \
  --end-date 2025-10-31

# 2. 生成 Fama-MacBeth 报告
mlbot cross-section report \
  --panel-path data/cross_sectional_panels/panel.parquet \
  --output-dir results/cross_sectional

# 3. 训练横截面模型
mlbot cross-section train \
  --panel-path data/cross_sectional_panels/panel.parquet

# 4. 自动选择因子
mlbot cross-section select \
  --panel-path data/cross_sectional_panels/panel.parquet \
  --output-path results/cross_sectional/selected_factors.json

# 5. SHAP 分析
mlbot cross-section shap \
  --model-path results/cross_sectional/model.pkl \
  --panel-path data/cross_sectional_panels/panel.parquet

# 6. SHAP 漂移监控
mlbot cross-section shap-drift \
  --model-path results/cross_sectional/model.pkl \
  --panel-path data/cross_sectional_panels/panel.parquet
```

## 文档

- **`docs/workflow_research_to_production.md`** - 完整工作流文档
- **`docs/simplified_workflow.md`** - 简化工作流指南
- **`docs/时序模型/架构：NN多头路径原语（Path Primitives）+Router解耦升级.md`** - NN 多头底座（路径原语）+ Router 解耦升级方案（生产级）

## 快速跑通：NN 多头底座 → Rule(3-action) → RL(e2e)

下面是一套 **最小可跑** 的端到端链路（支持多 symbol）：

- **目标**：训练/推理 NN 多头（path primitives）→ 生成 `mode`（NO_TRADE/MEAN/TREND）→ 组装 RL/BC logs（含 `ret_mean/ret_trend`）→ 一键跑 shadow/counterfactual/fsm。
- **核心产物**：
  - `preds_*.parquet`：包含 `pred_dir_prob/pred_mfe_atr/pred_mae_atr/pred_t_to_mfe`（回归头通常是 log1p 空间）
  - `mode_3action.parquet`：包含 `mode`（NO_TRADE/MEAN/TREND）
  - `logs_3action.parquet`：包含 `symbol,timestamp,mode,head_*,drawdown,ret_mean,ret_trend`
  - `results/rl/e2e/*`：shadow report / counterfactual report / fsm decision

### 命令解释（为什么叫 `counterfactual-eval-3action`？各命令在链路中是什么角色？）

这里的命名是“把研究假设写进命令名”，避免误用：

- **`3action`**：表示 Router 的动作空间是 3 个离散动作：`NO_TRADE / MEAN / TREND`（这是 Router 层的“结构原语动作”，不是具体的开平仓微观执行细节）。
- **`counterfactual`（反事实）**：表示用**同一条市场路径**、同一套 `ret_mean/ret_trend`（以及相同成本/约束）去对比两套策略/政策（Rule vs BC/RL）。  
  直觉：我们不让新策略真的去“改变市场/改变成交”，而是问一个反事实问题：  
  > “在同样的行情和同样的执行回报假设下，如果当时用的是 BC/RL 的 mode，会发生什么？”

下面按命令解释“输入/输出/用途”：

#### 1) `mlbot nnmultihead train`

- **用途**：训练 NN 多头路径原语模型（输出 heads，例如 `dir/mfe/mae/ttm`）。
- **输入**：raw 数据 + `config/nnmultihead/...`（特征与训练配置）。
- **输出**：`model.pt` + `report.html` + `metrics.json`（训练与评估产物）。

#### 2) `mlbot nnmultihead predict`

- **用途**：用训练好的 NN 多头模型推理，产出每个 symbol 的预测 heads。
- **输入**：`model.pt` + raw 数据 + nnmultihead config。
- **输出**：单 symbol 时输出一个 parquet；多 symbol 时输出一个目录 `preds_multi/`，其中包含 `preds_<SYMBOL>.parquet`。

#### 3) `mlbot rule mode-3action`

- **用途**：纯规则 Router（只看 heads）把预测映射成 `mode ∈ {NO_TRADE, MEAN, TREND}`，作为“可解释 baseline”与 BC 的监督信号来源。
- **输入**：`preds_*.parquet`（或 preds 目录）+（可选）`model.pt` 用于推断 preds 是否在 log1p 空间。
- **输出**：`mode_3action.parquet`（含 `symbol,timestamp,mode`）。

#### 4) `mlbot rl build-logs-3action`

- **用途**：组装 RL/BC 训练日志（logs），把 `heads + mode + ret_mean/ret_trend` 合到同一张表里。
- **输入**：
  - `preds`：NN heads 预测
  - `mode`：rule router 输出的 mode（监督 label）
  - raw 数据（用于把 close 等转换成 mode 对应的“下一步执行回报”序列）
- **输出**：`logs_3action.parquet`（要求至少包含 `symbol,timestamp,mode,head_*,drawdown,ret_mean,ret_trend`）。

补充：`ret_mean/ret_trend` 的来源与执行口径（`--returns-source`）：

- **`momentum_proxy`**：纯价格动量近似（最弱，但无依赖，用于兜底）
- **`rr_execution`**：不依赖 vectorbt 的 RR/ATR 执行模拟器（更贴近“止损/止盈/最长持仓”等执行语义）
- **`vectorbt_execution`**：用 vectorbt + RR exits 生成 portfolio step-returns（用于与回测口径对齐）

补充：Execution 专门化（同一 action 在不同市场用不同执行参数）：

- `--symbol-profiles-json`：为每个 symbol 指定 `market_profile`（会写入 logs 的 `market_profile` 列，便于审计）
- `--rr-profile-overrides-json` / `--vbt-profile-overrides-json`：针对不同 profile 覆盖 RR/fee/slippage/holding 等参数

示例（按 symbol → profile 给 RR 参数做差异化）：

```bash
mlbot rl build-logs-3action \
  --preds results/nnmultihead/preds_multi \
  --mode results/rule/mode_3action.parquet \
  --model results/nnmultihead/.../model.pt \
  --data-path data/parquet_data \
  --timeframe 240T \
  --returns-source rr_execution \
  --symbol-profiles-json '{"BTCUSDT":"btc","DOGEUSDT":"meme"}' \
  --rr-profile-overrides-json '{"meme":{"max_holding_bars":12,"take_profit_r":2.5},"btc":{"max_holding_bars":24,"take_profit_r":2.0}}' \
  --output results/rl/logs_3action.parquet \
  --no-docker
```

#### 5) `mlbot rl shadow-eval-3action`

- **用途**：训练一个 BC(3-action) policy，并在 test 段做“影子评估”（shadow）：主要看它能否在不影响真实交易的情况下稳定复现 rule mode（例如 accuracy/混淆矩阵等）。
- **输入**：`logs_3action.parquet`（不要求必须包含 ret_*，因为它更多是“行为一致性评估”）。
- **输出**：`shadow_report.html` + `metrics.json`。

#### 6) `mlbot rl counterfactual-eval-3action`

- **用途**：反事实 A/B：在 **同一份 logs** 上，把 Rule policy 的 mode 序列与 BC policy 的 mode 序列分别送进同一个 Router-level 模拟器（用 `ret_mean/ret_trend` 作为执行层回报代理），得到两条 equity 曲线并对比（Sharpe/Sortino/年化/回撤等）。
- **输入**：`logs_3action.parquet`（必须包含 `ret_mean/ret_trend`）。
- **输出**：`report.html` + `metrics.json` + `per_symbol.csv`。

#### 7) `mlbot rl fsm-decide`

- **用途**：根据 `counterfactual-eval-3action` 输出的 `metrics.json`，用 FSM gate 决定是否从 RULE → RL_CANDIDATE → RL_ACTIVE（或进入 RL_SUSPENDED），用于上线/回退的工程化阈值控制。
- **输入**：`metrics.json`。
- **输出**：`fsm_decision.json`（可落盘）。

#### 8) `mlbot rl run-e2e-3action`

- **用途**：一键串联：`shadow-eval-3action` → `counterfactual-eval-3action` → `fsm-decide`。
- **输入**：`logs_3action.parquet`。
- **输出**：`{out}/shadow/*`、`{out}/counterfactual/*`、`{out}/fsm_decision.json`。

#### 9) `mlbot rl exec control-check` / `mlbot rl exec chaos-test`（Execution 控制/安全壳）

这两个命令属于“执行控制层”的工程化工具：它们不产 alpha，只做 **invariants + kill-switch + 压测**，避免上线时因为数据/成本/切换异常导致系统失控。

**`mlbot rl exec control-check`**

- **用途**：对 logs 做执行层一致性检查（NaN/极端 returns、turnover/cost、DD），并输出 `kill_switch`（建议是否强制 NO_TRADE）。
- **输入**：`logs_3action.parquet`（至少含 `symbol,timestamp,mode,ret_mean,ret_trend`）
- **输出**：`report.html` + `metrics.json` + `per_symbol.csv`

**`mlbot rl exec chaos-test`**

- **用途**：对同一份 logs 做 baseline vs chaos 对比（注入 NaN / 放大 returns / 提高成本等），验证 kill-switch 是否按预期触发。
- **输出目录**：`{out}/baseline/*` 与 `{out}/chaos/*`

示例（对 logs 注入 2% NaN 并放大 returns）：  

```bash
mlbot rl exec chaos-test \
  --logs results/rl/logs_3action.parquet \
  --out results/rl/exec_chaos \
  --nan-ratio 0.02 \
  --return-scale 3.0 \
  --no-docker
```

### 0)（推荐）先构建 FeatureStore 宽表库：一次算好特征，后续训练/回测直接读（尤其含 ticks）

FeatureStore 是 **按月分区的“特征宽表”**（不是 `cache/features/*` 的计算缓存）。构建一次后：
- tree 策略训练可优先从 FeatureStore 取特征（缺列再回退到计算+缓存）
- nnmultihead 训练/推理可直接读 FeatureStore（不重复跑 feature pipeline）

示例（对 nnmultihead 的 config 生成 FeatureStore；`--layer AUTO` 会根据 config 内容生成稳定 id）：

```bash
mlbot feature-store build \
  --config config/nnmultihead/path_primitives_4h_80h_min \
  --symbols BTCUSDT,ETHUSDT \
  --timeframe 240T \
  --data-path data/parquet_data \
  --root feature_store \
  --layer AUTO \
  --no-docker
```

示例（对树模型策略 config 生成 FeatureStore）：

```bash
mlbot feature-store build \
  --config config/strategies/sr_reversal_long \
  --symbols BTCUSDT,ETHUSDT \
  --timeframe 240T \
  --data-path data/parquet_data \
  --root feature_store \
  --layer AUTO \
  --no-docker
```

tree 训练优先读 FeatureStore（缺列回退到计算缓存）：

```bash
mlbot train sr-reversal-long \
  --symbol BTCUSDT \
  --timeframe 240T \
  --data-path data/parquet_data \
  --feature-store-dir feature_store \
  --feature-store-layer AUTO \
  --no-docker
```

### 1) 训练 NN 多头（可选：你已经有模型就跳过）

```bash
mlbot nnmultihead train \
  --symbols BTCUSDT,ETHUSDT \
  --timeframe 240T \
  --data-path data/parquet_data \
  --config config/nnmultihead/path_primitives_4h_80h_min \
  --feature-store-root feature_store \
  --feature-store-layer AUTO \
  --epochs 10 \
  --output-dir results/nnmultihead \
  --no-docker
```

### 2) NN 多头推理（多 symbol 输出目录）

```bash
mlbot nnmultihead predict \
  --symbols BTCUSDT,ETHUSDT \
  --timeframe 240T \
  --data-path data/parquet_data \
  --config config/nnmultihead/path_primitives_4h_80h_min \
  --feature-store-root feature_store \
  --feature-store-layer AUTO \
  --model results/nnmultihead/.../model.pt \
  --output results/nnmultihead/preds_multi \
  --no-docker
```

### 3) 纯规则 Router：从 heads 生成 3-action `mode`

```bash
mlbot rule mode-3action \
  --preds results/nnmultihead/preds_multi \
  --model results/nnmultihead/.../model.pt \
  --output results/rule/mode_3action.parquet \
  --no-docker
```

### 4) 组装 RL/BC logs（把真实 close 转成 ret_mean/ret_trend，并合并 heads + mode）

```bash
mlbot rl build-logs-3action \
  --preds results/nnmultihead/preds_multi \
  --mode results/rule/mode_3action.parquet \
  --model results/nnmultihead/.../model.pt \
  --data-path data/parquet_data \
  --timeframe 240T \
  --output results/rl/logs_3action.parquet \
  --no-docker
```

### 5) 一键跑 RL(e2e)：shadow → counterfactual → fsm

```bash
mlbot rl run-e2e-3action \
  --logs results/rl/logs_3action.parquet \
  --out results/rl/e2e \
  --no-docker
```

## 获取帮助

查看所有可用命令：
```bash
mlbot --help
```

查看特定类别的命令：
```bash
mlbot analyze --help      # 分析和评估命令
mlbot train --help        # 训练命令
mlbot diagnose --help     # 诊断命令
mlbot optimize --help     # 优化命令
mlbot backtest --help     # 回测命令
mlbot cross-section --help # 横截面分析命令
mlbot data --help         # 数据管理命令
mlbot features --help     # 特征管理命令
mlbot dev --help          # 开发命令
```

另请参见：
- [迁移指南](docs/MIGRATION_GUIDE.md) - 从 Makefile 迁移到 mlbot 的完整指南
- [Makefile vs mlbot](docs/MAKEFILE_VS_MLBOT.md) - 命令对比表

## 开发环境

**推荐使用 VS Code Dev Container**：
1. 用 VS Code 打开项目
2. 选择 "Reopen in Container"（自动进入 Dev Container）
3. 在容器内直接运行 `mlbot` 命令（无需通过 Makefile）

**命令行使用**：
- 在 Dev Container 中：直接使用 `mlbot` 命令
- 在本地环境：使用 `mlbot` 命令（需要先 `pip install -e .`）

所有 `mlbot` 命令都支持 `--docker/--no-docker` 选项，可以根据环境自动选择合适的执行方式。

