# bpc 完整上线流程

## 1. 已经用 failure_rr_extreme 在 only_long 标签下 找到failure first

训练 failure_rr_extreme
```bash
mlbot train final --no-docker \
  --config config/strategies/bpc \
  --labels config/strategies/bpc/labels_rr_extreme.yaml \
  --symbol BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT \
  --timeframe 240T \
  --data-path data/parquet_data \
  --start-date 2023-01-01 \
  --end-date 2025-11-30 \
  --holdout-start-date 2024-05-01 \
  --holdout-end-date 2025-11-30 \
  --seed 42
```
[bpc_20260131_171621_report.html](http://localhost:8008/bpc_20260131_171621_report.html)

## 2. 导出核心到规则（mlbot train final 会自动导出了）
mlbot train export-rules --no-docker --model-dir models/bpc --strategy bpc --max-splits 30   --generate-risk-gate

mlbot train export-rules \
  --model-dir results/train_final_xxx/bpc \
  --strategy bpc \
  --generate-risk-gate

 /home/yin/trading/ml_trading_bot/models/bpc/bpc_tree_rules.md

## 3. 形成 risk_gate.yaml
后面可以直接copy到 config/nnmultihead/execution_archetypes.yaml 里面

## 4. 训练 failure_no_opportunity, 导出核心规则，形成risk_gate.yaml
```bash
mlbot train final --no-docker \
  --config config/strategies/bpc \
  --labels config/strategies/bpc/labels_no_opportunity.yaml \
  --symbol BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT \
  --timeframe 240T \
  --data-path data/parquet_data \
  --start-date 2023-01-01 \
  --end-date 2025-11-30 \
  --holdout-start-date 2024-05-01 \
  --holdout-end-date 2025-11-30 \
  --seed 42
```

## 5. 分层架构里面编写gate，evidence

config/nnmultihead/execution_archetypes.yaml 里面

## 6. 训练找到 evidence 最佳参数

用平坦高原办法

## 7. oos 测试各种灭绝数据

拿到灭绝数据不会毁灭证据

## 8. oss 测试长时间数据

拿到正常sharp数据

## 9. Tiers 迁移到 YAML + 分档优化（已完成 ✅）

4 档参数（强/中/弱/边缘）已从 `tier.py` 硬编码迁移到 `execution.yaml` 的 `tiers:` 段。

**完成内容:**
1. ✅ `execution.yaml` 新增 `tiers:` 段（4 档 evidence_min + 止损参数 + size_multiplier + timeout）
2. ✅ `backtest_execution_layer.py` 新增 `compute_evidence_scores()` 和 `assign_tiers()`
3. ✅ `simulate_rr_execution()` 支持 `use_tier_params=True`，per-entry 执行参数
4. ✅ `--tiers` CLI 标志，输出 Per-Tier Breakdown
5. ✅ evidence_score 由 evidence.yaml 8 个特征加权计算，分位数分箱

**回测结果 (11107 trades):**
| Mode | Sharpe | Mean R | Win |
|------|--------|--------|-----|
| Baseline | 0.308 | 13.20 | 46.6% |
| Tiers | 0.313 | 12.64 | 45.3% |

Per-Tier: 强证据 23.8%，中等 41.6%，弱 21.2%，边缘 11.2%

## 10. Noise Penalty 回测验证（已完成 ✅）

`ExecutionNoisePenalty` 已接入 `backtest_execution_layer.py`，
4 个数学特征从 FeatureStore 自动加载。

**完成内容:**
1. ✅ `--noise-penalty` CLI 标志
2. ✅ 自动从 FeatureStore 加载 wpt/spectrum/hilbert/hurst 4 个特征
3. ✅ 按 symbol 分别计算 noise_penalty（避免跨 symbol 污染）
4. ✅ 调整规则：SL 拉宽 (+50%*penalty)，Trail 拉宽 (+30%*penalty)，Size 缩小 (-70%*penalty)
5. ✅ 同时支持 Tier 模式和全局模式

**回测结果:**
| Mode | Sharpe | Mean R | Win | Avg Size |
|------|--------|--------|-----|----------|
| Baseline | 0.308 | 13.20 | 46.6% | 1.0x |
| Noise only | 0.305 | 13.20 | 46.8% | 0.65x |
| Tiers + Noise | 0.313 | 12.78 | 46.5% | 分档 |

结论：Noise Penalty 单独使用对 Sharpe 影响微小（-0.003），
主要作用是降低噪声环境中的仓位，降低最大回撤。

## 11. 逐K线路径模拟（已完成 ✅）

已实现 bar-by-bar 模拟，替代旧的 `forward_rr` 单值模拟。

**完成内容:**
1. ✅ 新增 `simulate_rr_execution()` 逐 bar 前向模拟函数
2. ✅ 入场方向由 `bpc_breakout_direction` 决定，使用 `entry_direction` 列标记入场信号
3. ✅ 连续 OHLC 体系：logs 有 OHLC 直接用，否则 FeatureStore 回退
4. ✅ Gate 过滤不删除行（保持 OHLC 连续性），而是将非 allow 行的 direction 置 0
5. ✅ 移除 `forward_rr` 引用，避免混淆
6. ✅ Grid Search + Per-Symbol HTML 报告正常工作

## 12. 入场时机过滤器 --entry-filter（已完成 ✅）

`bpc_breakout_direction` 的方向是 Donchian 20-bar 突破方向，用于确定 long/short。
但原来 100% bar 都有方向（forward-fill），每根 bar 都入场。

通过入场时机过滤，只在深回踩底部入场，Sharpe 显著提升。

**架构: Config-driven (`entry_filters.yaml`)**
- 所有入场模式定义在 `config/strategies/bpc/archetypes/entry_filters.yaml`
- 每个模式: id + enabled + conditions + backtest数据 + 语义描述
- 代码动态解析 conditions (feature + operator + value)，不再硬编码
- 未启用的模式也记录在 `disabled_filters:` 段（含拒绝原因）

**9 种已启用模式 (按 Sharpe 排序):**
| # | Filter | Trades | Sharpe | Win% | 特征维度 |
|---|--------|--------|--------|------|----------|
| 1 | deep_pullback_full | 121 | 0.492 | 52.1% | 结构+形态+订单流 |
| 2 | deep_pullback_vol | 429 | 0.424 | 51.7% | 结构+量能 |
| 3 | deep_pullback_momentum | 302 | 0.418 | 51.0% | 结构+动量 |
| 4 | deep_pullback_wick | 241 | 0.409 | 48.5% | 结构+形态 |
| 5 | deep_pullback_cvd | 993 | 0.381 | 52.1% | 结构+订单流 |
| 6 | deep_pullback_liq_void | 342 | 0.376 | 48.2% | 结构+footprint |
| 7 | deep_pullback_wpt | 336 | 0.375 | 51.2% | 结构+小波 |
| 8 | deep_pullback_bb | 585 | 0.370 | 51.8% | 结构+波动率 |
| 9 | deep_pullback | 1762 | 0.328 | 50.6% | 结构 |

**7 种已测试但未启用 (在 disabled_filters 段):**
- deep_pullback_sr (Sharpe 0.278 < baseline), deep_pullback_vpin (0.280),
  confidence_only (0.299), sr_quality_top10 (0.322), pc_transition (0.305),
  deep_pullback_vp (58 trades 太少), deep_pullback_impulse_vol (胜率低)

**组合模式:**
- `--entry-filter deep_pullback_cvd --tiers`: Sharpe=0.395 (+28%)
- 语义确认层应单独优化，各特征独立且可叠加

## 13. 上线bpc