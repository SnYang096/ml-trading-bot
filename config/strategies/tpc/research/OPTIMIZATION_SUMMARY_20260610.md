# TPC 策略优化总结 (2026-06-10)

> 完整记录 TPC 各层优化决策、实验依据与最终结论。

---

## 总览

| 层 | 优化项 | 基线 | 最优 | 提升 | 状态 |
|----|--------|------|------|------|:----:|
| 加仓 | 金字塔 + 阶梯门槛 | 0.5/0.25 二档 | [0.25,0.5,1.0] 三档 | +3 档 | ✅ 已应用 |
| 保本 | Breakeven 触发阈值 | 10 ATR | 6 ATR | 3.4× R | ✅ 已应用 |
| 退出 | 移动止损→结构退出 | Trailing | Regime-Adaptive | 2.4× bull | ✅ 代码就绪 |
| 风控 | 仓位风险 | 1%/笔 | 2%/笔 | 1.7× CAGR | ⏸️ 可选 |
| 宪法 | 最大加仓次数 | 2 | 3 | 适配三档 | ✅ 已应用 |

---

## 1. 加仓层 — 金字塔阶梯加仓

### 实验
- **Grid**: `config_experiments/tpc_add_full_ablation_strategies/grid.yaml`
- **Variants**: E8 (旧), E9 (基线)

### 决策

| 参数 | 旧值 | 新值 | 理由 |
|------|------|------|------|
| `add_size_multipliers` | `[0.5, 0.25]` | `[0.25, 0.5, 1.0]` | 三档金字塔，第三档满仓 |
| `min_current_r_by_add` | `[0.5, 1]` | `[0.5, 1, 1.5]` | 第三档需 1.5R 浮盈 |
| `constitution.max_add_times` | 2 | 3 | 允许三次加仓 |

### 结论
三档金字塔在趋势中能更充分地放大盈利，同时门槛递增确保只在强势趋势中满仓。

---

## 2. 保本层 — Breakeven 触发优化

### 实验
- **Grid**: `config_experiments/tpc_add_full_ablation_strategies/grid.yaml`
- **Variants**: E8 (be 10 ATR), E9 (be 6 ATR)

### 结果

| Variant | Bull Total R | 
|---------|:-----------:|
| E8 (be 10 ATR) | 9.32R |
| **E9 (be 6 ATR)** | **31.55R** |

### 结论
6 ATR (=1.5R) 的保本触发在捕获早期趋势时不会过早锁仓，而 10 ATR 太宽松导致利润回吐。**10→6 ATR 是本次最大单项提升（3.4×）。**

---

## 3. 退出层 — Regime 自适应退出

### 实验
- **Grid**: `config_experiments/tpc_exit_regime_ablation/grid.yaml`
- **Phase 4**: `config_experiments/tpc_exit_regime_ablation/grid_phase4.yaml`
- **E21**: `config_experiments/tpc_regime_adaptive_exit/grid.yaml`

### 对比 (6 币种, Bull 2023-2024)

| Variant | 机制 | Total R | CAGR | 笔数 |
|---------|------|:-------:|:----:|:----:|
| E9 | Trailing stop | 30.56R | 18.95% | 138 |
| E12 | Wide trail | 21.15R | 12.70% | — |
| E13 | Structural only | 63.33R | 31.27% | 92 |
| **E21** | **Regime-Adaptive** | **72.19R** | **39.09%** | **97** |

### 对比 (6 币种, Recent Range→Bear 2025-2026)

| Variant | 机制 | Total R | 
|---------|------|:-------:|
| E9 | Trailing stop | 16.98R |
| E13 | Structural only | -6.66R |
| E21 | Regime-Adaptive | ⏳ 待跑 |

### 机制说明

```
入场时检查 ema_1200_position:
  > 0.10 (牛市) → 禁止 trailing，纯结构退出 (EMA1200)
  ≤ 0.10 (熊市/震荡) → 启用 trailing (3.5R 激活, 6R 跟踪)
```

### 关键发现
1. **Trailing stop 是趋势策略的敌人**：E12 放宽 trailing 反而更差（21R），E13 直接关掉翻倍（63R）
2. **纯结构退出在熊市失效**：E13 在 recent 段 -6.66R
3. **Regime-Adaptive 在牛市优于纯结构**：E21=72R > E13=63R，说明自适应在牛市子区间内仍有 trailing 保护
4. **5x/10x Take-Profit 从未触发**：TPC 持仓周期太短，不适合 TP 目标

### 引擎代码
- `src/time_series_model/live/generic_live_strategy.py`: `ExecutionParamGenerator.generate_params()` 
- 配置路径: `execution.yaml → stop_loss.regime_adaptive_exit`

---

## 4. 风控层 — 仓位风险比例

### 实验
- **Grid**: `config_experiments/tpc_exit_regime_ablation/grid_2pct.yaml`

### 结果 (E13, Bull 2023-2024)

| 风险 | Total R | Final | CAGR |
|------|:-------:|:-----:|:----:|
| 1%/笔 | 63.33R | $15,405 | 31.27% |
| 2%/笔 | 63.68R | $19,831 | 53.90% |

> Total R 相同是因为 R 已做风险归一化。CAGR 从 31%→54% 验证 2% 生效。

### 结论
2% 风险适合牛市，但需要注意最大回撤限制。建议作为可选配置，根据市场环境切换。

---

## 5. 其他优化

| 优化项 | 说明 | 状态 |
|--------|------|:----:|
| Entry filter 禁用 | `tpc_deep_pullback_delta_absorb` 证据不足 | ✅ |
| signal_add bug 修复 | PCM 同 archetype 持仓时错误拒单 | ✅ |
| 死代码清理 | `bpc_follow_signal`, `me_momentum_expand`, `me_atr_step_add` | ✅ |
| Take-Profit 禁用 | TPC 不适合 5x/10x TP | ✅ |

---

## 实验目录索引

| 实验 | 路径 | 说明 |
|------|------|------|
| 加仓全消融 | `config_experiments/tpc_add_full_ablation_strategies/` | E8/E9 加仓+保本 |
| 退出机制消融 | `config_experiments/tpc_exit_regime_ablation/` | E9-E20 退出对比 |
| 退出 Phase 4 | `config_experiments/tpc_exit_regime_ablation/grid_phase4.yaml` | E18-E20 时间/TP |
| 2% 风险验证 | `config_experiments/tpc_exit_regime_ablation/grid_2pct.yaml` | 1% vs 2% |
| Regime-Adaptive | `config_experiments/tpc_regime_adaptive_exit/` | E21 自适应退出 |

---

## 待完成

- [ ] E21 recent_range_to_bear 段回测
- [ ] E21 推到 TPC/BPC/ME/SRB 生产基线
- [ ] HYPE 新 token 接入流程
- [ ] rolling_trend 引擎 + 回测

---

*最后更新: 2026-06-10*
