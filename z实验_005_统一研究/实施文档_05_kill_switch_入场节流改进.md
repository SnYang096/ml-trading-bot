# Kill Switch 入场节流改进

## 背景

2024-11 ME 策略回测中，616 个有效信号被 kill switch 拦截，导致整月错失
SOL/XRP/BTC 等主升浪行情。禁用 kill switch 后回测：150 trades, WR=67%,
total_r=+212.7R, sharpe=0.52。

## 问题根因

ME 动量策略的信号天然聚集——多周期对齐触发时，多 symbol 同时满足条件。
当前 constitution kill switch 参数与 ME slot 结构存在不匹配：

| 参数 | 当前值 | ME 实际 | 矛盾 |
|------|--------|---------|------|
| daily_loss_limit | 4% | 5 slots x 1% = 5% 同日 SL | 必触发 |
| cooldown_minutes | 240 (4h) | 2h 决策周期 = 仅跳 2 根 bar | 几乎无效 |
| max_add_times | 3 | 单 symbol 最多 4 腿 x 1% = 4% | 单币即可触发日限 |

触发链：聚集入场 → 集中止损 → daily 4% 触发 → cooldown 4h 后恢复 →
再亏再触发 → weekly 8% 触发 → 整月禁入。

实测数据（2024-11-01）：单日开 8 笔，11/01-04 共 21 笔试探，18 笔 SL。

## 改进方案

### 1. 新增 max_new_entries_per_day（核心）

在 `constitution.yaml` `per_strategy_limits` 中新增每策略每日新开仓上限。
不含加仓（add_position），仅统计首次开仓 intent。

```yaml
per_strategy_limits:
  me:
    max_new_entries_per_day: 3
```

效果：日内最大新开仓风险 = 3 x 1% = 3% < daily_loss_limit，保证单日不触发。

实现路径：
- constitution_executor.py: 解析新字段
- event_backtest.py: intents 循环中检查日入场计数
- live_pcm.py: select_intent 中检查日入场计数

### 2. 调整 kill switch 参数

```yaml
kill_switch:
  daily_loss_limit: 0.06    # 4% -> 6%
  cooldown_minutes: 720     # 4h -> 12h
```

- 6% = 5 slots x 1% + 1% 缓冲（入场节流后日内最多 3%，不会触发）
- 720min = 12h = 6 根 2h bar（一轮完整冷却）

不动的：weekly 8% / monthly 12% / max_dd 20%。

### 3. funnel 诊断增强

新增 `reject_daily_entry_limit` 计数器，与 `reject_kill_switch` 并列。

## 不改的

- 加仓腿 loss 计入方式：每条腿独立计入 equity 是正确的
- weekly/monthly/max_dd：账户安全网不放松
- no_kill_switch：仅限回测诊断

## 验证

```bash
mlbot pipeline run --all \
  --config config/prod_train_pipeline_2h_slow_me_only.yaml \
  --stage rolling_sim
```

对比 tag `me-momentum-overhaul-v1` 的基线结果。
