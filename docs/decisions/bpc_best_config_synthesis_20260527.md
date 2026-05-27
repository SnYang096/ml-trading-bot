# BPC 各层对照 — 当前最佳配置合成（2026-05-27）

> Regime 三向 grid 跑完后更新 §1；其余层来自已完成的 rd_loop + ABH。

## 推荐生产组合（在 regime grid 完成前）

| 层 | 推荐 | 依据 |
|---|---|---|
| **Regime** | `chop<=0.40` + `\|ema_1200_position\|>=0.10`（生产）；slope **暂不** | grid：ema_only recent +5.89 vs box -1.22；ema_slope recent +6.89 边际 |
| **Entry** | 生产 orderflow；**不用** v2 | [`bpc_entry_v2_experiment_20260527.md`](bpc_entry_v2_experiment_20260527.md)：v2 两段 totR 更差 |
| **Prefilter** | 保留四锚；**不删** `bpc_recent_breakout_strength` | ABH no_breakout ≡ A on recent |
| **Prefilter 微调** | `bpc_volume_compression_pct` 0.9295 → **0.95**（可选 P1） | plateau scan \|z\|=2.33 @0.95 |
| **Gate** | 生产 baseline（vol 全开）；**不**用 B/H | ABH：H recent -5.59R；B recent -4.18 |
| **Direction** | 不动 | 未做变体 |
| **Execution** | 不动 | 未做变体 |

## 1. Regime（`bpc_regime_ema_grid.yaml` — ✅ 见 [`bpc_regime_ema_experiment_20260527.md`](bpc_regime_ema_experiment_20260527.md)）

| 变体 | 说明 |
|---|---|
| box_legacy | chop + box（旧） |
| ema_only | chop + EMA1200（当前生产） |
| ema_slope | chop + EMA + \|slope_10\|>=0.002 |

## 2. Prefilter（rd_loop + ABH）

| 规则 | label 扫描 | event_backtest |
|---|---|---|
| breakout_strength>=0.4 | 保留区 succ **低于** 拒绝区（\|z\|=3.5） | no_breakout **≡ A**（recent 同 trades/totR）→ **保留** |
| pullback<=0.55, recovery>=0.5 | flat | 未单独 ABH |
| volume_compression>=0.9295 | plateau **0.95** | 未 ABH |

## 3. Gate（ABH 20260527，已完成）

| 窗 | A | B vol off | H bull-vol | 胜者 |
|---|---:|---:|---:|---|
| 2024 bull | +16.85 | +17.56 | +16.81 | B 略高（弱显著） |
| 2025–26 recent | **-1.22** | -4.18 | **-5.59** | **A** |

**结论**：维持 vol gates；不引入 TPC 式 H。

## 4. 与 TPC 差异（为何 BPC 不照搬 TPC regime/gate）

- rd_loop：**BPC strong-bull label -5.7pp** vs TPC +6.8pp → 同一 EMA 阈值语义不同。
- 近期 BPC totR 全负、TPC 仍正 → 优先查 **策略 archetype / PCM**，再动 regime。

## 5. 复现

```bash
# 已完成
PYTHONPATH=src:scripts python -m scripts.event_backtest \
  --variant-grid config/experiments/bpc_abh_variant_grid.yaml --quiet-signal-logs

# Regime 三向（进行中/重跑）
PYTHONPATH=src:scripts python -m scripts.event_backtest \
  --variant-grid config/experiments/bpc_regime_ema_grid.yaml --quiet-signal-logs
```
