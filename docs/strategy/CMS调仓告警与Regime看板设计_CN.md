# CMS 调仓告警与 Regime 看板设计

> **状态**：T2a–T2d 已落地（2026-06-12）  
> **读者**：实盘运维、CMS 开发  
> **上位**：[产品路线图_TODO优先级_CN.md](产品路线图_TODO优先级_CN.md)（T2）· [牛市Beta账本调仓与币本位取舍_CN.md](牛市Beta账本调仓与币本位取舍_CN.md)  
> **现状**：`/regime` 页仅有 **静态配置 + 离线 drift 表**（`RegimePage.tsx` → `/api/trend/regime-ops`），**无 live regime、无 NAV 占比、无调仓建议**

---

## 1. 核心判断：你的思路对不对？

**对。** A/B/C 的 regime **语义不同、时钟不同**，不应强行合成一个 `bull/bear` 标签去驱动所有系统。

| 层 | Regime 回答的问题 | 典型时钟 | 当前配置锚点 |
|----|-------------------|----------|--------------|
| **A·Spot** | 是否处于 **周线极端成本区**（深熊 deploy）？ | 周线 EMA200 | `spot_accum_simple/archetypes/prefilter.yaml` |
| **B·Trend** | 2H 结构上是 **牛/熊/中性** 趋势段？ | 120T / ADX+EMA1200 | `tpc/archetypes/regime.yaml` `allowed_regimes` |
| **C·Multi-leg** | 2H 微观是 **chop 网格** 还是 **动量 scalp**？ | 120T router | `chop_grid` / `trend_scalp` `extensions.multileg` |

**调仓告警的正确做法：**

1. **三层 regime 全部展示**（live 当前值 + 阈值 + 人类可读释义）——作 **参考面板**  
2. **另建一层「调仓策略」**（`rebalance_policy`）——把三层信号 + 账户 NAV **映射** 到目标占比带  
3. **不**用 B 的 `bull` 直接代表 A 该加仓、也 **不**用 C 的 chop 代表全局 risk-off  

---

## 2. 现状缺口

```mermaid
flowchart LR
  subgraph now [现在 /regime]
    T[配置表：yaml 路径 / drift 状态]
  end

  subgraph need [目标 Regime Cockpit]
    L1[A 面板：周线成本区]
    L2[B 面板：牛熊中性]
    L3[C 面板：chop / 动量]
    NAV[NAV 占比条]
    ALERT[调仓告警条]
  end

  now -.->|升级| need
```

| 已有 | 缺失 |
|------|------|
| `regime_ops.py` 读 yaml + `results/regime_drift_monitor` | Feature bus **最新 bar** 上的 live 分类 |
| `exchange_balances.py` 三 scope（spot/trend/multi_leg） | **NAV 占比** vs 目标带 |
| `regime_health.py` 离线 parquet `classify()` | CMS 实时展示 + 调仓建议 |
| `regime_watchdog_baseline.json`（TPC bull_share 基线） | 与 live 对比的可视化 |

---

## 3. 页面信息架构（`/regime` 升级为 Regime Cockpit）

### 3.1 布局（自上而下）

```
┌─────────────────────────────────────────────────────────────┐
│ ⚠ 调仓告警条  OK | WATCH | REBALANCE_SUGGEST + 一句话建议    │
├─────────────────────────────────────────────────────────────┤
│ NAV 占比（横向堆叠条）                                        │
│  A·Spot ████ 18%  │  Rolling —  │  B·Trend ████████ 42%  │  C ███ 12%  │
│  目标带（灰影）    10–25%       0–15%*      25–40%         10–20%      │
├──────────────┬──────────────┬──────────────────────────────┤
│  A·Beta 慢层  │  B·Swing α   │  C·Micro α                   │
│  周线成本区   │  牛熊中性     │  chop / 动量 router          │
├──────────────┴──────────────┴──────────────────────────────┤
│ Regime Ops（保留现有表：配置 / drift / 校准时间）              │
└─────────────────────────────────────────────────────────────┘
```
\* `rolling` scope 待 T1 live 后接入；MVP 可显示「未配置」。

### 3.2 三层 Live Regime 卡片（并列展示，不合并）

#### A·Spot — 「极端成本区 / 周线 EMA200」

| 字段 | 来源 | 展示 |
|------|------|------|
| `weekly_ema_200_position` | feature bus 120T | 数值 + 相对 EMA200 的 % |
| **状态** | `prefilter.rules` | `DEEP_BEAR`（<0，允许 deploy）/ `ABOVE_EMA200`（不买） |
| `abc_macro_regime_score`（可选） | 同 bus | 0–5 分 + risk-on 文字（≥4 牛，≤2 熊） |
| 释义 | 固定文案 | 「A 只在深熊吸筹；牛市应 **少 deploy、多持币**」 |

#### B·Trend — 「牛 / 熊 / 中性」

| 字段 | 来源 | 展示 |
|------|------|------|
| **当前 label** | `RegimeConfig.classify()` @ BTC（主） | `bull` / `bear` / `neutral` 徽章 |
| 驱动特征 | 同 bar | `adx_50`, `ema_1200_position` 实际值 vs 阈值 |
| **7d bull_share** | 近 7d bus 窗口 classify | 如 12% vs baseline 0% → 漂移提示 |
| 可选 ETH | 第二行 | 与 BTC 不一致时显示「分化」 |
| 释义 | `regime.yaml` description | bull=结构退出；bear/neutral=trailing 保护 |

#### C·Multi-leg — 「chop 区 vs 动量区」

两个子策略 **都显示**（同一账户 router 可能只跑其一，但监控两侧特征）：

| 子策略 | 特征 | 状态逻辑 |
|--------|------|----------|
| **chop_grid** | `bpc_semantic_chop` | ≥0.52 `CHOP_ENTRY`；<0.33 `CHOP_EXIT`；中间 `CHOP_HOLD` |
| **trend_scalp** | `trend_confidence` + cap `bpc_semantic_chop` | ≥0.7 且 chop≤0.25 → `MOMENTUM_ENTRY`；否则 `BLOCKED_BY_CHOP` |
| **Router 提示** | 启发式 | 哪侧「更接近可交易」（**不等于** live 正在跑哪条腿） |

释义：「C 是短周期状态机；chop 高 ≠ 全局熊市，只代表 **不宜做 B 式 swing**。」

---

## 4. 调仓策略层（`rebalance_policy`）— 合成逻辑

**原则：** 三层 regime **只作输入**；输出只有 `NAV 目标带` + `告警等级` + `自然语言建议`。

### 4.1 配置文件

`config/monitoring/rebalance_targets.yaml`（新建）：

```yaml
version: 1
# 占总 NAV（spot + rolling + trend + multi_leg；rolling 未上线时归并到 spot 或省略）
scopes:
  spot:
    label: "A·Spot"
    layer: a
  rolling:
    label: "A·Rolling"
    layer: a
    optional: true          # T1 前可缺省
  trend:
    label: "B·Trend"
    layer: b
  multi_leg:
    label: "C·Multi-leg"
    layer: c

# 由 composite_risk_on 分档选择目标带（非单策略 regime）
bands:
  risk_on:                  # 合成 risk-on
    spot:     { min: 0.15, target: 0.25, max: 0.35 }
    rolling:  { min: 0.10, target: 0.20, max: 0.30 }
    trend:    { min: 0.25, target: 0.35, max: 0.45 }
    multi_leg:{ min: 0.08, target: 0.15, max: 0.22 }
  neutral:
    spot:     { min: 0.20, target: 0.30, max: 0.40 }
    rolling:  { min: 0.05, target: 0.10, max: 0.20 }
    trend:    { min: 0.30, target: 0.40, max: 0.50 }
    multi_leg:{ min: 0.10, target: 0.18, max: 0.25 }
  risk_off:
    spot:     { min: 0.05, target: 0.12, max: 0.20 }
    rolling:  { min: 0.00, target: 0.05, max: 0.10 }
    trend:    { min: 0.35, target: 0.45, max: 0.55 }
    multi_leg:{ min: 0.12, target: 0.20, max: 0.28 }

tolerance_pct: 0.05         # 占比偏离 target 超过此值 → WATCH
hard_tolerance_pct: 0.12    # 超过 → REBALANCE_SUGGEST

# composite_risk_on 规则（可调，版本化）
composite:
  # 每项 score 0|1|2；加权求和后映射 risk_on / neutral / risk_off
  inputs:
    - id: a_macro
      weight: 2
      rules:
        - when: "abc_macro_regime_score >= 4"
          score: 2
        - when: "abc_macro_regime_score >= 3"
          score: 1
        - when: "abc_macro_regime_score <= 2"
          score: 0
    - id: a_weekly_not_deep_bear
      weight: 1
      rules:
        - when: "weekly_ema_200_position >= 0"
          score: 2          # 价在周线 EMA 上 → 偏 risk-on（少 deploy 多持有）
        - when: "weekly_ema_200_position < -0.05"
          score: 0          # 深熊 → 偏 deploy 现金，非 risk-on 加仓 beta
    - id: b_bull_share_7d
      weight: 2
      rules:
        - when: "tpc_bull_share_7d >= 0.25"
          score: 2
        - when: "tpc_bull_share_7d >= 0.10"
          score: 1
        - when: "tpc_bull_label == bull"
          score: 1          # 当前 bar 兜底
    - id: c_chop_dominant
      weight: 1
      rules:
        - when: "chop_semantic >= 0.52 and trend_confidence < 0.5"
          score: 0          # 纯 chop → 微观震荡，略降 risk-on 权重
        - when: "trend_confidence >= 0.7"
          score: 1
  map:
    - max_total: 3
      label: risk_off
    - max_total: 6
      label: neutral
    - max_total: 99
      label: risk_on
```

> **注意：** `composite` 是 **调仓专用** 启发式，**不写入** A/B/C 策略 yaml，避免污染交易逻辑。

### 4.2 告警等级

| 等级 | 条件 | CMS 展示 | 动作 |
|------|------|----------|------|
| `OK` | 各 scope 在 `[min,max]` 内 | 绿条 | 无 |
| `WATCH` | 任一 scope 偏离 `target` > `tolerance_pct` | 黄条 | 记录事件 |
| `REBALANCE_SUGGEST` | 偏离 > `hard_tolerance` **或** composite 升/降档与 NAV 明显矛盾 | 红条 | CMS + 可选 TG；**仍不自动划转** |

**矛盾示例（应告警）：** composite=`risk_on`，但 `spot+rolling` 合计 < 15% → 建议「考虑从 B 利润划转或现货买入 beta」。

### 4.3 建议文案模板（规则生成）

```
composite=risk_on, A现货占比偏低(12% < target 25%)
→ 建议：检查 A deploy 窗口；牛市优先持币/现货，可从 B·Trend 已实现利润划转。

composite=risk_off, rolling占比仍高(22%)
→ 建议：降低 rolling 杠杆或暂停滚仓；勿用 B 加长持有替代 A 降仓。
```

---

## 5. API 设计

### 5.1 新端点

```
GET /api/regime/cockpit
  ?symbol=BTCUSDT          # 主参考标的，默认 BTC
  &window_days=7           # bull_share 等滚动窗
```

**响应骨架：**

```json
{
  "ok": true,
  "data": {
    "as_of": "2026-06-12T08:00:00Z",
    "feature_bus": { "path": "...", "age_minutes": 18, "stale": false },
    "symbol": "BTCUSDT",
    "layers": {
      "a_spot": {
        "weekly_ema_200_position": -0.03,
        "deploy_state": "DEEP_BEAR",
        "deploy_allowed": true,
        "abc_macro_regime_score": 3,
        "hint": "深熊区可 deploy；宏观转换期"
      },
      "b_trend": {
        "current_label": "bear",
        "features": { "adx_50": 18.2, "ema_1200_position": -0.05 },
        "bull_share_7d": 0.08,
        "baseline_bull_share": 0.0,
        "drift_alert": false
      },
      "c_multileg": {
        "chop_grid": { "feature": 0.48, "state": "CHOP_HOLD", "entry_min": 0.52 },
        "trend_scalp": { "feature": 0.62, "state": "BELOW_ENTRY", "entry_min": 0.7 },
        "router_hint": "chop_neutral"
      }
    },
    "allocation": {
      "total_nav_usdt": 12500,
      "scopes": [
        { "scope": "spot", "equity_usdt": 2200, "nav_pct": 0.176,
          "band": { "min": 0.15, "target": 0.25, "max": 0.35 }, "status": "WATCH" }
      ],
      "composite": "neutral",
      "alert": "WATCH",
      "suggestions": ["A·Spot 低于目标带，composite=neutral，可选小额补齐 beta"]
    },
    "ops": [ "... 现有 regime_ops 行 ..." ]
  }
}
```

### 5.2 后端模块（建议路径）

| 模块 | 职责 |
|------|------|
| `src/mlbot_console/services/regime_live.py` | 读 feature bus 最新 120T bar；`RegimeConfig.classify`；multileg 阈值比较 |
| `src/mlbot_console/services/rebalance_advisor.py` | 读 `rebalance_targets.yaml`；拉 `exchange_balances`；算 composite + alert |
| `src/mlbot_console/routers/regime.py` | 新增 `GET /api/regime/cockpit`；保留 `regime-ops` |
| `frontend/src/pages/Regime/RegimePage.tsx` | 看板 UI + 保留底表 |

### 5.3 Feature bus 读取（复用现有）

- 路径：`SETTINGS.feature_bus_root` / `features/120T/{symbol}.parquet`（与 trade_map / account marks 一致）
- 列：按各层 yaml 收集 `required_features` 并集
- 陈旧阈值：bus age > 2× bar 周期 → 卡片标 `STALE`，告警降级为 `DATA_STALE`

---

## 6. 实现分期（T2 拆解）

| Phase | 交付 | 估时 |
|-------|------|------|
| **T2a** | `/api/regime/cockpit` 三层 live 卡片 + feature_bus age | ~3d |
| **T2b** | `rebalance_targets.yaml` + NAV 占比条 + composite + 告警条 | ~3d |
| **T2c** | CMS 前端看板（替换 Regime 页上半区） | ~3d |
| **T2d** | `monitor_event` 持久化 + 可选 TG；定时 cron 每 4h | ~2d |

**MVP 边界：**

- ✅ 三层 regime **都显示**，作调仓参考  
- ✅ 告警 + 建议文案  
- ❌ 不自动划转  
- ❌ 不统一 regime  schema  
- ❌ rolling scope 可占位至 T1  

---

## 7. 为何不用「一个 regime」？

| 若强行统一 | 问题 |
|------------|------|
| 用 B bull → 加大 A | A 的买点在 **深熊**，牛市 B bull 时 A 应 **少买多持** |
| 用 C chop → 全局 risk-off | chop 只是 2H 震荡，BTC 周线仍可能 bull |
| 用 A 周线熊 → 关 B | B 的 2H bear 与周线成本区 **时间尺度不同** |

因此：**展示三个真相，调仓策略第四个文件做映射。**

---

## 8. 与监控体系关系

| 现有 | 新看板 |
|------|--------|
| `mlbot monitor watchdog` 周跑 bull_share 漂移 | Cockpit 显示 **同一 classify 逻辑** 的 live + 7d 窗 |
| `/monitoring` 漂移页 | Regime 页偏 **运维调仓**；Monitoring 偏 **策略 promote** |
| `regime_drift_monitor.py` 离线 | Cockpit 在线；漂移结果仍在 Ops 底表 |

---

## 9. 验收清单

- [ ] BTC 最新 bar：A/B/C 三张卡片数值与手动算 `RegimeConfig.classify` 一致  
- [ ] spot/trend/multi_leg 三账户 NAV 占比之和 ≈ 100%  
- [ ] 修改 `rebalance_targets.yaml` 后告警阈值随之变化（无需改代码）  
- [ ] feature bus 缺失时页面不崩溃，标 `DATA_STALE`  
- [ ] 保留原 Regime Ops 表与 drift 列  

---

## 10. 相关文件

| 用途 | 路径 |
|------|------|
| 现 Regime 页 | `frontend/src/pages/Regime/RegimePage.tsx` |
| 现 API | `src/mlbot_console/routers/regime.py` |
| A prefilter | `live/highcap/config/strategies/spot_accum_simple/archetypes/prefilter.yaml` |
| B regime | `live/highcap/config/strategies/tpc/archetypes/regime.yaml` |
| C chop / scalp | `live/highcap/.../chop_grid|trend_scalp/archetypes/regime.yaml` |
| 账户 NAV | `src/mlbot_console/services/exchange_balances.py` |
| classify 实现 | `src/time_series_model/archetype/loader.py` `RegimeConfig` |
| 离线 drift | `src/monitoring/regime_health.py` |

---

*维护：T2 各 Phase 完成后勾选 §9；rolling scope 上线后更新 §4.1 bands。*
