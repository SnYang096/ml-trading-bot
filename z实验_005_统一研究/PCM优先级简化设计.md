# PCM 优先级简化设计 (v1 → v2)

> 创建时间: 2026-02-20
> 关联配置: `config/pcm_regime.yaml`
> 关联代码: `src/time_series_model/portfolio/live_pcm.py`

---

## 1. 问题：v1 两层优先级互相矛盾

v1 同时有两套优先级系统：

| 层 | 机制 | BPC vs ME vs FER 排序 |
|---|---|---|
| Layer 2 (Regime) | NORMAL 静态优先级 | BPC > ME > FER |
| Layer 3 (Override) | 条件覆盖 | FER > ME > BPC |

代码执行顺序：先查 Override → 再走 Regime。
Override 条件宽松（FER evidence ≥ 0.6），多数冲突场景 Layer 2 被架空。

**本质**：两层试图回答同一个问题（"冲突时谁赢"）但给出相反答案。

---

## 2. 方案：合并为单一优先级

### 决策依据：信号条件严格性

越严格的 archetype 触发越稀有，冲突时应被尊重：

```
LV (liquidation cluster + OI异常)  →  最严格，触发最少
FER (均衡偏离反转)                  →  次严格
ME (动能扩张)                       →  中等
BPC (趋势延续)                      →  最宽松，触发最频繁
```

### 为什么 BPC 最低优先不影响"骨架"角色

- BPC 94% 时间无竞争者 → 自然胜出
- 冲突率仅 5.52% → 优先级只在极少数情况下起作用
- "骨架"靠的是**触发频率**，不是**冲突优先级**

### v2 Regime 定义

| Regime | 优先级 | 语义 |
|--------|--------|------|
| NORMAL | LV > FER > ME > BPC | 按条件严格性 |
| HIGH_VOL | LV > ME > FER > BPC | ME 擅长高波动，提升至 FER 之上 |
| HIGH_LEVERAGE | LV > FER > ME > BPC | 与 NORMAL 相同（LV 已在首位） |

---

## 3. 删除的内容

### Override 段 (L84-121)

```yaml
# 已删除
override:
  LV:
    overrides: "ALL"
    min_evidence: 0.0
  FER:
    overrides: ["ME"]
    min_evidence: 0.6
  ME:
    overrides: ["BPC"]
    min_evidence: 0.7
    conditions:
      - feature: "atr_percentile"
        operator: ">"
        threshold: 0.75
```

**删除原因**：Override 的意图（LV > FER > ME > BPC）现在就是 NORMAL regime 的优先级。

### 代码保留

`_check_override()` 方法保留但不触发（`_override_config` 为空 → 立即返回 None）。
如果未来需要重新引入极端覆盖逻辑，无需改代码，只需在 YAML 加回 `override` 段。

---

## 4. 变更范围

| 文件 | 变更 |
|------|------|
| `config/pcm_regime.yaml` | 移除 override 段，更新 3 个 regime 优先级 |
| `src/.../live_pcm.py` | 更新 `DEFAULT_REGIME_PRIORITIES` + `DEFAULT_ARCHETYPE_PRIORITY` |
| `scripts/backtest_execution_layer.py` | 更新 `_PCM_DEFAULT_PRIORITY` |
| `scripts/run_live.py` | 更新 `archetype_priority` 参数 |
| `scripts/demo_three_strategies.py` | 更新 `archetype_priority` 参数 |
| `tests/unit/test_live_pcm.py` | 更新 8 个测试的期望值，54/54 通过 |

---

## 5. 验证 TODO

- [ ] 用历史 predictions.parquet 重跑 PCM 回测，对比 v1 vs v2 的冲突解决效果
- [ ] 确认 v2 优先级下 Sharpe 不退化（冲突率低，预期影响小）
- [ ] 反事实分析：v2 被拒信号的事后 R 和胜率
