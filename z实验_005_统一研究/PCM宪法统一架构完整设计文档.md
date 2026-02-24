# PCM 宪法统一架构完整设计文档

> 创建: 2026-02-17 | 最后更新: 2026-02-24
> 状态: **设计完成，待实现**

---

## 1. 问题诊断：6 个现存矛盾

### 矛盾 ①：Slot 数量双源定义

| 位置 | 字段 | 值 |
|------|------|----|
| `config/constitution/constitution.yaml` → `slots.slot_count` | 宪法 slot 上限 | **2** |
| `config/pcm_regime.yaml` → `max_slots` | PCM slot 上限 | **2** |

**问题**: 两个文件各自定义 max_slots，如果修改一个忘改另一个就产生行为不一致。
`ConstitutionExecutor.reserve_slot()` 读 constitution.yaml；`LivePCM.__init__()` 读 pcm_regime.yaml。

**解决**: Constitution 是唯一权威源。PCM 从 constitution 读取 slot_count，pcm_regime.yaml 移除 max_slots。

### 矛盾 ②：风险预算双源定义

| 位置 | 字段 | 值 |
|------|------|----|
| `constitution.yaml` → `slots.risk_per_slot` | 全局每 slot 风险 | **0.015** |
| `pcm_regime.yaml` → `archetype_risk.{arch}.max_risk_per_trade` | 每策略风险 | **0.015/0.010** |

**问题**: Constitution 定义"每 slot 1.5%"，但 PCM 定义"LV 每笔 1.0%"。
当 LV 占一个 slot 时，用 1.5% 还是 1.0%？两个文件没有明确的上下级关系。

**解决**: Constitution 定义 **全局上限** (`risk_per_slot` = 每 slot 最大风险)。
PCM 定义 **每策略实际使用** (`max_risk_per_trade`)，必须 ≤ `risk_per_slot`。
加入运行时校验：`pcm.archetype_risk[arch] <= constitution.risk_per_slot`。

### 矛盾 ③：执行路径断裂（回测 vs 实盘）

| 路径 | 宪法检查 | PCM 仲裁 | Slot 管理 |
|------|---------|---------|----------|
| **实盘** (`run_live.py`) | ✅ `enforce_before_order()` | ✅ `LivePCM.decide()` | ✅ `ConstitutionExecutor.reserve_slot()` |
| **回测** (`backtest_execution_layer.py --pcm`) | ❌ **缺失** | ✅ `_run_pcm_mode()` 内联仲裁 | ❌ **缺失** |

**问题**: PCM 回测不模拟宪法约束（kill switch / drawdown / slot 占用），
导致回测可能批准实盘会被拒的交易，回测 Sharpe 虚高。

**解决**: 在 `_run_pcm_mode()` 中加入宪法模拟：跟踪 equity 曲线 → 计算 drawdown → 检查 kill switch。
用模拟版 ConstitutionExecutor（不持久化，但执行相同逻辑）。

### 矛盾 ④：PCM 回测未集成到研究 Pipeline

| 步骤 | 当前状态 |
|------|---------|
| Step 0-8 | ✅ 单策略训练全自动 |
| Step 9 | ✅ 单策略回测 |
| Step 9.5 (PCM 联合回测) | ❌ **缺失** |
| Step 10 | ✅ Baseline 导出 |

**问题**: `auto_research_pipeline.py` 只做单策略回测（Step 9），没有 PCM 联合回测步骤。
无法自动验证"多策略联合后 Sharpe 是否仍可接受"、"冲突率是否合理"。

**解决**: 新增 Step 9.5 — 自动收集所有已训练策略的 predictions，跑 PCM 联合回测，
输出结构化 `pcm_stats.json`，纳入 ADOPT/KEEP/ALERT 决策。

### 矛盾 ⑤：PCM 决策统计只打印不保存

**问题**: `_run_pcm_mode()` 打印冲突数/per-archetype/反事实到 console，
但不输出结构化文件。Pipeline 无法自动读取和对比。

**解决**: 新增 `--pcm-stats-json <path>` 参数，输出：
```json
{
  "total_entries": 1162,
  "conflicts": 5,
  "conflict_rate": 0.0043,
  "per_archetype": {"bpc": {"trades": 148, "mean_r": 0.28, ...}, ...},
  "counterfactual": {"bpc": {"rejected": 3, "mean_r": -0.12}, ...},
  "constitution_sim": {"max_dd_hit": false, "peak_dd": 0.08, ...},
  "sharpe_daily": 3.26,
  "sharpe_per_trade": 0.196
}
```

### 矛盾 ⑥：PCM 配置无 promote / adopt 机制

**问题**: 单策略 archetypes (gate/evidence/execution) 有 `--promote` 自动写回实验目录。
但 PCM 配置 (`pcm_regime.yaml`) 没有类似机制，修改 regime 阈值或优先级后无法自动验证和采纳。

**解决**: `pcm_regime.yaml` 纳入研究 pipeline 管理：
- PCM 回测后生成 `pcm_stats.json`
- 对比前后版本 PCM 指标，决定是否 ADOPT
- ADOPT 时自动将 `pcm_regime.yaml` 复制到实验快照

---

## 2. 统一架构设计

### 2.1 配置权威层级

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 0: Constitution (不可违反的硬约束)                    │
│  config/constitution/constitution.yaml                      │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ kill_switch: max_dd, daily/weekly/monthly loss       │    │
│  │ slots: slot_count, risk_per_slot                     │    │
│  │ add_position: policy                                 │    │
│  │ resource_allocation: ← NEW 统一资源约束              │    │
│  └─────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────┤
│  Layer 1: PCM Regime (可调整的仲裁策略)                      │
│  config/pcm_regime.yaml                                     │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ regimes: NORMAL/HIGH_VOL/HIGH_LEVERAGE 优先级        │    │
│  │ detection: regime 检测阈值                           │    │
│  │ archetype_risk: 每策略风险 (≤ constitution 上限)     │    │
│  │ constitution_ref: 引用 constitution.yaml             │    │
│  └─────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────┤
│  Layer 2: Per-Strategy Config (每策略独立)                   │
│  config/strategies/{arch}/archetypes/*.yaml                 │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ gate / evidence / entry_filters / execution / ...    │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

**读取顺序** (实盘 / 回测统一):
1. 加载 `constitution.yaml` → 获取 `slot_count`, `risk_per_slot`, `resource_allocation`
2. 加载 `pcm_regime.yaml` → 获取 `regimes`, `detection`, `archetype_risk`
3. **校验**: `archetype_risk[*].max_risk_per_trade ≤ constitution.risk_per_slot`
4. **校验**: `pcm_regime.yaml` 不得定义 `max_slots` (从 constitution 读取)
5. 加载各策略 `archetypes/*.yaml`

### 2.2 Constitution 新增 `resource_allocation` 段

```yaml
# ── 新增 (在 constitution.yaml 中) ──
resource_allocation:
  # 宪法层面: 每策略最大资源约束
  # PCM 在此约束内自由分配
  per_strategy_limits:
    bpc:
      max_slots: 2            # BPC 最多占 2 个 slot (骨架策略)
      allow_add_position: true
    me:
      max_slots: 2
      allow_add_position: true
    fer:
      max_slots: 2
      allow_add_position: false  # FER 反转不加仓
    lv:
      max_slots: 1            # LV 快进快出，最多 1 个
      allow_add_position: false

  # PCM 配置引用 (非硬约束，告诉宪法去哪里找 PCM 配置)
  pcm_config_ref: "config/pcm_regime.yaml"
```

### 2.3 PCM Regime 配置调整

```yaml
# config/pcm_regime.yaml — 移除 max_slots，新增 constitution_ref
version: 3

# 从 constitution.yaml 读取硬约束 (slot_count, risk_per_slot)
constitution_ref: "config/constitution/constitution.yaml"
# ↑ 运行时 PCM 会校验: archetype_risk ≤ constitution.risk_per_slot

# max_slots: 删除! 从 constitution.yaml 的 slots.slot_count 读取

# 保持不变: regimes / detection / min_bars_in_regime / archetype_risk
```

### 2.4 实盘执行流 (统一后)

```
WebSocket Tick
  → IncrementalFeatureComputer → features
  → LivePCM.decide(features, symbol)        ← Layer 1 仲裁
    ├─ RegimeDetector.detect(features)       ← 动态优先级
    ├─ 遍历策略 → 收集候选信号
    ├─ 优先级 + Evidence 排序 → 选 winner
    └─ return [TradeIntent]
  → enforce_before_order(...)                ← Layer 0 宪法检查
    ├─ validate_drawdown()                   ← kill switch
    ├─ validate_resource_allocation()        ← NEW: 每策略 slot 上限
    ├─ reserve_slot()                        ← slot 占用
    └─ persist state
  → OrderManager.place_order()
```

### 2.5 回测执行流 (统一后)

```
backtest_execution_layer.py --pcm
  ├─ 加载 constitution.yaml → slot_count, risk_per_slot, resource_allocation
  ├─ 加载 pcm_regime.yaml → regimes, detection, archetype_risk
  ├─ 校验一致性
  ├─ 逐 bar 仲裁:
  │   ├─ PCM 仲裁 (优先级 + Evidence)
  │   ├─ Slot 计数约束 (max concurrent = slot_count)
  │   ├─ per_strategy_limits 约束 (每策略 max_slots)
  │   └─ Drawdown 模拟 (equity curve → peak → dd → kill switch?)
  ├─ Bar-by-bar 执行模拟
  ├─ 输出:
  │   ├─ Console 统计 (现有)
  │   ├─ pcm_stats.json (NEW: 结构化统计)
  │   ├─ HTML 报告 + 交易地图 (现有)
  │   └─ Constitution 模拟结果 (是否触发 kill switch)
```

---

## 3. PCM 回测统计输出规范 (`pcm_stats.json`)

### 3.1 Schema

```json
{
  "version": 1,
  "timestamp": "2026-02-24T12:00:00Z",
  "archetypes": ["bpc", "me", "fer"],
  "config": {
    "slot_count": 2,
    "risk_per_slot": 0.015,
    "priority_default": ["LV", "FER", "ME", "BPC"],
    "regime_enabled": true
  },
  "overall": {
    "total_entries": 1162,
    "span_years": 2.5,
    "trades_per_year": 465,
    "mean_r": 0.3548,
    "std_r": 1.8071,
    "win_rate": 0.5843,
    "sharpe_per_trade": 0.1963,
    "sharpe_annualized": 4.13,
    "sharpe_daily": 3.26
  },
  "arbitration": {
    "total_signals": 1200,
    "conflicts": 5,
    "conflict_rate": 0.0042,
    "slot_rejections": 3,
    "constitution_rejections": 0
  },
  "per_archetype": {
    "bpc": {
      "signals": 1100,
      "wins": 148,
      "win_rate_in_conflicts": 0.60,
      "trades": 148,
      "mean_r": 0.2844,
      "sharpe": 0.2113,
      "win_rate": 0.730
    },
    "fer": {
      "signals": 1020,
      "wins": 1014,
      "win_rate_in_conflicts": 0.80,
      "trades": 1014,
      "mean_r": 0.3650,
      "sharpe": 0.1957,
      "win_rate": 0.563
    }
  },
  "counterfactual": {
    "bpc": {"rejected": 3, "mean_r": -0.12, "win_rate": 0.33},
    "fer": {"rejected": 0, "mean_r": null, "win_rate": null}
  },
  "regime_stats": {
    "NORMAL": {"bars": 9500, "pct": 0.95, "entries": 1100},
    "HIGH_VOL": {"bars": 450, "pct": 0.045, "entries": 55},
    "HIGH_LEVERAGE": {"bars": 50, "pct": 0.005, "entries": 7}
  },
  "constitution_sim": {
    "peak_equity": 1.45,
    "min_equity": 0.92,
    "max_drawdown": 0.085,
    "max_dd_limit": 0.20,
    "kill_switch_triggered": false,
    "daily_loss_breach": 0,
    "weekly_loss_breach": 0,
    "monthly_loss_breach": 0
  }
}
```

### 3.2 在回测中打印的新增统计段

```
================================================================================
📊 PCM DECISION STATISTICS
================================================================================

🔀 Arbitration Summary:
   Total signals:      1200
   Conflicts:          5 (0.42%)
   Slot rejections:    3
   Constitution blocks: 0

📊 Per-Archetype Arbitration:
   Archetype   Signals   Wins   WinRate(conflict)  Trades  MeanR   Sharpe  Win%
   ─────────────────────────────────────────────────────────────────────────────
   bpc         1100      148    60.0%              148     0.2844  0.2113  73.0%
   fer         1020      1014   80.0%              1014    0.3650  0.1957  56.3%

🌐 Regime Distribution:
   NORMAL:         9500 bars (95.0%), 1100 entries
   HIGH_VOL:       450 bars (4.5%), 55 entries
   HIGH_LEVERAGE:  50 bars (0.5%), 7 entries

🛡️ Constitution Simulation:
   Max Drawdown:     8.5% (limit: 20.0%) ✅
   Kill Switch:      NOT triggered ✅
   Daily Loss Max:   2.1% (limit: 4.0%) ✅
   Weekly Loss Max:  3.2% (limit: 8.0%) ✅
   Monthly Loss Max: 5.1% (limit: 12.0%) ✅

🔍 Counterfactual (rejected signals):
   bpc rejected: 3 trades, mean_R=-0.1200, win=33.33%
```

---

## 4. 研究 Pipeline 集成方案

### 4.1 训练链升级: 11 步 → 12 步

```
Step 0:   Data Download + Convert
Step 1:   Feature Store Build
Step 2:   Prepare Only (features_labeled.parquet)
Step 3:   Prefilter Analyze (--promote)
Step 4:   Direction Validation (--promote)
Step 5:   Gate Optimize (--promote)
Step 6:   Evidence Optimize (--promote)
Step 7:   Entry Filter Optimize (--promote)
Step 8:   Execution Grid Optimize (--promote)
Step 9:   Single-Strategy Backtest
Step 9.5: PCM Joint Backtest (NEW)           ← 多策略联合 + pcm_stats.json
Step 10:  Export Training Baseline
  ↓
决策 (扩展):
  单策略决策: 同现有 (Sharpe 比值 → ADOPT/KEEP/ALERT)
  PCM 联合决策 (NEW):
    conflict_rate > 0.15          → ALERT
    constitution_sim.kill_switch  → ERROR
    pcm_sharpe_daily < 1.0        → ALERT
    pcm_sharpe / sum(individual)  → 组合效率比
```

### 4.2 Step 9.5 实现细节

**触发条件**: 当前策略训练完成后，检查是否存在其他已训练策略的 predictions：
```python
# 扫描 results/ 下所有策略的最新 predictions.parquet
other_predictions = find_latest_predictions(
    exclude=current_strategy,
    strategies=["bpc", "me", "fer"]
)
if len(other_predictions) >= 1:
    # 有其他策略可联合 → 执行 PCM 回测
    run_pcm_joint_backtest(
        current_strategy_predictions,
        other_predictions,
        pcm_stats_output=f"{run_dir}/pcm_stats.json"
    )
```

**命令生成**:
```bash
python scripts/backtest_execution_layer.py \
  --pcm bpc:{bpc_predictions} fer:{fer_predictions} me:{me_predictions} \
  --output {run_dir}/pcm_trading_map.html \
  --pcm-stats-json {run_dir}/pcm_stats.json \
  --constitution config/constitution/constitution.yaml
```

### 4.3 PCM 统计自动 Promote / Adopt

PCM 回测完成后，`pcm_stats.json` 保存到实验目录：

```
results/research_history/{strategy}/{YYYYMMDD_HHMMSS}/
  ├── report.json                    # 单策略指标
  ├── pcm_stats.json                 # NEW: PCM 联合指标
  ├── pcm_regime_snapshot.yaml       # NEW: PCM 配置快照
  ├── archetypes/                    # 策略配置快照
  └── pipeline.log
```

**对比决策 (扩展)**:

| 指标 | 条件 | 决策 |
|------|------|------|
| `pcm_stats.sharpe_daily` | ≤ 0 | ERROR: 联合后亏损 |
| `pcm_stats.conflict_rate` | > 0.15 | ALERT: 冲突率过高 |
| `pcm_stats.constitution_sim.kill_switch` | true | ERROR: 回测触发熔断 |
| `pcm_stats.sharpe_daily` / `prev.sharpe_daily` | < 0.7 | ALERT: 联合 Sharpe 显著衰减 |
| 上述均通过 | - | PASS: PCM 联合验证通过 |

**最终决策矩阵**:
```
单策略决策 = ADOPT  AND  PCM决策 = PASS  → 最终 ADOPT
单策略决策 = ADOPT  AND  PCM决策 = ALERT → 最终 ADOPT + WARNING
单策略决策 = ADOPT  AND  PCM决策 = ERROR → 最终 KEEP (需人工检查)
单策略决策 ≠ ADOPT                       → 按单策略决策
```

---

## 5. 实现计划

### Phase 1: 配置统一 (纯配置变更，零代码)

- [ ] `constitution.yaml` 添加 `resource_allocation` 段
- [ ] `pcm_regime.yaml` 升级到 v3: 移除 `max_slots`，添加 `constitution_ref`
- [ ] `live/highcap/config/constitution/constitution.yaml` 同步更新
- [ ] 验证: 两份 constitution.yaml 一致

### Phase 2: ConstitutionExecutor 增强

- [ ] `constitution_executor.py`: 新增 `validate_resource_allocation()` 方法
  - 检查 per_strategy slot 使用是否超限
  - 检查 archetype risk ≤ constitution risk_per_slot
- [ ] `enforcement.py`: `enforce_before_order()` 调用新增校验
- [ ] 测试: 单元测试验证 per_strategy_limits 约束

### Phase 3: PCM 统一加载

- [ ] `live_pcm.py`: `LivePCM.__init__()` 接受 constitution_config
  - `max_slots` 从 constitution 读取，不再从 pcm_regime.yaml 读取
  - 启动时校验 `archetype_risk ≤ risk_per_slot`
- [ ] `run_live.py`: 加载顺序统一
  ```python
  constitution = ConstitutionExecutor(constitution_yaml)
  pcm = LivePCM(
      max_slots=constitution.slot_count,     # 从宪法读取
      regime_config_path="config/pcm_regime.yaml",
  )
  ```
- [ ] 测试: 校验一致性的单元测试

### Phase 4: 回测宪法模拟

- [ ] `backtest_execution_layer.py` `_run_pcm_mode()` 增强:
  - 加载 `--constitution` 配置
  - 逐 bar 跟踪 equity curve → 计算 drawdown
  - 模拟 kill switch 逻辑 (当 dd > max_dd 时停止新入场)
  - 模拟 per_strategy slot 限制
  - 新增 `--pcm-stats-json` 输出参数
- [ ] 测试: 构造 drawdown > 20% 的场景验证 kill switch 模拟

### Phase 5: PCM 统计输出

- [ ] `_run_pcm_mode()` 收集完整 PCM 统计到 dict
- [ ] 新增打印段: "PCM DECISION STATISTICS" (含仲裁/regime/宪法模拟)
- [ ] `--pcm-stats-json` 写出 JSON
- [ ] 解析器: `parse_pcm_stats_stdout()` 供 pipeline 使用

### Phase 6: 研究 Pipeline 集成

- [ ] `auto_research_pipeline.py` 新增 Step 9.5
  - 扫描其他策略最新 predictions
  - 调用 PCM 联合回测 + `--pcm-stats-json`
  - 解析 pcm_stats.json
- [ ] 扩展 `save_report()`: 包含 pcm_stats
- [ ] 扩展对比决策: 加入 PCM 联合 Sharpe / conflict_rate / constitution_sim
- [ ] 快照: 保存 `pcm_regime_snapshot.yaml` 到实验目录
- [ ] 测试: dry-run 验证 Step 9.5 命令正确

### Phase 7: 验证 & 文档

- [ ] 端到端验证: BPC + FER PCM 联合回测 → pcm_stats.json → 正确解析
- [ ] 验证实盘路径: LivePCM 从 constitution 读取 slot_count
- [ ] 运行所有现有测试确保无回归

---

## 6. 验证方案

### 6.1 配置一致性验证 (自动, CI 可用)

```bash
# 新增脚本: scripts/validate_constitution_pcm_consistency.py
python scripts/validate_constitution_pcm_consistency.py \
  --constitution config/constitution/constitution.yaml \
  --pcm-regime config/pcm_regime.yaml
```

**检查项**:
1. `pcm_regime.yaml` 不含 `max_slots` (已迁移到 constitution)
2. 所有 `archetype_risk.*.max_risk_per_trade ≤ constitution.risk_per_slot`
3. 所有 `archetype_risk` 中的 archetype 都在 `resource_allocation.per_strategy_limits` 中有定义
4. `per_strategy_limits.*.max_slots ≤ slots.slot_count`
5. `constitution_ref` 指向的文件存在
6. config/ 和 live/highcap/config/ 的 constitution.yaml 关键字段一致

**输出**: `✅ 6/6 checks passed` 或 `❌ check 2 failed: LV risk 0.015 > risk_per_slot 0.010`

### 6.2 回测对比验证

| 测试 | 方法 | 预期 |
|------|------|------|
| Slot 约束生效 | 构造 3 个同时信号 + slot_count=2 | 第 3 个被拒 |
| per_strategy 约束 | LV max_slots=1, 制造 2 个 LV 信号 | 第 2 个 LV 被拒 |
| Drawdown kill | 注入 -25% drawdown 数据 | 后续 bar 全部 NO_TRADE |
| 宪法 vs 无宪法 | 同数据对比 | 有宪法 trades ≤ 无宪法 trades |
| PCM 回测 = 实盘逻辑 | 同优先级/同配置 | 仲裁结果完全一致 |

### 6.3 Pipeline 端到端验证

```bash
# Step 1: 单策略训练 (使用已有数据)
python scripts/auto_research_pipeline.py --strategy fer --dry-run

# Step 2: 验证 Step 9.5 命令生成
# 预期: 打印 backtest_execution_layer.py --pcm ... --pcm-stats-json ...

# Step 3: 实际运行 (需已有 predictions)
python scripts/backtest_execution_layer.py \
  --pcm bpc:results/.../bpc/predictions.parquet \
       fer:results/.../fer/predictions.parquet \
  --pcm-stats-json /tmp/test_pcm_stats.json \
  --constitution config/constitution/constitution.yaml

# Step 4: 验证 JSON 输出
python -c "import json; d=json.load(open('/tmp/test_pcm_stats.json')); \
  assert 'overall' in d; assert 'constitution_sim' in d; print('✅ OK')"
```

---

## 7. 兼容性

### 7.1 向后兼容

- `constitution.yaml` 新增 `resource_allocation` 段，现有代码不读此段 → 无影响
- `pcm_regime.yaml` 仍保留 `max_slots` 作为 deprecated fallback (代码优先读 constitution)
- 现有单策略回测 (`--logs --strategy bpc`) 不受影响
- 现有 PCM 回测 (`--pcm`) 不带 `--constitution` 时行为不变

### 7.2 渐进式实施

```
Week 1: Phase 1-2 (配置 + ConstitutionExecutor 增强)
  → 运行测试确认无回归
Week 2: Phase 3-4 (PCM 统一加载 + 回测宪法模拟)
  → BPC+FER 联合回测验证
Week 3: Phase 5-6 (统计输出 + Pipeline 集成)
  → 端到端 dry-run + 实际运行
Week 4: Phase 7 (验证 + 文档)
  → 全量测试通过，可进入实盘部署
```
