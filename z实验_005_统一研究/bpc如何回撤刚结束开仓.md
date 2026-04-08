# BPC：如何在「回撤刚结束、延续刚起」时开仓

## 1. 目标

- **BPC**：先突破 → 有回踩 → 再延续；入场尽量靠近 **pullback 结束、continuation 刚起**。
- **避免**：与 **ME** 一样在「趋势已明显加速」后才触发（减少两条策略在 PCM 里时间重合）。

## 2. 信号链（谁决定早/晚）

入场由多层共同决定，不是单一文件：

| 层级 | 文件 | 作用 |
|------|------|------|
| Prefilter | `config/strategies/bpc/archetypes/prefilter.yaml` | 语义粗筛（含回踩、恢复等） |
| Direction | `config/strategies/bpc/archetypes/direction.yaml` | 方向与结构/VWAP 带通 |
| Gate | `config/strategies/bpc/archetypes/gate.yaml` | 硬拒绝（含 VWAP 近场死区等） |
| Entry | `config/strategies/bpc/archetypes/entry_filters.yaml` | **最终「是否此刻开仓」**（多 filter **OR**） |
| Execution | `config/strategies/bpc/archetypes/execution.yaml` | 止损/加仓/结构出场（基本不改变「第一根信号 bar」） |

当前 turbo 管线里：`entry_filter.meta_algorithm: false`，rolling 主要吃 **手改 archetypes**，不会自动重扫 entry 组合。

## 3. 改动优先级（建议顺序）

### 3.1 Entry Filter（最高优先）

**文件**：`config/strategies/bpc/archetypes/entry_filters.yaml`

- 当前为 `combination_mode: or`，且 `ef_trend_confirm_tier_a` / `ef_trend_confirm_tier_b` 为 **强趋势确认臂**，容易把入场推到后段，与 ME 重叠。
- **建议**：将上述两条改为 `enabled: false`（可先关一条做 A/B），保留 `ef_bpc_pullback_absorb_dual_compression`、`ef_pullback_reaccept_flow` 等 **pullback / 再接受** 臂。

### 3.2 Gate（第二优先）

**文件**：`config/strategies/bpc/archetypes/gate.yaml`

- `HARD_MACRO_TP_VWAP_1200_DEADZONE`（约 `-0.005 ~ 0.005`）会在 **贴近 VWAP** 时拒单，回调末端常被挡，需等离开死区。
- **建议**：先把死区 **缩窄**（如 `±0.003`）观察，勿一次性删规则。

### 3.3 Prefilter（第三优先，小步）

**文件**：`config/strategies/bpc/archetypes/prefilter.yaml`

- `bpc_recovery_strength >= 0.40` 要求「恢复已较明显」，也会偏晚。
- **建议**：仅在 entry/gate 调完后再试 `0.35` → `0.30`，注意交易数与噪声。

## 4. 一句话

**先动 entry（关 trend_confirm 臂）→ 再缩 gate VWAP 死区 → 最后小调 prefilter recovery。**

（为何这样排、与高原 / ME 错位的整体结论见 **第 7 节**。）

## 5. 一键跑极致回撤不破 lab（唯一保留的 BPC 实验根）

子目录必须叫 **`bpc/`**（与 pipeline 策略键一致）；除下列实体文件外，其余 **symlink** 到主线 `config/strategies/bpc`。

| 项 | 路径 |
|----|------|
| 策略根 | `config/strategies_pullback_lab_extreme_pullback/bpc/` |
| 实体文件 | `archetypes/entry_filters.yaml`、`gate.yaml`、`prefilter.yaml`、`direction.yaml`、`execution.yaml`；根目录 `features_entry_filter.yaml`（含 `macro_tp_vwap_1200_position_f`） |
| Pipeline | `config/prod_train_pipeline_2h_turbo_2024bull_bpc_pullback_lab_extreme_pullback.yaml` |
| 结果目录 | `results/bpc/turbo-rolling-sim-pullback-lab-extreme-pullback` |

**语义摘要（breakout → pullback → 次级别吸收 → 弱化 continue）**：**prefilter** 用 `bpc_is_after_breakout`、`bpc_score_pullback`/`bpc_pullback_depth`、`bpc_score_continuation` 上限与 `bpc_recovery_strength` 下限做粗链；**direction** 用 `bpc_breakout_direction`+MACD 与 `macro_tp_vwap_1200_position` 带通同号；**entry** 在各 OR 臂上再收紧并叠加 `bpc_pullback_delta_absorption`、`bpc_cvd_absorption`、`fp_imbalance_absorption_score` 或 `vpin_absorption_score`；**gate** 仅风控 + VWAP 死区 **±0.001**；**execution** 仍为 `vwap1200` 结构出场，与宏观标尺一致。

仓库根目录（默认即本 lab）：

```bash
./scripts/repair_bpc_pullback_lab_symlinks.sh
chmod +x scripts/run_turbo_rolling_sim_batch.sh
./scripts/run_turbo_rolling_sim_batch.sh
```

单月快检：

```bash
STAGE=fast_month MONTH=2024-09 ./scripts/run_turbo_rolling_sim_batch.sh
```

与主线对比时另跑 `prod_train_pipeline_2h_turbo_2024bull_thresholds_only_bpc_only.yaml`；其他 pipeline 可用 `CONFIGS="..." ./scripts/run_turbo_rolling_sim_batch.sh` 覆盖默认。

克隆后 symlink 若断链：在仓库根执行 `./scripts/repair_bpc_pullback_lab_symlinks.sh`（从 `archetypes/gate.yaml` 生成实体 `gate_draft.yaml`）。

> 已删除曾用的分层 lab（只改 entry / 只改 gate / 只改 prefilter 各一套）：rolling 上效果与主线拉不开差距，避免仓库噪音。

## 6. 管线仍会把手写语义洗回去（本 lab 的对策）

在 **`fast_loop` + 月度阈值校准** 下，仅改 archetypes 不够，除非与 promote / 输入路径对齐：

1. **Entry**：`archetype_plateau` 会扫 **locked** filter；`promote_never_disable: true` 会把臂 **强制写回 enabled**。极致 lab 里新臂使用 **`promote_never_disable: false` + `skip_plateau: true`**（见 `entry_filters.yaml`）。
2. **Gate**：优化成功时仍会改阈值；死区类规则需 **`frozen: true`**，且 promote 合并需识别 **`status: frozen`**（见 `optimize_gate_unified` 历史修复）。
3. **Prefilter**：`disable_model_training` 时可能 **locked prefilter 重锚定**，抹平手写 recovery。本 pipeline 设 **`skip_locked_prefilter_reanchor: true`**。
4. **Gate 草稿**：若 `gate_draft.yaml` symlink 到主线树草稿，月度 gate 与手写 `archetypes/gate.yaml` **脱钩**。repair 脚本用 **`archetypes/gate.yaml` → 实体 `gate_draft.yaml`**。

验收以各月 `strategies_calibrated/bpc/archetypes/` 实参为准。

## 7. 方法论结论：语义先行，再在各层找高原

**是的**：更稳妥的顺序是 **先验证「回撤结束附近能否、且值得开出信号」**，再在这一 **语义前提** 下，对 **prefilter → gate → entry**（或本文 3.1→3.2→3.3 的优先顺序）**分别**做平坦高原 / 显著性门槛内的标定，而不是一上来就在全参数空间里让优化器自由找「最平的那块」。

这样做的意义在于：

1. **语义与统计分工清楚**  
   - **第一阶段（验证）**：用手写或窄范围改动回答「晚入场是不是被某几层挡住、信号时间轴能否前移」——对应机制与敏感性，不是最终上线数值。  
   - **第二阶段（高原）**：在已接受的机制上找 **稳健阈值**（标定窗高原 + OOS / 滚动验收），对应可部署的参数。

2. **更贴 BPC 叙事**  
   高原若不加约束，容易把 OR 组合「优化回」**强趋势确认 / 高参与** 那一类臂上，与「突破→回踩→再延续」的 **位置语义** 脱钩。先固定或冻结语义锚点（如关 trend_confirm 臂、死区规则 `frozen`、recovery 小步），再高原，等价于 **在正确的结构子空间里找数**。

3. **更易与 ME 拆开、争取不同 alpha**  
   ME 偏 **扩张早段 / 动量加速**；BPC 若要在 **回撤刚结束、延续刚起** 拿货，需要在 **时间轴上与 ME 错位**。先证明 BPC 在 pullback 末端有稳定触发，再各层高原，才能讨论「和 ME 低相关的第二条腿」；若 BPC 与 ME 总在同一根加速 bar 上撞车，PCM 里只是在 **同一 alpha 上再分配权重**，很难说是 **不同来源的 alpha**。

4. **与第 6 节的关系**  
   Lab 里的写死 / `skip_plateau` / `frozen` / `skip_locked_prefilter_reanchor` 是为 **通过管线把第一阶段验证跑完**；**通过后**应在同一语义下逐步放开高原（或缩小 frozen 范围），用滚动结果验收，而不是永远停在手写数上。

**实操摘要**：先验收「回撤结束处有信号、且相对 ME 不过度重叠」→ 再按层高原 → 最后才考虑把多条臂一起交给更宽的联合搜索。
