# 120T 实验 TODO（仅 120T，不跑 60T/240T）

## 0. 目标与范围

- 只关注 `120T`：
  - `bpc-long-120T`
  - `bpc-short-120T`
  - `fer-long-120T`
  - `fer-short-120T`
  - `me-long-120T`
  - `me-short-120T`
- 不同时上线 `240T` 和 `120T`，本轮只做 `120T` 运维模拟与上线验收。

---

## 1. 配置准备（一次性）

- [ ] 复制 2024 趋势窗口配置（strict/turbo）
- [ ] 复制 2025 震荡窗口配置（strict/turbo）
- [ ] 确认 `strategy_scope.direction`、`dates`、`rolling.mode`

### 1.1 建议配置文件

- `config/prod_train_pipeline_2h_strict_2024bull.yaml`
- `config/prod_train_pipeline_2h_turbo_2024bull.yaml`
- `config/prod_train_pipeline_2h_2025_range_strict.yaml`
- `config/prod_train_pipeline_2h_2025_range_turbo.yaml`
- （可选极速）`config/prod_train_pipeline_2h_turbo_2024bull_thresholds_only.yaml`

### 1.2 配置复制命令

```bash
cp config/prod_train_pipeline_2h.yaml config/prod_train_pipeline_2h_strict_2024bull.yaml
cp config/prod_train_pipeline_2h.yaml config/prod_train_pipeline_2h_turbo_2024bull.yaml
cp config/prod_train_pipeline_2h.yaml config/prod_train_pipeline_2h_2025_range_strict.yaml
cp config/prod_train_pipeline_2h.yaml config/prod_train_pipeline_2h_2025_range_turbo.yaml
```

---

## 2. L0 单策略快速调试（rolling + execution）

- [ ] 先 `bpc-long-120T` 跑一个月（fast_month）
- [ ] 再跑该策略 rolling_sim
- [ ] 检查 `monthly_ledger.jsonl`、`stitched_summary.json`、交易地图

### 命令（L0）

```bash
# 单月快测（例：2024-03）
mlbot pipeline run --strategy bpc-long-120T --config config/prod_train_pipeline_2h_strict_2024bull.yaml --stage fast_month --month 2024-03

# 单策略滚动
mlbot pipeline run --strategy bpc-long-120T --config config/prod_train_pipeline_2h_strict_2024bull.yaml --stage rolling_sim

mlbot pipeline run --strategy bpc-long-120T --config config/prod_train_pipeline_2h_turbo_2024bull_thresholds_only.yaml --stage rolling_sim --skip-shap

mlbot pipeline run --strategy bpc-short-120T bpc-long-120T  --config config/prod_train_pipeline_2h_turbo_2024bull_thresholds_only.yaml --stage rolling_sim --skip-shap

mlbot pipeline run --all --config config/prod_train_pipeline_2h_turbo_2024bull_thresholds_only.yaml --stage rolling_sim --skip-shap

mlbot pipeline run --all --config config/prod_train_pipeline_2h_turbo_2024bull_thresholds_only_bpc_only.yaml --stage rolling_sim --skip-shap

python scripts/plot_monthly_threshold_drift.py \
  --run-root results/120T/prod_train_history/_rolling_sim/20260328_174021 \
  --strategy bpc-long-120T  bpc-short-120T
```

---

## 3. L1 趋势专项（仅 BPC 120T）

- [ ] 跑 `bpc-long-120T` + `bpc-short-120T`
- [ ] 看趋势月贡献、加仓次数、近止损率、回撤

### 命令（L1）

```bash
for s in bpc-long-120T bpc-short-120T; do
  mlbot pipeline run --strategy "$s" --config config/prod_train_pipeline_2h_strict_2024bull.yaml --stage rolling_sim
done
```

---

## 4. L2 真实运维模拟（全 120T 组合）

- [ ] 120T 六策略全量 rolling_sim（主结论）
- [ ] strict 与 turbo 对照
- [ ] 记录 run_id 做后续复盘

### 命令（L2）

```bash
# strict（主验收）
mlbot pipeline run --all --config config/prod_train_pipeline_2h_strict_2024bull.yaml --stage rolling_sim

# turbo（快速对照）
mlbot pipeline run --all --config config/prod_train_pipeline_2h_turbo_2024bull.yaml --stage rolling_sim

# 极速阈值版（可选）
mlbot pipeline run --all --config config/prod_train_pipeline_2h_turbo_2024bull_thresholds_only.yaml --stage rolling_sim
```

---

## 5. L3 跨 Regime 验收（2024 + 2025）

- [ ] 2024 趋势窗口：验证进攻能力与趋势骑乘
- [ ] 2025 震荡窗口：验证控回撤与稳定性
- [ ] 双窗口都通过才可上线

### 命令（L3）

```bash
# 2024
mlbot pipeline run --all --config config/prod_train_pipeline_2h_strict_2024bull.yaml --stage rolling_sim
mlbot pipeline run --all --config config/prod_train_pipeline_2h_turbo_2024bull.yaml --stage rolling_sim

# 2025
mlbot pipeline run --all --config config/prod_train_pipeline_2h_2025_range_strict.yaml --stage rolling_sim
mlbot pipeline run --all --config config/prod_train_pipeline_2h_2025_range_turbo.yaml --stage rolling_sim
```

---

## 6. L4 高原专项（上线前门禁）

- [ ] execution 高原
- [ ] slot 高原
- [ ] 汇总评分卡（不通过则回滚）

### 命令（L4）

```bash
# execution 参数高原
mlbot pipeline run --all --config config/prod_train_pipeline_2h_strict_2024bull.yaml --stage execution_opt
mlbot pipeline run --all --config config/prod_train_pipeline_2h_strict_2024bull.yaml --stage event_backtest

# slot 高原
mlbot pipeline run --all --config config/prod_train_pipeline_2h_strict_2024bull.yaml --stage pcm_slot_grid
```

---

## 7. 分层调试命令（只跑到某层）

```bash
mlbot pipeline run --strategy bpc-long-120T --config config/prod_train_pipeline_2h_strict_2024bull.yaml --stage prefilter
mlbot pipeline run --strategy bpc-long-120T --config config/prod_train_pipeline_2h_strict_2024bull.yaml --stage gate
mlbot pipeline run --strategy bpc-long-120T --config config/prod_train_pipeline_2h_strict_2024bull.yaml --stage entry_filter
mlbot pipeline run --strategy bpc-long-120T --config config/prod_train_pipeline_2h_strict_2024bull.yaml --stage event_backtest
```

---

## 8. 结果复盘命令

> 先从滚动输出日志中记下 `<run_id>`，再执行。

```bash
mlbot pipeline report-side-state --run-id <run_id> --config config/prod_train_pipeline_2h_strict_2024bull.yaml

mlbot pipeline debug-quality --run-id <run_id> --month 2024-06 --config config/prod_train_pipeline_2h_strict_2024bull.yaml
mlbot pipeline debug-quality --run-id <run_id> --month 2024-10 --config config/prod_train_pipeline_2h_strict_2024bull.yaml
mlbot pipeline debug-quality --run-id <run_id> --month 2025-03 --config config/prod_train_pipeline_2h_2025_range_strict.yaml
mlbot pipeline debug-quality --run-id <run_id> --month 2025-09 --config config/prod_train_pipeline_2h_2025_range_strict.yaml
```

---

## 9. 上线门槛（120T）

- [ ] 2024 与 2025 两个窗口都通过
- [ ] `stitched_total_r > 0`
- [ ] `max_drawdown_r` 可接受（相对历史基线不恶化）
- [ ] `near_stop_rate` 不明显恶化
- [ ] 交易数达标（防止靠极少交易抬指标）
- [ ] execution/slot 高原通过（邻域不脆弱）

---

## 10. 回滚顺序（固定）

1. execution 参数  
2. slot case  
3. 慢变量结构  

