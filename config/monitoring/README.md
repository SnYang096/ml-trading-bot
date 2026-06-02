# Monitoring manifests

Authoritative guide: [`docs/strategy/漂移监控_mlbot_monitor_CN.md`](../../docs/strategy/漂移监控_mlbot_monitor_CN.md).

Promote 因果门槛 + bundle checklist：[`config/experiments/LAYER_PROMOTION_CRITERIA.md`](../experiments/LAYER_PROMOTION_CRITERIA.md) §4.

## 本地监控 bundle（promote 后 · 先做这套）

**在本地完成 reference 标定并 git push**；远程 cron 只消费 git 里的 baseline/plateaus + 自产 bus current。标定窗建议 **`recent_6m_oos`**（见 [`config/market_segment.yaml`](../market_segment.yaml)）。

### 0. 数据 + 标定 parquet

```bash
cd /path/to/ml_trading_bot

# 若 archive 未齐，先拉数据（示例 highcap）
mlbot data download --symbols BTCUSDT,ETHUSDT --start-year 2025 --start-month 1
mlbot data convert

# 标定窗 features + label（以 TPC 为例；bpc/me/srb 换 --config）
mlbot train final --prepare-only \
  --config config/strategies/tpc \
  --symbol BTCUSDT \
  --timeframe 120T \
  --segment recent_6m_oos

export PARQ=$(ls -t results/train_final/tpc/train_final_*/tpc/features_labeled.parquet | head -1)
echo "PARQ=$PARQ"
```

### 1. 写 `last_calibration.plateaus`

现网 regime 规则多在 `any_of` 里（如 `ema_1200_position`），`regime_threshold_calibrate` **仅匹配顶层单条 rule**。推荐：在 `$PARQ` 上分位后 **手写入** `config/strategies/<slug>/archetypes/regime.yaml`：

```bash
python - <<'PY'
import os, pandas as pd
p = os.environ["PARQ"]
s = pd.to_numeric(pd.read_parquet(p, columns=["ema_1200_position"])["ema_1200_position"], errors="coerce").dropna()
for q in (0.1, 0.25, 0.5, 0.75, 0.9):
    print(f"p{int(q*100):02d}", float(s.quantile(q)))
PY
```

在 `last_calibration.plateaus` 增加一条（drift 只比 feature 列 + 区间）：

```yaml
plateaus:
  - feature: ema_1200_position
    operator: ">="
    plateau: { start: <p10>, end: <p90>, mid: <p50> }
```

对 **bpc / tpc / me / srb** 各写一份；同步 `live/highcap/config/strategies/*/archetypes/regime.yaml`。

若规则是顶层单条 `(feature, operator)`，可用脚本：

```bash
python scripts/regime_threshold_calibrate.py \
  --strategies tpc --labeled-parquet "$PARQ" \
  --feature <feat> --operator "<=" --dry-run
# 人审 proposal 后 --apply
```

### 2. 刷 watchdog baseline JSON

在 **同一份 `$PARQ`** 上跑 watchdog，从报告抄数进 `config/monitoring/regime_watchdog_baseline.json`：

```bash
python scripts/regime_watchdog.py \
  --strategies tpc \
  --window-parquet "$PARQ" \
  --out-dir results/monitoring/baseline_refresh
# report.json → bull_share、trigger_rates → JSON 的 "<slug>" 块
```

### 3. PSI 参考 parquet（进 git，相对路径）

```bash
mkdir -p config/monitoring/reference
python - <<'PY'
import os, pandas as pd
from pathlib import Path
df = pd.read_parquet(os.environ["PARQ"])
cols = ["timestamp", "ema_1200_position", "vol_persistence", "vol_leverage_asymmetry"]
use = [c for c in cols if c in df.columns]
out = Path("config/monitoring/reference/tpc_psi_ref.parquet")
df[use].to_parquet(out, index=False)
print("wrote", out, len(df), "rows", use)
PY
```

更新 `config/monitoring/factor_ic_baseline_<slug>_<date>.json`：

- `"source_parquet": "config/monitoring/reference/tpc_psi_ref.parquet"`（**勿用绝对路径**）
- `regime_watchdog_baseline.json` 里设 `"factor_ic_baseline_ref": "config/monitoring/..."`

IC sign-flip 对照表：沿用或刷新 `factor_ic_baseline_*.json` 的 `rows[]`（本地 `$PARQ` 有 `forward_rr` 时可算；**周跑 bus 无 label，远程 IC 自动 skip**）。

### 4. 本地 smoke（current = 标定窗，验证 bundle 可跑通）

```bash
mlbot monitor run --config config/monitoring/weekly_rule_stack.yaml \
  --run-ts local_smoke --dry-run

# 真跑（用标定窗 parquet 代替 export-window 产物，仅本地调试）
export WATCHDOG_PARQUET="$PARQ"
mlbot monitor watchdog --window-parquet "$PARQ"
mlbot monitor drift --window-parquet "$PARQ"
# drift 不应全员 NO_PLATEAUS；watchdog 不应因缺 forward_rr 误报（bus 场景才 skip IC）
```

### 5. 提交

```bash
git add config/strategies config/monitoring live/highcap docs/decisions
git commit -m "chore(monitor): Tier-0 plateaus + PSI ref + watchdog baseline for <slug>"
git push
```

远程之后 `git pull` + enable timer 即可；**不必**把 `results/train_final` 整包上传。

---

## 远程自动跑（promote + bundle push 之后）

**不要**手敲 `mlbot monitor run …`。用 **独立调度进程**（不要塞进 feature-bus publisher）：

```bash
mlbot monitor schedule --cadence weekly   # 或 monthly | quarterly | yearly
mlbot monitor schedule --list
```

systemd（远程）：

```bash
sudo systemctl enable --now mlbot-monitor-daily.timer
sudo systemctl enable --now mlbot-monitor-weekly.timer
sudo systemctl enable --now mlbot-monitor-monthly.timer
# quarterly / yearly 同理
```

CMS 页面：**/monitoring**（漂移监控卡片：绿/红/橙缺勤）。

Telegram：

- 每次 `schedule` 若 **exit≠0** 或 **watchdog/drift ALERT** → 自动发（`MLBOT_MONITOR_SKIP_TG=1` 可关）
- **daily** 跑完后检查各 cadence 是否缺勤 → 缺勤发 TG（`mlbot monitor check-staleness`）

```bash
# 仅缺勤检查
mlbot monitor check-staleness
```

每次跑完会写：

| 产物 | 路径 | CMS |
|------|------|-----|
| 汇总索引 | `results/monitoring/index.json` | `GET /api/monitoring/index` |
| 分 cadence 快照 | `results/monitoring/latest_<cadence>.json` | 同上 |
| 明细报告 | `…/watchdog/*/report.json`, `…/drift/*/drift_report.json` | 下钻路径在 index 里 |
| SQLite 索引 | `results/rd_registry.sqlite` → `monitor_event` | `GET /api/monitoring/events` |

本地与远程 **同路径、同 schema**，可直接 diff `index.json` 或拷贝 `rd_registry.sqlite` 对比。

## 远程数据原则（bus-only）

| 不做 | 做 |
|------|-----|
| `train_final` 全历史当 current | Publisher → **feature bus** 滚动 parquet |
| monitor 时 `prepare-only` | `export-window` 从 bus 切片 |
| 手動 cron `monitor run` | `monitor schedule --cadence …` |

## 上线前检查（避免假绿 / 假红）

| 检查项 | 期望 |
|--------|------|
| Feature bus | `…/features/120T/*.parquet` 有数据且深度足够 |
| Regime plateau | 各策略 `last_calibration.plateaus` 非空；否则 drift **NO_PLATEAUS**（不发 TG，CMS 标未校准） |
| IC 漂移 | live bus 无 `forward_rr` → 自动 skip，不误报 ALERT |
| PSI 参考 | 远程需能读 IC baseline 的 `source_parquet`；否则 PSI skip（不自比 window） |
| Watchdog 调参 | manifest `watchdog_defaults` 或步骤内 `watchdog:` 覆盖 `psi_tol` 等 |

## Cadence → manifest

见 [`schedules.yaml`](schedules.yaml)：

| Cadence | Manifest | 内容 |
|---------|----------|------|
| weekly | `weekly_rule_stack.yaml` | 7d + 全 bus；watchdog + drift |
| monthly | `monthly_drift.yaml` | 30d bus；drift |
| quarterly | `quarterly_drift.yaml` | 30d + 全 bus；watchdog + drift |
| yearly | `yearly_drift.yaml` | 全 bus；drift 归档 |

## Run（调试）

```bash
mlbot monitor schedule --cadence weekly --dry-run
mlbot monitor run --config config/monitoring/weekly_rule_stack.yaml   # 仅调试
```

## Environment

| Variable | Default |
|----------|---------|
| `MLBOT_FEATURE_BUS_ROOT` | `live/shared_feature_bus` |
| `MLBOT_WATCHDOG_LOOKBACK_DAYS` | `7` |
| `MLBOT_DRIFT_LOOKBACK_DAYS` | `0`（全 bus） |
| `MLBOT_RD_REGISTRY_DB` | `results/rd_registry.sqlite` |
| `MLBOT_MONITOR_FORCE_SUBPROCESS` | unset（默认进程内直调；设 1 强制走 subprocess 旧路径，作为紧急回退） |

Legacy: [`mlbot-weekly-watchdog.service`](../../etc/systemd/mlbot-weekly-watchdog.service) 已改为调用 `schedule --cadence weekly`。
