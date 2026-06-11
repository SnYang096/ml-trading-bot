# Monitoring manifests

Authoritative guide: [`docs/strategy/漂移监控_mlbot_monitor_CN.md`](../../docs/strategy/漂移监控_mlbot_monitor_CN.md).

B / C 统一 cadence 与 regime schema 设计：[`docs/strategy/漂移监控_BC统一设计_CN.md`](../../docs/strategy/漂移监控_BC统一设计_CN.md)（C 系示例 manifest：`weekly_c_regime.yaml.example`、`monthly_multileg_c.yaml.example`）。

Promote 因果门槛 + bundle checklist：[`config/experiments/LAYER_PROMOTION_CRITERIA.md`](../experiments/LAYER_PROMOTION_CRITERIA.md) §4.

## 发现 labeled parquet（`mlbot monitor catalog`）

本地无 feature bus 时，先用 catalog 挑 **`features_labeled.parquet`**，再设 `PARQ` 跑 watchdog/drift。

```bash
cd /path/to/ml_trading_bot

# 最近 30 个（默认搜 results/）
mlbot monitor catalog

# 按策略过滤
mlbot monitor catalog --strategy tpc --limit 10
mlbot monitor catalog --strategy bpc --limit 10

# 单个文件：完整 JSON（rows / symbols / 日期 / monitor 列）
mlbot monitor catalog \
  --path results/monitoring/tier0/tpc_20260603/features_labeled_recent_6m_oos.parquet \
  --json

# 扩大搜索范围或条数
mlbot monitor catalog --root results --limit 50
```

**表格列**：`mtime` · `rows` · `symbols` · `time_start/end` · `fwd_rr` · `ema1200` · `path`

**monitor 选型要点**（不必是最新 mtime）：

| 优先 | 条件 |
|------|------|
| 1 | 路径在 **active 实验 yaml / PHASE1_REPORT** 里被引用 |
| 2 | **`results/monitoring/tier0/`** — 与 git baseline 同窗（`recent_6m_oos`） |
| 3 | **`train_final_*_rd_rerun`** — 当前 rd_loop 主输入 |
| 4 | `time_start/end` 对齐 `mlbot monitor segments` 里的标定窗 |

```bash
mlbot monitor segments    # 看 recent_6m_oos 等日期
export PARQ=<catalog 里选中的 path>
mlbot monitor watchdog --window-parquet "$PARQ" --strategies tpc
mlbot monitor drift --window-parquet "$PARQ" --strategies tpc
```

### 哪些该留、哪些可删

全库约有 **100+** 个 `features_labeled.parquet`；**mtime 新 ≠ monitor 该用**。删前用 `rg features_labeled` 在 `config/experiments/` 确认无引用。

**建议保留（果实 / 仍有用）**

| 路径模式 | 原因 |
|----------|------|
| `results/monitoring/tier0/**` | Tier-0 标定窗；与 `regime_watchdog_baseline.json` / PSI ref 对齐 |
| `config/monitoring/reference/*.parquet` | 已在 git 的 PSI reference |
| `results/train_final/tpc/train_final_20260604_rd_rerun/` | 当前 TPC rd_loop / 实验引用 |
| `results/train_final/bpc/train_final_20260604_rd_rerun/` | BPC prod L=20 |
| `results/train_final/bpc/bpc_lb{120,240}_train_final_20260604_rd_rerun/` | BPC lookback 实验仍在用 |
| `results/validation_smoke/<slug>/` | 体量小；快速 smoke |
| 任意被 **`config/experiments/**/rd_loop*.yaml`** 或 **`DECISION.md`** 写死的 parquet | 删了实验复现会断 |

**一般可删（确认无引用后）**

| 路径模式 | 原因 |
|----------|------|
| `results/*/validate_static.constrained/**` | **ROUTINE_R&D_DEPRECATED**；同内容多次 timestamp 重复 |
| `results/*/validate_static.full_study/**` | 同上 |
| `results/train_final/<slug>/train_final_202605{21,22,23}_*/` | 已被 `20260604_rd_rerun` 取代的旧 prepare |
| `results/train_final/fast_scalp/prepare_*`、`gate_features*` | 树实验中间产物；DECISION 归档后可清 |
| `results/tpc/calibrate_roll.default/`（**~245GB** 整树） | 历史 calibrate_roll 输出；若不再跑该 yaml，可整目录删 |
| `results/<slug>/*/20*/results/features_labeled.parquet` | 旧 pipeline 单次 run 目录（非 train_final 主链） |

**删目录示例（先 dry-run 列表，勿盲目 rm -rf）**

```bash
# 列出 deprecated validate_static 下的 labeled parquet（不删除）
find results -path '*/validate_static.constrained/*/features_labeled.parquet' 2>/dev/null

# 列出 202605 旧 train_final（对照 catalog 后再删）
find results/train_final -path '*/train_final_202605*/**/features_labeled.parquet' 2>/dev/null

# 确认实验 yaml 无引用后再删整目录，例如：
# rm -rf results/tpc/validate_static.constrained
# rm -rf results/tpc/calibrate_roll.default   # 仅当确定不再做 calibrate_roll
```

**勿删**：`results/monitoring/`（除明确过期的 `window/` 调试产物）、git 已跟踪的 `config/monitoring/reference/`、仍在 promote 流程中的 Tier-0 切片。

### Manifest 窗口名（`near` / `deep`）

YAML 里 **`near` / `deep` 表示时间窗深浅**，与交易多空 **无关**（旧名 `short` / `long` 仍兼容，但易误解）。

| 键 | 含义 | weekly 示例 |
|----|------|-------------|
| **`near`** | 近端窗 → **watchdog**（gate / PSI / bull_share） | `lookback_days: 7` |
| **`deep`** | 较深窗 → **drift**（regime plateau） | `lookback_days: 0`（bus 全深度） |

## current vs reference（数据从哪来）

| 角色 | 谁生成 | 放哪 | 用途 |
|------|--------|------|------|
| **reference / baseline** | promote 或 Tier-0 标定（本地一次） | **git**：`regime_watchdog_baseline.json`、`regime.yaml` plateaus、PSI ref parquet | watchdog / drift **对照表** |
| **current（远程）** | `export-window` 从 **feature bus** 拼 near/deep | `results/monitoring/window/<ts>/` | 周 cron **告警权威**；不重算特征 |
| **local replay** | 已有或新跑 `features_labeled.parquet` | `results/...` | 本地 smoke / IC 复核；**不**替代远程日常告警 |

- **`mlbot monitor export-window`**：读 bus 里**已算好的特征**，按 `lookback_days` 切片，写出监控 parquet。**不是**重新跑 R&D pipeline。
- **`watchdog` / `drift`**：用 current parquet 与 git reference 对比；**不会改** `archetypes/*.yaml`。

### Symbol 列表（universe，不要手写 HIGHCAP）

实盘 highcap 币 list 真值：[`live/highcap/universe.yaml`](../../live/highcap/universe.yaml)（与 `start_live.sh` / multileg 同源）。

- **`export-window`**：manifest / CLI 未写 `symbols` 时 → bus 目录里有的币；bus 空则脚本内硬编码回退（**与 universe 未自动接线**，见 backlog）。
- **不要**在 shell 里再维护一套 `HIGHCAP=...` 变量；应对齐 universe keys，或从 universe 复制一次到 `--symbols`。
- **baseline 与 current 的 symbol 集合必须一致**（例如 baseline 仅 BTC 时，不要用全币 current 去比）。

### PSI 监控列（契约，非 features.yaml 全量）

默认 3 列在 [`weekly_rule_stack.yaml`](weekly_rule_stack.yaml) `watchdog_defaults.psi_features` 与 `regime_watchdog.py` CLI default：

`ema_1200_position`, `vol_persistence`, `vol_leverage_asymmetry`

- **设计**：PSI 只盯 **生产 gate/prefilter 契约列**（见 [`LAYER_PROMOTION_CRITERIA.md`](../experiments/LAYER_PROMOTION_CRITERIA.md) §4），**不是** `features.yaml` 全量。
- **改列**：编辑 manifest 的 `psi_features`（或 per-step `watchdog.psi_features`），promote 时与 `gate.yaml` 人工对齐。
- **drift plateau 列**：来自各策略 `regime.yaml` → `last_calibration.plateaus`（与 PSI 默认 3 列独立）。
- **backlog**：从 `gate.yaml` 自动推导 PSI 列（代码已有 `_extract_features_from_gate`，尚未接入 watchdog）。

## 本地 highcap 监控（决策）

```text
有 feature bus?
  ├─ 是 → export-window near(7d) + deep(0d)；symbols 与 universe.yaml 一致
  │       → mlbot monitor run weekly_rule_stack（或单独 watchdog/drift）
  └─ 否 → 已有 features_labeled.parquet?
            ├─ 是 → 设 WATCHDOG_PARQUET / DRIFT_PARQUET，跑 watchdog + drift（smoke）
            └─ 否 → train_strategy_pipeline --prepare-only（见 §0 B）
baseline 与 current 同 symbol 集合?
  ├─ 否 → 先 Tier-0 重刷 baseline，再解读 alert
  └─ 是 → 解读 OK / ALERT
```

有 bus 时示例（symbol 与 universe 一致，勿另建 HIGHCAP 变量名）：

```bash
# 从 universe.yaml 取 keys（示例）
SYMS=$(python - <<'PY'
import yaml
from pathlib import Path
u = yaml.safe_load(Path("live/highcap/universe.yaml").read_text())
print(",".join(sorted(u.get("symbols") or {})))
PY
)

mlbot monitor export-window --lookback-days 7 --symbols "$SYMS" \
  --output /tmp/features_near_7d.parquet
mlbot monitor export-window --lookback-days 0 --symbols "$SYMS" \
  --output /tmp/features_deep.parquet

export WATCHDOG_PARQUET=/tmp/features_near_7d.parquet
export DRIFT_PARQUET=/tmp/features_deep.parquet
mlbot monitor watchdog --window-parquet "$WATCHDOG_PARQUET" --strategies bpc,tpc,me,srb
mlbot monitor drift --window-parquet "$DRIFT_PARQUET" --strategies bpc,tpc,me,srb
```

## 本地监控 bundle（promote 后 · 先做这套）

**在本地完成 reference 标定并 git push**；远程 cron 只消费 git 里的 baseline/plateaus + 自产 bus current。标定窗建议 **`recent_6m_oos`**（见 [`config/market_segment.yaml`](../market_segment.yaml)）。

### 0. 数据 + 标定 parquet

**优先复用已有 R&D 产物**，不必每次重跑 prepare。

#### A. 选已有 `features_labeled.parquet`

见上文 **[发现 labeled parquet（catalog）](#发现-labeled-parquetmlbot-monitor-catalog)**。快速命令：

```bash
mlbot monitor catalog --strategy tpc --limit 10
mlbot monitor catalog --path "$PARQ" --json   # 选定后核对元信息
```

选定前核对（**symbol 集合须与 baseline 一致**）：

| 检查项 | 如何看 |
|--------|--------|
| symbol 集合 | `python -c "import pandas as pd; df=pd.read_parquet('PARQ'); print(df['symbol'].unique() if 'symbol' in df.columns else 'no symbol col')"` |
| 日期窗 | `timestamp` 的 min/max |
| 有 `forward_rr` | 本地 IC 才能算；bus current 无此列 |
| segment 意图 | 应对齐 `recent_6m_oos`（`mlbot monitor segments` 看日期） |

```bash
export PARQ=/path/to/chosen/features_labeled.parquet
echo "PARQ=$PARQ"
```

#### B. 必须新跑（highcap 全币 + recent_6m_oos）

**注意**：`mlbot train final` **没有** `--segment`；须用 **`--start-date` / `--end-date`**，或直接调 `train_strategy_pipeline.py`（与 [`archive_batch_window.py`](../../scripts/monitoring/archive_batch_window.py) 同源）。

日期：`mlbot monitor segments` → `recent_6m_oos` 现为 `2025-10-01` → `2026-03-31`。

```bash
# 若 archive 未齐，先拉数据（symbol 与 universe.yaml 一致）
mlbot data download --symbols BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT --start-year 2025 --start-month 1
mlbot data convert

# 多币 pooled prepare（TPC 示例；bpc/me/srb 换 --config）
python scripts/train_strategy_pipeline.py --prepare-only \
  --config config/strategies/tpc \
  --symbol BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT \
  --timeframe 120T \
  --start-date 2025-10-01 \
  --end-date 2026-03-31 \
  --train-all \
  --output-root results/monitoring/tier0/tpc_highcap_$(date +%Y%m%d)

export PARQ=$(ls -t results/monitoring/tier0/tpc_highcap_*/tpc/features_labeled.parquet | head -1)
echo "PARQ=$PARQ"
```

单币、与现 BTC-only baseline 对齐时，可用：

```bash
mlbot monitor archive-batch --strategy tpc --segment recent_6m_oos \
  --symbol BTCUSDT --output /tmp/tpc_6m.parquet
export PARQ=/tmp/tpc_6m.parquet
```

等价 `mlbot train final`（须显式日期）：

```bash
mlbot train final --prepare-only \
  --config config/strategies/tpc \
  --symbol BTCUSDT \
  --timeframe 120T \
  --start-date 2025-10-01 \
  --end-date 2026-03-31
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

systemd（远程 CI deploy 会自动 enable timer；手动调试）：

```bash
sudo systemctl enable --now mlbot-monitor-daily.timer
sudo systemctl enable --now mlbot-monitor-weekly.timer
sudo systemctl enable --now mlbot-monitor-monthly.timer
# quarterly / yearly 同理
# 单次 smoke（写 index.json，CMS 需 compose 挂载 results/）:
/opt/quant-engine/deploy/systemd/mlbot-monitor-docker-run.sh weekly
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
| weekly | `weekly_rule_stack.yaml` | near 7d + deep 全 bus；watchdog + drift |
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

## Backlog（代码未做，文档先约定）

| 项 | 说明 |
|----|------|
| ~~`mlbot monitor catalog`~~ | **已实现** — 见 README [catalog 小节](#发现-labeled-parquetmlbot-monitor-catalog) |
| `mlbot monitor prepare-window` | `--universe highcap --segment recent_6m_oos` 封装 §0 B + universe.yaml |
| `export-window` 读 universe | manifest `universe: highcap` 自动解析 `live/highcap/universe.yaml` |
| PSI 从 gate 推导 | 复用 `live_feature_plan._extract_features_from_gate`，manifest 仍可 override |
