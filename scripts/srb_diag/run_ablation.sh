#!/usr/bin/env bash
# SRB one-knob ablation runner.
#
# Baselines `execution.yaml` to the exact content used in 20260417_163432
# (= current working tree minus: sr_feature_injection / expand_with_primary_atr /
# widened activation-trail / low_adx_high_er), then for each experiment applies
# a single delta, runs slow rolling_sim, and snapshots the result folder.
#
# Usage:
#   bash scripts/srb_diag/run_ablation.sh exp1      # wide SR injection only
#   bash scripts/srb_diag/run_ablation.sh exp2      # adaptive ATR trailing only
#   bash scripts/srb_diag/run_ablation.sh exp3      # wider activation/trail only
#   bash scripts/srb_diag/run_ablation.sh exp4      # add_position bucket only
#   bash scripts/srb_diag/run_ablation.sh restore   # restore HEAD working-tree yaml
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

YAML="config/strategies/srb/archetypes/execution.yaml"
BACKUP="/tmp/srb_execution_worktree.yaml"
BASELINE="/tmp/srb_execution_baseline.yaml"   # matches 20260417_163432

# 1) Back up current work-tree yaml if not done yet
if [[ ! -f "$BACKUP" ]]; then
  cp "$YAML" "$BACKUP"
  echo "[ablation] backed up work-tree yaml -> $BACKUP"
fi

# 2) Build the "baseline" yaml that exactly matches run 20260417_163432
python3 - <<'PY'
import re
src = open('/tmp/srb_execution_worktree.yaml').read()

# drop sr_feature_injection block (two lines + header comment + blank)
src = re.sub(
    r"\n# 更长窗口 swing 极值[^\n]*\nsr_feature_injection:\n  swing_lookback_wide_bars:[^\n]*\n",
    "\n",
    src,
)

# drop expand_with_primary_atr + comment
src = re.sub(
    r"    # 波动扩张后仍用入场 ATR[^\n]*\n    expand_with_primary_atr:[^\n]*\n",
    "",
    src,
)

# revert activation_r 7.0 -> 6.0, trail_r 6.0 -> 5.0
src = src.replace("    activation_r: 7.0", "    activation_r: 6.0")
src = src.replace("    trail_r: 6.0", "    trail_r: 5.0")

# drop low_adx_high_er from allow_regime_buckets
src = re.sub(
    r"    - low_adx_high_er[^\n]*\n",
    "",
    src,
)

open('/tmp/srb_execution_baseline.yaml', 'w').write(src)
print("[ablation] wrote /tmp/srb_execution_baseline.yaml")
PY

mode="${1:-}"

apply_baseline() { cp "$BASELINE" "$YAML"; }

case "$mode" in
  restore)
    cp "$BACKUP" "$YAML"
    echo "[ablation] restored work-tree yaml"
    exit 0
    ;;
  exp1)
    apply_baseline
    python3 - <<'PY'
p='config/strategies/srb/archetypes/execution.yaml'
src=open(p).read()
inj = ("\n# 宽窗 SR (wide swing) — 只注入为特征,本轮不进决策链 (ablation exp1)\n"
       "sr_feature_injection:\n  swing_lookback_wide_bars: 96\n")
src = src.replace("add_position:", inj + "add_position:", 1)
open(p,'w').write(src)
print("[exp1] patched: wide SR injection only")
PY
    tag="exp1_wide_sr_only"
    ;;
  exp2)
    apply_baseline
    python3 - <<'PY'
p='config/strategies/srb/archetypes/execution.yaml'
src=open(p).read()
src = src.replace(
    "  trailing:\n    enabled: true\n    activation_r: 6.0\n    trail_r: 5.0\n",
    "  trailing:\n    enabled: true\n    expand_with_primary_atr: true\n    activation_r: 6.0\n    trail_r: 5.0\n",
)
open(p,'w').write(src)
print("[exp2] patched: expand_with_primary_atr only")
PY
    tag="exp2_adaptive_atr_only"
    ;;
  exp3)
    apply_baseline
    python3 - <<'PY'
p='config/strategies/srb/archetypes/execution.yaml'
src=open(p).read()
src = src.replace("    activation_r: 6.0", "    activation_r: 7.0")
src = src.replace("    trail_r: 5.0", "    trail_r: 6.0")
open(p,'w').write(src)
print("[exp3] patched: wider defaults only (7/6)")
PY
    tag="exp3_wider_defaults_only"
    ;;
  exp4)
    apply_baseline
    python3 - <<'PY'
p='config/strategies/srb/archetypes/execution.yaml'
src=open(p).read()
src = src.replace(
    "    - high_adx_high_er\n",
    "    - high_adx_high_er\n    - low_adx_high_er\n",
    1,
)
open(p,'w').write(src)
print("[exp4] patched: + low_adx_high_er bucket")
PY
    tag="exp4_add_low_adx_high_er"
    ;;
  *)
    echo "usage: $0 {exp1|exp2|exp3|exp4|restore}"
    exit 1
    ;;
esac

# 3) Sanity-print the diff vs baseline
echo "---- effective yaml delta vs baseline ----"
diff -u "$BASELINE" "$YAML" || true
echo "------------------------------------------"

# 4) Run the rolling sim
CFG="config/prod_train_pipeline_2h_slow_srb_only.yaml"
echo "[ablation] running: mlbot pipeline run --all --config $CFG --stage rolling_sim --skip-shap"
mlbot pipeline run --all --config "$CFG" --stage rolling_sim --skip-shap 2>&1 | tee "/tmp/srb_${tag}.log" | tail -40

# 5) Find the newest rolling run dir and copy/rename
NEW_DIR="$(ls -dt results/srb/slow-rolling-sim/_rolling_sim/*/ | head -1)"
OUT_DIR="results/srb/diag/ablation_20260418/${tag}"
mkdir -p "$(dirname "$OUT_DIR")"
cp -r "$NEW_DIR" "$OUT_DIR"
echo "[ablation] snapshot saved -> $OUT_DIR"

# 6) Restore work-tree yaml
cp "$BACKUP" "$YAML"
echo "[ablation] work-tree yaml restored"
