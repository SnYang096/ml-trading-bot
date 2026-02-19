# Pipeline 执行日志

> 执行时间: 2026-02-18
> Universe: highcap 6 symbols (BTC/ETH/BNB/SOL/XRP/ADA) — 与 live/highcap/universe.yaml 对齐
> 数据范围: 2023-01-01 ~ 2025-12-31, warmup 3 months

---

## Step 1a: ME 1H Feature Store

**命令:**
```bash
mlbot feature-store build \
  --config config/strategies/me \
  --universe-config config/download/crypto_4h_token_universe_groups.yaml \
  --universe-set starter_a --universe-groups highcap \
  --timeframe 60T \
  --start-date 2023-01-01 --end-date 2025-12-31 \
  --warmup-months 3 --no-docker
```

**状态:** 🔄 运行中 (PID 4296, 重启 23:22)
**日志:** /tmp/fs_me_1h.log
**结果:** (构建中)

---

## Step 1b: BPC/FER 4H Feature Store

**命令:**
```bash
mlbot feature-store build \
  --config config/strategies/bpc \
  --universe-config config/download/crypto_4h_token_universe_groups.yaml \
  --universe-set starter_a --universe-groups highcap \
  --timeframe 240T \
  --start-date 2023-01-01 --end-date 2025-12-31 \
  --warmup-months 3 --no-docker
```

**状态:** ✅ 完成 (缓存命中，216 months skipped)
**日志:** /tmp/fs_bpc_4h.log
**结果:** 6/6 symbols, 0 failed, meta=features_792208f36f

---

## Step 1c: LV 15min Feature Store

**命令:**
```bash
mlbot feature-store build \
  --config config/strategies/lv \
  --universe-config config/download/crypto_4h_token_universe_groups.yaml \
  --universe-set starter_a --universe-groups highcap \
  --timeframe 15T \
  --start-date 2023-01-01 --end-date 2025-12-31 \
  --warmup-months 3 --no-docker
```

**状态:** 🔄 运行中 (PID 4497, 启动 23:22)
**日志:** /tmp/fs_lv_15m.log
**结果:** (构建中)

---

## Step 2: BPC 训练 (4H)

**命令序列:**
```bash
# 2a. Train final
mlbot train final --strategy bpc --label-config labels_rr_extreme.yaml

# 2b. Apply archetype gate
python scripts/apply_archetype_gate.py --strategy bpc

# 2c. Optimize gate
python scripts/optimize_gate_unified.py --strategy bpc

# 2d. Optimize evidence
python scripts/optimize_evidence_plateau.py --logs <bpc_predictions> --strategy bpc
```

**状态:** ⏳ 待 Step 1b 完成
**结果:** (待完成)

---

## Step 3: ME 训练 (1H)

**命令序列:**
```bash
# 3a. Train final
mlbot train final --strategy me --timeframe 60T --label-config labels_rr_extreme.yaml

# 3b. Apply archetype gate
python scripts/apply_archetype_gate.py --strategy me

# 3c. Optimize gate
python scripts/optimize_gate_unified.py --strategy me

# 3d. Optimize evidence
python scripts/optimize_evidence_plateau.py --logs <me_predictions> --strategy me
```

**状态:** ⏳ 待 Step 1a 完成
**结果:** (待完成)

---

## Step 4: FER 训练 (4H)

**命令序列:**
```bash
# 4a. Train final
mlbot train final --strategy fer --label-config labels_rr_extreme.yaml

# 4b. Apply archetype gate
python scripts/apply_archetype_gate.py --strategy fer

# 4c. Optimize gate
python scripts/optimize_gate_unified.py --strategy fer

# 4d. Optimize evidence
python scripts/optimize_evidence_plateau.py --logs <fer_predictions> --strategy fer
```

**状态:** ⏳ 待 Step 1b 完成
**结果:** (待完成)

---

## Step 5: LV 训练 (15min)

**命令序列:**
```bash
# 5a. Train final
mlbot train final --strategy lv --timeframe 15T --label-config labels_rr_extreme.yaml

# 5b. Apply archetype gate
python scripts/apply_archetype_gate.py --strategy lv

# 5c. Optimize gate
python scripts/optimize_gate_unified.py --strategy lv

# 5d. Optimize evidence
python scripts/optimize_evidence_plateau.py --logs <lv_predictions> --strategy lv
```

**状态:** ⏳ 待 Step 1c 完成
**结果:** (待完成)

---

## Step 6: PCM 联合回测

**命令:**
```bash
python scripts/backtest_execution_layer.py \
  --pcm bpc:<bpc_predictions> me:<me_predictions> fer:<fer_predictions> lv:<lv_predictions>

python scripts/evaluate_pcm_allocation.py --pcm-report <pcm_report>
```

**状态:** ⏳ 待 Step 2-5 完成
**结果:** (待完成)

---

## Step 7: 同步到实盘 + 冒烟测试

**命令:**
```bash
# 同步配置
rsync -av config/strategies/ live/highcap/config/strategies/

# 冒烟测试
python scripts/run_live.py --testnet --smoke-test
```

**状态:** ⏳ 待 Step 6 完成
**结果:** (待完成)
