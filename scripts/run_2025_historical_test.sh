#!/bin/bash
# 2025历史实验复现脚本（Sharpe 1.759）
# 数据范围: 2025-05-01 到 2025-10-31
# Symbols: BTCUSDT, ETHUSDT, XRPUSDT (较完整的数据)

set -e

# 配置
TASK_SPEC="config/tasks/task_spec_highcap6_2024_202510.yaml"
SYMBOLS="BTCUSDT,ETHUSDT,XRPUSDT"
TIMEFRAME="240T"
START_DATE="2025-05-01"
END_DATE="2025-10-31"
FEATURE_STORE_LAYER="nnmh_highcap6_240T_2025_historical"
FEATURE_STORE_ROOT="feature_store"
RUN_ID="pipeline_2025_historical_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="results/${RUN_ID}"

# 模型路径（使用现有的模型）
MODEL_PATH="${MODEL_PATH:-results/nnmultihead/nnmh_highcap6_2024_202510/nnmh_config_train_multi_240T/model.pt}"

echo "=========================================="
echo "2025历史实验复现 (Sharpe 1.759)"
echo "=========================================="
echo "TaskSpec: ${TASK_SPEC}"
echo "Symbols: ${SYMBOLS}"
echo "Timeframe: ${TIMEFRAME}"
echo "Date Range: ${START_DATE} to ${END_DATE}"
echo "Run ID: ${RUN_ID}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "=========================================="

# 创建输出目录
mkdir -p "${OUTPUT_DIR}"

# 步骤1: FeatureStore构建（如果需要）
if [ "${SKIP_FEATURESTORE}" != "true" ]; then
    echo ""
    echo "步骤1: 构建FeatureStore (2025数据)..."
    mlbot nnmultihead build-feature-store \
        --task-spec "${TASK_SPEC}" \
        --symbols "${SYMBOLS}" \
        --timeframe "${TIMEFRAME}" \
        --start-date "${START_DATE}" \
        --end-date "${END_DATE}" \
        --feature-store-root "${FEATURE_STORE_ROOT}" \
        --layer "${FEATURE_STORE_LAYER}" \
        --warmup-months 1 \
        --no-docker 2>&1 | tee "${OUTPUT_DIR}/featurestore_build.log"
    echo "✅ FeatureStore构建完成"
fi

# 步骤2: 模型预测
echo ""
echo "步骤2: 生成预测..."
mlbot nnmultihead predict \
    --task-spec "${TASK_SPEC}" \
    --symbols "${SYMBOLS}" \
    --timeframe "${TIMEFRAME}" \
    --start-date "${START_DATE}" \
    --end-date "${END_DATE}" \
    --model "${MODEL_PATH}" \
    --feature-store-layer "${FEATURE_STORE_LAYER}" \
    --feature-store-root "${FEATURE_STORE_ROOT}" \
    --output "${OUTPUT_DIR}/preds" \
    --no-docker 2>&1 | tee "${OUTPUT_DIR}/predict.log"
echo "✅ 预测完成"

# 步骤3: 构建执行日志
echo ""
echo "步骤3: 构建执行日志..."
mlbot nnmultihead build-execution-logs \
    --task-spec "${TASK_SPEC}" \
    --symbols "${SYMBOLS}" \
    --timeframe "${TIMEFRAME}" \
    --start-date "${START_DATE}" \
    --end-date "${END_DATE}" \
    --preds-dir "${OUTPUT_DIR}/preds" \
    --feature-store-layer "${FEATURE_STORE_LAYER}" \
    --feature-store-root "${FEATURE_STORE_ROOT}" \
    --output "${OUTPUT_DIR}/logs_execution.parquet" \
    --no-docker 2>&1 | tee "${OUTPUT_DIR}/build_execution_logs.log"
echo "✅ 执行日志构建完成"

# 步骤4: 生成evidence quantiles
echo ""
echo "步骤4: 生成evidence quantiles..."
python3 scripts/build_evidence_quantiles.py \
    --logs "${OUTPUT_DIR}/logs_execution.parquet" \
    --output "${OUTPUT_DIR}/evidence_quantiles.json" \
    2>&1 | tee "${OUTPUT_DIR}/build_evidence_quantiles.log"
echo "✅ Evidence quantiles生成完成"

# 步骤5: 应用FR gate (使用历史MEAN_REGIME条件)
echo ""
echo "步骤5: 应用FR gate (历史MEAN_REGIME条件)..."
python3 scripts/apply_archetype_gate.py \
    --logs "${OUTPUT_DIR}/logs_execution.parquet" \
    --out "${OUTPUT_DIR}/logs_execution_gated.parquet" \
    --features-store-root "${FEATURE_STORE_ROOT}" \
    --features-store-layer "${FEATURE_STORE_LAYER}" \
    --timeframe "${TIMEFRAME}" \
    --start-date "${START_DATE}" \
    --end-date "${END_DATE}" \
    --evidence-quantiles "${OUTPUT_DIR}/evidence_quantiles.json" \
    --archetype-filter FR \
    2>&1 | tee "${OUTPUT_DIR}/apply_gate.log"
echo "✅ Gate应用完成"

# 步骤6: 生成E2E KPI报告
echo ""
echo "步骤6: 生成E2E KPI报告..."
python3 scripts/diagnose_e2e_kpi.py \
    --logs "${OUTPUT_DIR}/logs_execution_gated.parquet" \
    --gate "${OUTPUT_DIR}/logs_execution_gated.parquet" \
    --output-json "${OUTPUT_DIR}/e2e_kpi_report.json" \
    --output-md "${OUTPUT_DIR}/e2e_kpi_report.md" \
    2>&1 | tee "${OUTPUT_DIR}/diagnose_e2e_kpi.log"
echo "✅ E2E KPI报告生成完成"

echo ""
echo "=========================================="
echo "✅ 2025历史实验复现完成"
echo "=========================================="
echo "输出目录: ${OUTPUT_DIR}"
echo "E2E KPI报告: ${OUTPUT_DIR}/e2e_kpi_report.md"
echo ""
echo "预期结果（历史实验）:"
echo "  - Sample count: 27"
echo "  - ret_mean: 0.001384"
echo "  - Win rate: 44.4%"
echo "  - Sharpe: 1.759"
echo "=========================================="
