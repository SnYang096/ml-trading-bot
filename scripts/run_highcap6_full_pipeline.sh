#!/bin/bash
# HighCap6完整Pipeline流程脚本
# 确保包含所有6个token（BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT, XRPUSDT, ADAUSDT）

set -e

# 配置
TASK_SPEC="config/tasks/task_spec_highcap6_2024_202510.yaml"
SYMBOLS="BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT"
TIMEFRAME="240T"
START_DATE="2024-01-01"
END_DATE="2024-12-31"
FEATURE_STORE_LAYER="nnmh_highcap6_240T_2024_with_reflexivity"
FEATURE_STORE_ROOT="feature_store"
RUN_ID="pipeline_highcap6_2024_full_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="results/${RUN_ID}"

# 模型路径（需要根据实际情况修改）
MODEL_PATH="${MODEL_PATH:-results/nnmultihead/.../model.pt}"

echo "=========================================="
echo "HighCap6完整Pipeline流程"
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
    echo "步骤1: 构建FeatureStore..."
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

# 步骤3: 构建Execution日志
echo ""
echo "步骤3: 构建Execution日志..."
mlbot nnmultihead build-execution-logs \
    --preds "${OUTPUT_DIR}/preds" \
    --model "${MODEL_PATH}" \
    --symbols "${SYMBOLS}" \
    --timeframe "${TIMEFRAME}" \
    --start-date "${START_DATE}" \
    --end-date "${END_DATE}" \
    --data-path data/parquet_data \
    --returns-source rr_execution \
    --output "${OUTPUT_DIR}/logs_execution.parquet" \
    --no-docker 2>&1 | tee "${OUTPUT_DIR}/build_logs.log"
echo "✅ Execution日志构建完成"

# 步骤4: 应用Gate过滤（全松阈值，regime 已内嵌）
echo ""
echo "步骤4: 应用Gate过滤（全松阈值）..."
mlbot rule apply-tree-gate \
    --logs "${OUTPUT_DIR}/logs_execution.parquet" \
    --out "${OUTPUT_DIR}/logs_execution_gated.parquet" \
    --features-store-layer "${FEATURE_STORE_LAYER}" \
    --features-store-root "${FEATURE_STORE_ROOT}" \
    --live-config config/nnmultihead/live/meta_router_live_config.yaml \
    --timeframe "${TIMEFRAME}" \
    --start-date "${START_DATE}" \
    --end-date "${END_DATE}" \
    --no-docker 2>&1 | tee "${OUTPUT_DIR}/gate.log"
echo "✅ Gate过滤完成"

# 步骤5: 构建Stage Logs
echo ""
echo "步骤5: 构建Stage Logs..."
PYTHONPATH=. python3 scripts/build_execution_log_stages.py \
    --preds "${OUTPUT_DIR}/preds" \
    --logs "${OUTPUT_DIR}/logs_execution.parquet" \
    --gated-logs "${OUTPUT_DIR}/logs_execution_gated.parquet" \
    --out-dir "${OUTPUT_DIR}/exec_logs" \
    --timeframe "${TIMEFRAME}" \
    --run-id "${RUN_ID}" \
    --strategy-name "pipeline-3action-e2e" \
    2>&1 | tee "${OUTPUT_DIR}/build_stages.log"
echo "✅ Stage Logs构建完成"

# 步骤6: 聚合Canonical Log
echo ""
echo "步骤6: 聚合Canonical Log..."
PYTHONPATH=. python3 scripts/aggregate_execution_log_stages.py \
    --stage-dir "${OUTPUT_DIR}/exec_logs" \
    --out "${OUTPUT_DIR}/execution_log.jsonl" \
    2>&1 | tee "${OUTPUT_DIR}/aggregate.log"
echo "✅ Canonical Log聚合完成"

# 步骤7: 生成E2E KPI报告
echo ""
echo "步骤7: 生成E2E KPI报告..."
python3 scripts/diagnose_e2e_kpi.py \
    --logs "${OUTPUT_DIR}/logs_execution_gated.parquet" \
    --gate "${OUTPUT_DIR}/logs_execution_gated.parquet" \
    --output-md "${OUTPUT_DIR}/e2e_kpi_report.md" \
    --output-json "${OUTPUT_DIR}/e2e_kpi_report.json" \
    --no-regime-filter \
    2>&1 | tee "${OUTPUT_DIR}/e2e_kpi.log"
echo "✅ E2E KPI报告生成完成"

echo ""
echo "=========================================="
echo "✅ 完整Pipeline流程完成！"
echo "输出目录: ${OUTPUT_DIR}"
echo "=========================================="
