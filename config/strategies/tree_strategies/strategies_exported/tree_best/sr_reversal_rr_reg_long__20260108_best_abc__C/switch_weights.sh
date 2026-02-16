#!/bin/bash
# 快速切换权重配置的脚本

CONFIG_DIR="$(dirname "$0")"
LABELS_YAML="$CONFIG_DIR/labels.yaml"
LABELS_BACKUP="$CONFIG_DIR/labels.yaml.backup"
LABELS_WEIGHTS="$CONFIG_DIR/labels_with_weights.yaml"

# 检查文件是否存在
if [ ! -f "$LABELS_WEIGHTS" ]; then
    echo "❌ 错误: $LABELS_WEIGHTS 不存在"
    exit 1
fi

# 检查当前状态
if [ -f "$LABELS_BACKUP" ]; then
    # 当前使用的是带权重版本，切换到无权重版本
    if [ -f "$LABELS_YAML" ]; then
        # 保存当前带权重版本
        cp "$LABELS_YAML" "$CONFIG_DIR/labels.yaml.with_weights" 2>/dev/null || true
    fi
    cp "$LABELS_BACKUP" "$LABELS_YAML"
    echo "✅ 已切换到无权重版本 (使用 labels.yaml.backup)"
    echo "   当前配置: compute_weights = false"
else
    # 当前使用的是无权重版本，切换到带权重版本
    if [ -f "$LABELS_YAML" ]; then
        # 备份原配置
        cp "$LABELS_YAML" "$LABELS_BACKUP"
    fi
    cp "$LABELS_WEIGHTS" "$LABELS_YAML"
    echo "✅ 已切换到带权重版本 (使用 labels_with_weights.yaml)"
    echo "   当前配置: compute_weights = true, weight_strategy = result_based_rr"
fi

