# 如何切换到带权重的配置

## 方案 1：直接替换（推荐，最简单）

直接替换 `labels.yaml` 的内容：

```bash
# 备份原配置
cp labels.yaml labels.yaml.backup

# 使用带权重的配置
cp labels_with_weights.yaml labels.yaml
```

**优点**：
- 最简单，无需修改代码
- 无需创建新策略目录

**缺点**：
- 需要手动切换
- 无法同时保留两个版本

## 方案 2：创建新策略目录（推荐用于对比测试）

如果你想同时保留两个版本进行对比，可以创建新策略目录：

```bash
# 复制整个策略目录
cp -r config/strategies/sr_reversal_rr_reg config/strategies/sr_reversal_rr_reg_weighted

# 在新目录中使用带权重的配置
cd config/strategies/sr_reversal_rr_reg_weighted
cp labels_with_weights.yaml labels.yaml
```

然后训练时指定新策略：
```bash
python scripts/train_strategy_pipeline.py --config config/strategies/sr_reversal_rr_reg_weighted
```

**优点**：
- 可以同时保留两个版本
- 方便对比测试
- 不影响原策略

**缺点**：
- 需要维护两个策略目录
- 需要复制其他配置文件

## 方案 3：使用符号链接（高级）

如果你想灵活切换，可以使用符号链接：

```bash
# 备份原配置
mv labels.yaml labels.yaml.original

# 创建符号链接指向带权重的配置
ln -s labels_with_weights.yaml labels.yaml

# 需要切换回原配置时
rm labels.yaml
ln -s labels.yaml.original labels.yaml
```

**优点**：
- 灵活切换
- 不需要复制文件

**缺点**：
- 需要手动管理符号链接

## 推荐方案

**如果只是测试权重效果**：使用方案 1（直接替换）
**如果需要长期对比**：使用方案 2（创建新策略目录）

## 快速切换脚本

创建一个简单的切换脚本 `switch_weights.sh`：

```bash
#!/bin/bash

CONFIG_DIR="config/strategies/sr_reversal_rr_reg"

if [ -f "$CONFIG_DIR/labels.yaml.backup" ]; then
    # 切换到带权重版本
    mv "$CONFIG_DIR/labels.yaml" "$CONFIG_DIR/labels.yaml.no_weights"
    cp "$CONFIG_DIR/labels_with_weights.yaml" "$CONFIG_DIR/labels.yaml"
    echo "✅ 已切换到带权重版本"
else
    # 切换到无权重版本
    mv "$CONFIG_DIR/labels.yaml" "$CONFIG_DIR/labels.yaml.with_weights"
    cp "$CONFIG_DIR/labels.yaml.backup" "$CONFIG_DIR/labels.yaml"
    echo "✅ 已切换到无权重版本"
fi
```

使用方法：
```bash
chmod +x switch_weights.sh
./switch_weights.sh
```

