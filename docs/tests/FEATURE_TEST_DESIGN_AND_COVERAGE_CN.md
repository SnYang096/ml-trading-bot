## 特征测试设计与覆盖（CN）

本文件把“特征测试应该长什么样”标准化，方便后续：
- 新增特征时按模板补齐测试
- 快速检查当前覆盖是否对齐
- 保存覆盖快照，做长期回归对比（regression）

> 目标：让“特征计算正确 + 可用于生产”的关键性质有统一、可执行、可审计的测试约束。

---

### 四类核心测试（建议每个特征类别都覆盖）

来自 `docs/TODO_LIST.md` 的总结口径（统一在这里固化）：

1) **无未来函数测试（No Lookahead / No Future Leak）** ⭐⭐⭐⭐⭐  
   - 目的：确保任何特征值只依赖当前及过去数据（避免 look-ahead bias）
   - 常见做法：固定历史窗口；只修改未来数据；验证历史特征值不变
   - 适用：几乎所有特征（强制）

2) **多资产归一化测试（Multi-Asset Normalization / Cross-Asset Comparability）** ⭐⭐⭐⭐  
   - 目的：跨资产训练时避免“尺度差异”导致模型学 shortcut；确保分布对齐/可比
   - 常见做法：构造多 symbol（BTC/ETH/...）的样本，验证归一化后的统计量/分位数一致性
   - 适用：标注为 `normalized: true` 或声明 `unitless`/`bounded` 的输出列（强烈推荐）

3) **流式 vs 批量一致性（Streaming vs Batch Parity）** ⭐⭐⭐⭐  
   - 目的：生产推理常是流式/增量，训练常是批量；两者不一致会导致线上失真
   - 常见做法：同一份数据：
     - 批量：一次性算全量
     - 流式：按时间逐步推进（或分块）重复计算
     - 对齐并比较同 timestamp 的输出
   - 适用：带状态/滚动窗口/缓存/按月拼接 warmup 的特征（强制）

4) **特征数学正确性验证（Math Correctness / Invariants）** ⭐⭐⭐  
   - 目的：验证公式/边界条件/单位语义正确，避免 silent bug
   - 常见做法：用小样本手算、断言不变量（范围、单调性、对称性、极值、NaN/Inf 行为）
   - 适用：关键特征与高复杂特征（推荐）

---

### 目录与示例（现有代码在哪里）

#### 1) 推荐的特征测试目录
- `tests/features/`：更“生产导向”的特征测试套件（很多文件已经按 4 类测试组织）
  - 示例：
    - `tests/features/test_market_cap_features.py`
    - `tests/features/test_sr_structure_features.py`
    - `tests/features/test_volume_profile_volatility_future_leak_and_multi_asset.py`

#### 2) 覆盖检查脚本（保存/对齐用）

当前 repo 有一个“按文件检查四类测试覆盖”的脚本：
- `tests/check_feature_test_coverage.py`

使用方式：

```bash
python tests/check_feature_test_coverage.py
python tests/check_feature_test_coverage.py --detailed
python tests/check_feature_test_coverage.py --missing-only
```

> 建议：每次大规模改动（归一化/依赖/缓存/FeatureStore）之后跑一次，并把输出保存到 `results/quality/` 做快照对比。

保存覆盖快照（推荐命令）：

```bash
mkdir -p results/quality
python tests/check_feature_test_coverage.py --detailed > results/quality/feature_test_coverage.txt
```

#### 3) 质量检查脚本（更偏“按特征名”扫描）
- `tests/check_feature_quality.py`
  - 会从 `config/feature_dependencies.yaml` 读取全量特征名，并扫描测试文件中是否有：
    - 未来泄露验证
    - 多资产归一化验证
    - 模拟数据等

---

### 新增特征时的最小要求（Checklist）

- **No-lookahead**：必须有
- **Streaming vs Batch**：如果是 rolling/stateful 特征，必须有
- **Normalization / Contract**：如果输出是 unitless/bounded/normalized，建议有 cross-asset
- **Math correctness**：至少覆盖关键列的范围、NaN/Inf 行为与一个手算样例

---

### 常见坑（建议在评审时重点看）

- **rolling window 的 warmup**：流式/按月 FeatureStore 的 warmup 逻辑不同步会导致不一致
- **单位语义漂移**：例如 ATR（price_unit）被误当成 unitless 继续被下游复用
- **多输出特征**：只测了其中 1 列，其他列 silently broken
- **symbol 泄露**：把尺度列直接喂给模型（已通过 `exclude_columns` 机制规避）


