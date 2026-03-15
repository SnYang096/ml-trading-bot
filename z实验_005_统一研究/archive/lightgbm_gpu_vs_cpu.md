# LightGBM GPU vs CPU 选择：基于本项目实际数据的结论

## 当前配置现状

### 代码中的设备策略

| 训练入口                                      | 文件                         | 设备策略                            | 实际效果               |
| --------------------------------------------- | ---------------------------- | ----------------------------------- | ---------------------- |
| meta-algorithm 管线 (gate/evidence/prefilter) | `model_training.py` L346-351 | `use_gpu=torch.cuda.is_available()` | **有 CUDA 自动用 GPU** |
| 策略模型训练 (LightGBM/XGBoost/CatBoost)      | `strategy_trainer.py` L132   | `use_gpu=True` (默认)               | **默认 GPU**           |
| SHAP 特征筛选 (8-fold)                        | `shap_feature_selection.py`  | 直接调 `lgb.train()`, 无 GPU 设置   | **CPU**                |
| 旧入口 (LightGBMTrainer)                      | `lightgbm_model.py` L39-41   | `USE_GPU=False`, `device: "cpu"`    | **CPU** (被覆盖)       |

**结论: 管线中 LightGBM 混用 GPU 和 CPU。meta-algorithm 管线和策略训练走 GPU, SHAP 走 CPU。**

### max_bin 问题

- `lightgbm_model.py`: `max_bin=255`
- `model_training.py` / `strategy_trainer.py`: **未设置** → LightGBM 默认 255
- GPU 最优: `max_bin=63`（官方建议）
- 当前 max_bin=255 + GPU → **不是 GPU 最优配置，GPU 加速效果打折**

---

## 本项目实际数据规模

| 策略   | 时间粒度        | 总行数       | 单 SHAP fold | 单 WF fold 训练 | 单 WF fold holdout |
| ------ | --------------- | ------------ | ------------ | --------------- | ------------------ |
| BPC    | 240T (4h)       | ~39,000      | ~4,875       | 17K-27K         | ~6,500             |
| FER    | 240T (4h)       | ~45,000      | ~5,625       | 20K-31K         | ~7,500             |
| ME     | 60T (1h)        | ~187,000     | ~23,375      | 82K-134K        | ~26,000            |
| **LV** | **15T (15min)** | **~750,000** | **~93,750**  | **330K-530K**   | **~105,000**       |

特征数: 50-200 列（SHAP 裁剪后更少）

> LV 数据量估算: 3年 × 365天 × 24h × (60/15) bars × 6 symbols ≈ 750K rows
> LV 是唯一数据量接近 1M 的策略，GPU 训练有明确收益

---

## GPU vs CPU 在本项目数据下的判断

### 硬件

```
GPU: RTX 3080 (29 TFLOPS, 760 GB/s)
CPU: i7 12核 (~1 TFLOPS, 50-90 GB/s)
```

### 数据规模 → 设备选择

| 数据规模  | 谁更快      | 本项目对应场景                   |
| --------- | ----------- | -------------------------------- |
| < 50K     | CPU 快      | BPC/FER 全量, SHAP 每 fold       |
| 50K-100K  | 差不多      | ME SHAP 每 fold, LV SHAP 每 fold |
| 100K-500K | GPU 稍快    | ME 全量训练, LV WF fold 训练     |
| 500K-1M   | GPU 快 2-3× | **LV 全量训练 (~750K)**          |

### 逐场景判断

| 场景                  | 数据量         | 当前设备 | 最优设备             | 差异                                 |
| --------------------- | -------------- | -------- | -------------------- | ------------------------------------ |
| BPC meta-algorithm    | ~39K rows      | GPU      | **CPU**              | GPU overhead > 计算收益, CPU 快      |
| FER meta-algorithm    | ~45K rows      | GPU      | **CPU**              | 同上                                 |
| ME meta-algorithm     | ~187K rows     | GPU      | **GPU (有微弱优势)** | 接近临界点, 收益不大                 |
| **LV meta-algorithm** | **~750K rows** | GPU      | **GPU (明确优势)**   | **数据量进入 GPU 最优区间, 快 2-3×** |
| SHAP 8-fold (BPC)     | ~4,875/fold    | CPU      | **CPU**              | 正确, 太小不值得 GPU                 |
| SHAP 8-fold (ME)      | ~23K/fold      | CPU      | **CPU**              | 正确, 仍太小                         |
| **SHAP 8-fold (LV)**  | **~93K/fold**  | CPU      | **差不多**           | **接近 CPU/GPU 临界点**              |
| 策略模型训练 (BPC)    | ~27K rows      | GPU      | **CPU**              | GPU overhead 不值                    |
| 策略模型训练 (ME)     | ~134K rows     | GPU      | **GPU (微弱)**       | 接近临界点                           |
| **策略模型训练 (LV)** | **~530K rows** | GPU      | **GPU (快 2-3×)**    | **数据量大, GPU 收益明确**           |

---

## 结论

### 一句话

```
本项目 BPC/FER 数据量 < 50K → CPU 更快
本项目 ME 数据量 ~187K → GPU 微弱优势, 但 max_bin=255 拖后腿
本项目 LV 数据量 ~750K → GPU 快 2-3×, 是唯一值得 GPU 的策略
SHAP 8-fold 每 fold < 25K → CPU 更快 (LV ~93K/fold 接近临界点)
```

### 具体建议

1. **BPC / FER**: 应该用 CPU。当前 `model_training.py` 自动检测到 CUDA 就用 GPU, 对 BPC/FER 反而更慢
2. **ME**: GPU 有微弱优势 (~20-30% 提速), 但不是数量级差异。如果用 GPU, 应设 `max_bin=63`
3. **LV (15min)**: ~750K rows, **GPU 有明确收益 (快 2-3×)**。LV 是唯一值得 GPU 训练的策略。如果用 GPU, 必须设 `max_bin=63`
4. **SHAP 8-fold**: 当前已是 CPU, 正确 (LV 的 ~93K/fold 接近临界点, 可选 GPU)

### max_bin 问题

如果保留 GPU 训练 ME:
```
# 当前 (默认)
max_bin = 255  → GPU histogram 慢

# 推荐 (GPU 优化)
max_bin = 63   → GPU histogram 快 (官方建议)
```

但 `max_bin=63` 会**降低模型精度**（bin 更粗糙），对量化特征可能有影响。需实验验证。

### 为什么很多量化团队 GPU 训练反而变慢

1. 数据量不够大 (< 100K) → GPU kernel 启动 + 数据拷贝 overhead > 计算收益
2. max_bin 没改成 63 → GPU histogram 没有加速
3. LightGBM GPU 版本只加速 histogram 构建, 不加速 tree splitting → 整体加速有上限
4. CPU→GPU 每轮 iteration 都有数据同步开销

---

## 行动建议

### 方案 A: 全部改 CPU（推荐, 简单稳定）

BPC/FER/ME 数据量在 CPU 最优区间, GPU 收益微弱; 但 **LV 除外**。

```python
# model_training.py L346-351
# 改为:
production_params["device"] = "cpu"

# strategy_trainer.py 各模型训练函数
# 默认 use_gpu=False
```

### 方案 B: 按数据量自动切换（推荐）

LV (~750K) 用 GPU 快 2-3×, 其他策略用 CPU。

```python
# model_training.py
if torch.cuda.is_available() and len(y_train) > 100_000:
    production_params["device"] = "cuda"
    production_params["max_bin"] = 63  # GPU 优化
else:
    production_params["device"] = "cpu"
```

这样 BPC (~39K) / FER (~45K) 自动走 CPU, ME (~187K) 和 LV (~750K) 走 GPU。

### 方案 C: 保持现状

当前 GPU 对 ME 有微弱优势, BPC/FER 有微弱劣势。LV 尚未进入训练管线 (暂缓中)。
等 LV 上线时再统一调整。
