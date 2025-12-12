# VPIN 和 Trade Clustering 缓存结构分析

## 📁 一、缓存目录

### ✅ **放在同一个目录下**

**默认缓存目录**：`cache/features/monthly`

**配置方式**：
- **VPIN**：通过 `monthly_cache_dir` 参数（默认：`"cache/features/monthly"`）
- **Trade Clustering**：通过 `monthly_cache_dir` 参数（默认：`"cache/features/monthly"`）

**代码位置**：
```python
# VPIN
def compute_vpin_from_cached_ticks(
    ...
    monthly_cache_dir: Optional[str] = "cache/features/monthly",
    ...
)

# Trade Clustering
def extract_trade_clustering_features(
    ...
    monthly_cache_dir: Optional[str] = "cache/features/monthly",
    ...
)
```

---

## 🔑 二、缓存文件命名规则

### 1. **VPIN 缓存文件**

**文件格式**：`.pkl` (Pickle 格式)

**命名规则**：
- 缓存键通过 `_get_monthly_vpin_cache_key()` 生成
- 键的前缀：`vpin_monthly_` 或 `vpin_monthly_usd_`
- 最终文件名：`{md5_hash}.pkl`

**示例**：
```python
# 缓存键生成逻辑
key_str = f"vpin_monthly_{month_str}_{bucket_volume:.6f}"
# 或
key_str = f"vpin_monthly_usd_{month_str}_{bucket_volume_usd:.6f}"

# 如果有 prev_bucket_state
key_str = f"{key_str}_state_{state_str}"

# 最终文件名（MD5 哈希）
cache_file = cache_dir / f"{hashlib.md5(key_str.encode()).hexdigest()}.pkl"
```

**实际文件名示例**：
```
cache/features/monthly/a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6.pkl  # VPIN 缓存
```

---

### 2. **Trade Clustering 缓存文件**

**文件格式**：`.pkl` (Pickle 格式) 或 `.parquet` (Parquet 格式)

#### A. PKL 格式（状态缓存）

**命名规则**：
- 缓存键通过 `_get_monthly_trade_clustering_cache_key()` 生成
- 键的前缀：`trade_clustering_monthly_`
- 最终文件名：`{md5_hash}.pkl`

**示例**：
```python
# 缓存键生成逻辑
key_str = f"trade_clustering_monthly_{month_str}_{window_size}"

# 如果有 initial_state
key_str = f"{key_str}_state_{state_str}"

# 最终文件名（MD5 哈希）
cache_file = cache_dir / f"{hashlib.md5(key_str.encode()).hexdigest()}.pkl"
```

**实际文件名示例**：
```
cache/features/monthly/b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7.pkl  # Trade Clustering 状态缓存
```

#### B. Parquet 格式（月度持久化）

**命名规则**：
- 直接使用月份和窗口大小
- 文件名格式：`trade_cluster_{month_str}_ws{window_size}.parquet`

**示例**：
```python
file_path = cache_dir / f"trade_cluster_{month_str}_ws{window_size}.parquet"
```

**实际文件名示例**：
```
cache/features/monthly/trade_cluster_2024-01_ws100.parquet  # Trade Clustering 月度数据
cache/features/monthly/trade_cluster_2024-02_ws100.parquet
```

---

## 📊 三、缓存文件对比

| 特征 | VPIN | Trade Clustering |
|------|------|-----------------|
| **缓存目录** | `cache/features/monthly` | `cache/features/monthly` |
| **文件格式** | `.pkl` | `.pkl` (状态缓存) + `.parquet` (月度数据) |
| **文件命名** | MD5 哈希（`{hash}.pkl`） | MD5 哈希（`{hash}.pkl`）或 直接命名（`trade_cluster_{month}_ws{size}.parquet`） |
| **键前缀** | `vpin_monthly_` 或 `vpin_monthly_usd_` | `trade_clustering_monthly_` |
| **内容** | `(buckets, final_state)` 或 `(None, final_state)` | `(DataFrame, state)` 或 `(None, state)` |

---

## 🔍 四、目录结构示例

```
cache/features/monthly/
├── a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6.pkl          # VPIN 缓存（2024-01, bucket_volume=1000）
├── b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7.pkl          # VPIN 缓存（2024-01, bucket_volume=1000, with state）
├── c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8.pkl          # Trade Clustering 状态缓存（2024-01, window_size=100）
├── d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9.pkl          # Trade Clustering 状态缓存（2024-01, window_size=100, with state）
├── trade_cluster_2024-01_ws100.parquet           # Trade Clustering 月度数据（2024-01, window_size=100）
├── trade_cluster_2024-02_ws100.parquet           # Trade Clustering 月度数据（2024-02, window_size=100）
└── ...
```

---

## ✅ 五、总结

### **放在一起，但通过文件名区分**

1. **相同点**：
   - ✅ 使用**同一个缓存目录**：`cache/features/monthly`
   - ✅ 都支持**按月缓存**
   - ✅ 都支持**状态缓存**（跨月连续性）

2. **不同点**：
   - ✅ **VPIN**：只使用 `.pkl` 格式，文件名是 MD5 哈希
   - ✅ **Trade Clustering**：使用 `.pkl` 格式（状态缓存）+ `.parquet` 格式（月度数据）
   - ✅ **文件命名**：通过不同的前缀区分（`vpin_monthly_` vs `trade_clustering_monthly_`）

3. **优势**：
   - ✅ **统一管理**：所有月度缓存都在一个目录下，便于清理和维护
   - ✅ **避免冲突**：通过不同的文件名前缀和哈希值，确保不会冲突
   - ✅ **灵活配置**：可以通过 `monthly_cache_dir` 参数统一配置缓存目录

---

## 🔧 六、缓存清理建议

如果需要清理缓存，可以：

```bash
# 清理所有缓存
rm -rf cache/features/monthly/*

# 只清理 VPIN 缓存（通过文件名前缀识别）
find cache/features/monthly -name "*.pkl" -exec grep -l "vpin" {} \;

# 只清理 Trade Clustering 缓存
rm -f cache/features/monthly/trade_cluster_*.parquet
find cache/features/monthly -name "*.pkl" -exec grep -l "trade_clustering" {} \;
```

---

## 📝 七、代码位置

- **VPIN 缓存键生成**：`src/data_tools/tick_loader.py::_get_monthly_vpin_cache_key()`
- **VPIN 缓存加载**：`src/data_tools/tick_loader.py::_load_monthly_vpin_cache()`
- **VPIN 缓存保存**：`src/data_tools/tick_loader.py::_save_monthly_vpin_cache()`
- **Trade Clustering 缓存键生成**：`src/data_tools/tick_loader.py::_get_monthly_trade_clustering_cache_key()`
- **Trade Clustering 缓存加载**：`src/data_tools/tick_loader.py::_load_monthly_trade_clustering_cache()`
- **Trade Clustering 缓存保存**：`src/data_tools/tick_loader.py::_save_monthly_trade_clustering_cache()`

