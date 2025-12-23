# `ticks_loader_json` 的来源和传递路径

## 问题

从错误信息可以看到：
```
TypeError: compute_footprint_features() missing 1 required positional argument: 'ticks'
```

这说明 `footprint_basic` 特征没有获取到 `ticks_loader_json`。

## `ticks_loader_json` 的创建流程

### 1. 在训练脚本中创建（`train_strategy_pipeline.py`）

```python
# 第 699-705 行
_maybe_configure_vpin_ticks(
    feature_loader,
    symbol=args.symbol,
    data_path=args.data_path,
    start_ts=start_ts,  # 从 df_raw 的时间范围获取
    end_ts=end_ts,
)
```

### 2. `_maybe_configure_vpin_ticks` 函数（第 128-168 行）

```python
def _maybe_configure_vpin_ticks(...):
    """If tick data exists, configure ticks_loader_json for VPIN features."""
    
    # 1. 获取特征配置
    features_cfg = feature_loader.feature_deps.get("features", {})
    vpin_cfg = features_cfg.get("vpin_features")  # ⚠️ 只给 vpin_features 设置
    
    # 2. 查找 tick 文件
    tick_files = list_tick_files(
        symbol=symbol,
        start_ts=start_ts,
        end_ts=end_ts,
        ticks_dir=str(data_path),
        lookback_minutes=60,
    )
    
    # 3. 创建 tick_params 字典
    tick_params = {
        "symbol": symbol,
        "tick_files": [str(Path(f)) for f in tick_files],
        "start_ts": start_ts,
        "end_ts": end_ts,
        "lookback_minutes": 60,
    }
    
    # 4. 序列化为 JSON 字符串并存储到 vpin_features 的 compute_params
    compute_params["ticks_loader_json"] = serialize_tick_loader_params(tick_params)
    # 存储位置：features_cfg["vpin_features"]["compute_params"]["ticks_loader_json"]
```

### 3. 存储位置

```
feature_loader.feature_deps["features"]["vpin_features"]["compute_params"]["ticks_loader_json"]
```

这是一个 JSON 字符串，包含：
```json
{
  "symbol": "BTCUSDT",
  "tick_files": ["/workspace/data/parquet_data/BTCUSDT_2025-01.parquet", ...],
  "start_ts": "2025-01-01 00:00:00",
  "end_ts": "2025-07-31 23:59:59",
  "lookback_minutes": 60
}
```

## 问题：`footprint_basic` 没有 `ticks_loader_json`

### 当前情况

1. ✅ `vpin_features` 有 `ticks_loader_json`（在 `_maybe_configure_vpin_ticks` 中设置）
2. ❌ `footprint_basic` 没有 `ticks_loader_json`（没有被设置）

### 在 `_build_call_args` 中的获取

```python
# src/features/loader/parallel_computer.py 第 80-130 行
def _build_call_args(feature_info, df, ticks_loader_json=None):
    compute_params = feature_info.get("compute_params", {}) or {}
    
    # 如果 ticks_loader_json 参数为 None，尝试从 compute_params 获取
    if ticks_loader_json is None:
        ticks_loader_json = compute_params.get("ticks_loader_json")
    
    # 检查函数是否需要 ticks
    if "ticks" in func_sig.parameters and ticks_loader_json:
        # 加载 ticks...
```

### 在并行计算中的传递

```python
# src/features/loader/parallel_computer.py 第 640-650 行
compute_params = feature_info.get("compute_params", {})
ticks_loader_json = compute_params.get("ticks_loader_json")  # ⚠️ 从当前特征的 compute_params 获取

# 传递给 worker
future = executor.submit(
    _compute_single_feature_worker_monthly,
    ...
    ticks_loader_json,  # 如果 footprint_basic 没有，这里就是 None
)
```

## 解决方案

### 方案 1：为所有需要 ticks 的特征设置 `ticks_loader_json`

修改 `_maybe_configure_vpin_ticks`，为所有需要 ticks 的特征设置：

```python
def _maybe_configure_vpin_ticks(...):
    # ... 查找 tick_files ...
    
    tick_params = {...}
    ticks_loader_json = serialize_tick_loader_params(tick_params)
    
    # 为所有需要 ticks 的特征设置
    features_need_ticks = ["vpin_features", "footprint_basic"]
    for feature_name in features_need_ticks:
        if feature_name in features_cfg:
            compute_params = features_cfg[feature_name].setdefault("compute_params", {})
            if not compute_params.get("ticks_loader_json"):
                compute_params["ticks_loader_json"] = ticks_loader_json
```

### 方案 2：在 `_build_call_args` 中共享 `ticks_loader_json`

如果当前特征没有 `ticks_loader_json`，尝试从其他特征（如 `vpin_features`）中获取：

```python
def _build_call_args(feature_info, df, ticks_loader_json=None, all_features=None):
    # 如果当前特征没有 ticks_loader_json，尝试从其他特征获取
    if ticks_loader_json is None and all_features:
        # 尝试从 vpin_features 获取（因为它通常最先被配置）
        vpin_cfg = all_features.get("vpin_features", {})
        vpin_params = vpin_cfg.get("compute_params", {})
        ticks_loader_json = vpin_params.get("ticks_loader_json")
```

### 方案 3：在并行计算器中共享

在 `FeatureComputer` 中维护一个全局的 `ticks_loader_json`：

```python
class FeatureComputer:
    def __init__(self, ...):
        self.ticks_loader_json = None  # 全局共享
    
    def compute_features_parallel(self, ...):
        # 第一次遇到需要 ticks 的特征时，从 vpin_features 获取
        if self.ticks_loader_json is None:
            vpin_cfg = features.get("vpin_features", {})
            vpin_params = vpin_cfg.get("compute_params", {})
            self.ticks_loader_json = vpin_params.get("ticks_loader_json")
        
        # 传递给所有 worker
        ticks_loader_json = self.ticks_loader_json
```

## 推荐方案

**推荐使用方案 1**，因为：
1. 最直接：每个需要 ticks 的特征都有自己的配置
2. 最清晰：配置明确，易于理解和维护
3. 最灵活：不同特征可以使用不同的 tick 参数（如果需要）

## 当前错误的原因

```
TypeError: compute_footprint_features() missing 1 required positional argument: 'ticks'
```

**原因**：
1. `footprint_basic` 的 `compute_params` 中没有 `ticks_loader_json`
2. `_build_call_args` 中的 `ticks_loader_json` 参数为 `None`
3. 无法加载 ticks 数据
4. `call_kwargs["ticks"]` 没有被设置
5. 函数调用时缺少 `ticks` 参数

## 修复步骤

1. 修改 `_maybe_configure_vpin_ticks`，为 `footprint_basic` 也设置 `ticks_loader_json`
2. 或者，修改 `_build_call_args` 的调用，传递全局的 `ticks_loader_json`

