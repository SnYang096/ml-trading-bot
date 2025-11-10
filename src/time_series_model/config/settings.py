"""Configuration settings for the ML trading system."""

# Timeframes for multi-timeframe analysis
TIMEFRAMES = ["5T", "15T", "45T", "60T", "240T"]

# Pipeline stages
STAGE_1_TARGET = "signal"  # Binary classification target (long/short/no_position)
STAGE_2_TARGET = "expected_return"  # Regression target

# Feature engineering settings
TECHNICAL_INDICATORS = ["rsi", "macd", "bbands", "atr", "zigzag"]

# Model parameters
DEFAULT_LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "verbose": -1,
}

# GPU acceleration parameters
# 注意: 
# - "cpu" = CPU 模式（推荐，稳定且快速）
# - "gpu" = OpenCL（需要 OpenCL 设备，容器中不可用）
# - "cuda" = NVIDIA CUDA（当前 LightGBM 版本不支持，会崩溃）
# 
# 推荐：USE_GPU = False (CPU 模式)
# PyTorch 仍然可以使用 GPU 进行深度学习特征提取
USE_GPU = False  # 使用 CPU 模式（推荐，稳定且性能足够）

GPU_LGBM_PARAMS = {
    "device": "cpu",  # CPU 模式（推荐）
    "gpu_platform_id": 0,  # GPU platform ID (仅当使用 gpu/cuda 时)
    "gpu_device_id": 0,  # GPU device ID (仅当使用 gpu/cuda 时)
    "max_bin": 255,  # Reduce bin count
}

# Risk management parameters
STOP_LOSS_MULTIPLIER = 2.0
TAKE_PROFIT_MULTIPLIER = 1.5
MAX_CONSECUTIVE_LOSSES = 3
