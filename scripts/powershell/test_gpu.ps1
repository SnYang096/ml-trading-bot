# Test GPU availability for LightGBM
# 测试GPU是否可用

Write-Host ""
Write-Host "================================" -ForegroundColor Cyan
Write-Host "  GPU Availability Test" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# Test 1: Check LightGBM installation
Write-Host "[1/4] Checking LightGBM installation..." -ForegroundColor Yellow
try {
    $version = python -c "import lightgbm as lgb; print(lgb.__version__)" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ LightGBM installed: version $version" -ForegroundColor Green
    } else {
        Write-Host "❌ LightGBM not installed!" -ForegroundColor Red
        Write-Host "   Install with: pip install lightgbm" -ForegroundColor Yellow
        exit 1
    }
} catch {
    Write-Host "❌ Error checking LightGBM" -ForegroundColor Red
    exit 1
}

Write-Host ""

# Test 2: Check NVIDIA driver
Write-Host "[2/4] Checking NVIDIA GPU..." -ForegroundColor Yellow
try {
    $nvidiaInfo = nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ NVIDIA GPU detected:" -ForegroundColor Green
        Write-Host "   $nvidiaInfo" -ForegroundColor Gray
    } else {
        Write-Host "⚠️  nvidia-smi not found (GPU may still work)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "⚠️  Could not check NVIDIA driver" -ForegroundColor Yellow
}

Write-Host ""

# Test 3: Test GPU with LightGBM
Write-Host "[3/4] Testing LightGBM GPU training..." -ForegroundColor Yellow
$gpuTest = python -c @"
import warnings
warnings.filterwarnings('ignore')
import lightgbm as lgb
import numpy as np
from sklearn.datasets import make_classification

# Create sample data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Try GPU training
params = {
    'device': 'gpu',
    'objective': 'binary',
    'verbosity': -1,
    'num_leaves': 31,
}

try:
    train_data = lgb.Dataset(X, label=y)
    model = lgb.train(params, train_data, num_boost_round=10, callbacks=[lgb.log_evaluation(0)])
    print('✅ GPU training successful!')
    exit(0)
except Exception as e:
    print(f'❌ GPU training failed: {str(e)}')
    exit(1)
"@ 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Host $gpuTest -ForegroundColor Green
} else {
    Write-Host $gpuTest -ForegroundColor Red
    Write-Host ""
    Write-Host "GPU training failed. This might be because:" -ForegroundColor Yellow
    Write-Host "  - LightGBM was not compiled with GPU support" -ForegroundColor Gray
    Write-Host "  - OpenCL/CUDA drivers are missing" -ForegroundColor Gray
    Write-Host "  - GPU is not compatible" -ForegroundColor Gray
    Write-Host ""
    Write-Host "You can still use CPU for training (will be slower)" -ForegroundColor Cyan
}

Write-Host ""

# Test 4: Check current config
Write-Host "[4/4] Checking configuration..." -ForegroundColor Yellow
$env:PYTHONPATH = "src"
$configCheck = python -c "from ml_trading.config.settings import USE_GPU, GPU_LGBM_PARAMS; print(f'USE_GPU: {USE_GPU}'); print(f'GPU_DEVICE_ID: {GPU_LGBM_PARAMS[\"gpu_device_id\"]}')" 2>$null

if ($LASTEXITCODE -eq 0) {
    Write-Host $configCheck -ForegroundColor Cyan
} else {
    Write-Host "⚠️  Could not read config" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "================================" -ForegroundColor Cyan
Write-Host "  Test Complete!" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  - Run training: .\quick_gpu_train.ps1" -ForegroundColor White
Write-Host "  - Or use menu: .\run_gpu_training.ps1" -ForegroundColor White
Write-Host ""

