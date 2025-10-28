# 🚀 一键启动脚本 - Windows GPU训练
# START HERE - One-click start for GPU training

Write-Host ""
Write-Host "╔════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║   ML Trading - GPU Training Setup     ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# 切换到脚本所在目录
Set-Location $PSScriptRoot

Write-Host "📍 Working directory: $PWD" -ForegroundColor Gray
Write-Host ""

# Step 1: 测试GPU
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "  [Step 1/2] Testing GPU..." -ForegroundColor Yellow
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host ""

$env:PYTHONPATH = "src"

# Quick GPU test
$gpuAvailable = $false
try {
    $testResult = python -c @"
import warnings
warnings.filterwarnings('ignore')
try:
    import lightgbm as lgb
    import numpy as np
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    params = {'device': 'gpu', 'objective': 'binary', 'verbosity': -1}
    train_data = lgb.Dataset(X, label=y)
    model = lgb.train(params, train_data, num_boost_round=5, verbose_eval=False)
    print('GPU_OK')
except:
    print('GPU_FAIL')
"@ 2>$null
    
    if ($testResult -like "*GPU_OK*") {
        $gpuAvailable = $true
        Write-Host "✅ GPU is available and working!" -ForegroundColor Green
    } else {
        Write-Host "⚠️  GPU not available, will use CPU (slower)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "⚠️  GPU test failed, will use CPU" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "GPU Status in config: " -NoNewline -ForegroundColor Gray
$configGpu = python -c "from ml_trading.config.settings import USE_GPU; print('ENABLED' if USE_GPU else 'DISABLED')" 2>$null
if ($configGpu -eq "ENABLED") {
    Write-Host "✅ ENABLED" -ForegroundColor Green
} else {
    Write-Host "❌ DISABLED" -ForegroundColor Yellow
}

Write-Host ""

# Step 2: 询问是否继续
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "  [Step 2/2] What do you want to do?" -ForegroundColor Yellow
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host ""
Write-Host "Options:" -ForegroundColor Cyan
Write-Host "  [1] Run full test (recommended first time)" -ForegroundColor White
Write-Host "  [2] Quick train + OOS (skip tests)" -ForegroundColor White
Write-Host "  [3] Interactive menu" -ForegroundColor White
Write-Host "  [4] Exit" -ForegroundColor White
Write-Host ""

$choice = Read-Host "Enter your choice (1-4)"

switch ($choice) {
    "1" {
        Write-Host ""
        Write-Host "Running full GPU test..." -ForegroundColor Cyan
        .\test_gpu.ps1
        
        Write-Host ""
        $proceed = Read-Host "Continue to training? (y/N)"
        if ($proceed -eq "y" -or $proceed -eq "Y") {
            .\quick_gpu_train.ps1
        }
    }
    
    "2" {
        Write-Host ""
        Write-Host "Starting quick training..." -ForegroundColor Cyan
        .\quick_gpu_train.ps1
    }
    
    "3" {
        Write-Host ""
        .\run_gpu_training.ps1
    }
    
    "4" {
        Write-Host "Exiting..." -ForegroundColor Yellow
        exit 0
    }
    
    default {
        Write-Host "Invalid choice! Run script again." -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "╔════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║         All Done! 🎉                   ║" -ForegroundColor Green
Write-Host "╚════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

