# PowerShell script for Windows GPU training
# 运行GPU训练和OOS测试

Write-Host "================================" -ForegroundColor Cyan
Write-Host "  ML Trading GPU Training" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# 设置Python路径和PYTHONPATH
$env:PYTHONPATH = "src"

# 检查是否安装了lightgbm
Write-Host "Checking LightGBM installation..." -ForegroundColor Yellow
python -c "import lightgbm as lgb; print(f'LightGBM version: {lgb.__version__}')" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: LightGBM not installed!" -ForegroundColor Red
    Write-Host "Please install it first: pip install lightgbm" -ForegroundColor Yellow
    exit 1
}

# 检查GPU可用性
Write-Host ""
Write-Host "Checking GPU availability..." -ForegroundColor Yellow
python -c "import lightgbm as lgb; print('Testing GPU support...'); import numpy as np; from sklearn.datasets import make_classification; X, y = make_classification(n_samples=100, n_features=10, random_state=42); import warnings; warnings.filterwarnings('ignore'); params = {'device': 'gpu', 'verbosity': -1, 'objective': 'binary'}; try: model = lgb.train(params, lgb.Dataset(X, y), num_boost_round=5); print('✅ GPU is available and working!'); except Exception as e: print(f'⚠️  GPU not available: {e}')" 2>$null

Write-Host ""
Write-Host "================================" -ForegroundColor Cyan
Write-Host "  Select Action:" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host "1. Train with GPU (May 2025 data)" -ForegroundColor Green
Write-Host "2. Run OOS test (June 2025)" -ForegroundColor Green
Write-Host "3. Train + OOS (Full pipeline)" -ForegroundColor Green
Write-Host "4. Generate reports" -ForegroundColor Green
Write-Host "5. Exit" -ForegroundColor Yellow
Write-Host ""

$choice = Read-Host "Enter your choice (1-5)"

switch ($choice) {
    "1" {
        Write-Host ""
        Write-Host "================================" -ForegroundColor Cyan
        Write-Host "  Training with GPU..." -ForegroundColor Cyan
        Write-Host "================================" -ForegroundColor Cyan
        Write-Host ""
        
        $startTime = Get-Date
        python scripts/train_model_wavelet.py
        $endTime = Get-Date
        $duration = ($endTime - $startTime).TotalSeconds
        
        Write-Host ""
        Write-Host "✅ Training completed in $([math]::Round($duration, 2)) seconds" -ForegroundColor Green
    }
    
    "2" {
        Write-Host ""
        Write-Host "================================" -ForegroundColor Cyan
        Write-Host "  Running OOS test (June)..." -ForegroundColor Cyan
        Write-Host "================================" -ForegroundColor Cyan
        Write-Host ""
        
        # Check if model exists
        if (-not (Test-Path "models/trained_model_wavelet_may_2025.pkl")) {
            Write-Host "ERROR: Model not found! Please train first (option 1)" -ForegroundColor Red
            exit 1
        }
        
        $startTime = Get-Date
        python scripts/oos_june.py
        $endTime = Get-Date
        $duration = ($endTime - $startTime).TotalSeconds
        
        Write-Host ""
        Write-Host "✅ OOS test completed in $([math]::Round($duration, 2)) seconds" -ForegroundColor Green
    }
    
    "3" {
        Write-Host ""
        Write-Host "================================" -ForegroundColor Cyan
        Write-Host "  Full Pipeline: Train + OOS" -ForegroundColor Cyan
        Write-Host "================================" -ForegroundColor Cyan
        Write-Host ""
        
        # Step 1: Training
        Write-Host "[Step 1/2] Training with GPU..." -ForegroundColor Yellow
        $trainStart = Get-Date
        python scripts/train_model_wavelet.py
        $trainEnd = Get-Date
        $trainDuration = ($trainEnd - $trainStart).TotalSeconds
        Write-Host "✅ Training completed in $([math]::Round($trainDuration, 2)) seconds" -ForegroundColor Green
        
        Write-Host ""
        
        # Step 2: OOS
        Write-Host "[Step 2/2] Running OOS test..." -ForegroundColor Yellow
        $oosStart = Get-Date
        python scripts/oos_june.py
        $oosEnd = Get-Date
        $oosDuration = ($oosEnd - $oosStart).TotalSeconds
        Write-Host "✅ OOS test completed in $([math]::Round($oosDuration, 2)) seconds" -ForegroundColor Green
        
        Write-Host ""
        Write-Host "================================" -ForegroundColor Cyan
        Write-Host "  Pipeline Summary" -ForegroundColor Cyan
        Write-Host "================================" -ForegroundColor Cyan
        Write-Host "Training time: $([math]::Round($trainDuration, 2))s" -ForegroundColor White
        Write-Host "OOS time: $([math]::Round($oosDuration, 2))s" -ForegroundColor White
        Write-Host "Total time: $([math]::Round($trainDuration + $oosDuration, 2))s" -ForegroundColor White
        Write-Host ""
    }
    
    "4" {
        Write-Host ""
        Write-Host "================================" -ForegroundColor Cyan
        Write-Host "  Generating Reports..." -ForegroundColor Cyan
        Write-Host "================================" -ForegroundColor Cyan
        Write-Host ""
        
        python scripts/reports_june.py
        Write-Host "✅ Reports generated!" -ForegroundColor Green
    }
    
    "5" {
        Write-Host "Exiting..." -ForegroundColor Yellow
        exit 0
    }
    
    default {
        Write-Host "Invalid choice!" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "================================" -ForegroundColor Cyan
Write-Host "  Done!" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Cyan

