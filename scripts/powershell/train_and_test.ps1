# Universal GPU Training and OOS Test Pipeline
# 通用GPU训练和OOS测试流程

param(
    [string]$TrainData = "D:\GitHub\trading\rlbot\data\agg_data\BTCUSDT-aggTrades-2025-01.zip",
    [string]$TestData = "D:\GitHub\trading\rlbot\data\agg_data\BTCUSDT-aggTrades-2025-02.zip",
    [string]$ModelName = "model_btc",
    [string]$Timeframe = "5T"
)

Write-Host ""
Write-Host "╔═══════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║                                                               ║" -ForegroundColor Cyan
Write-Host "║     🚀 Universal GPU Training & OOS Test Pipeline           ║" -ForegroundColor Cyan
Write-Host "║                                                               ║" -ForegroundColor Cyan
Write-Host "╚═══════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# 设置PYTHONPATH
$env:PYTHONPATH = "src"

Write-Host "📋 Configuration:" -ForegroundColor Yellow
Write-Host "   Training data  : $TrainData" -ForegroundColor White
Write-Host "   Test data      : $TestData" -ForegroundColor White
Write-Host "   Model name     : $ModelName" -ForegroundColor White
Write-Host "   Timeframe      : $Timeframe" -ForegroundColor White
Write-Host ""

# Check data files
Write-Host "📋 Checking data files..." -ForegroundColor Yellow
Write-Host ""

$missingFiles = @()

if (-not (Test-Path $TrainData)) {
    Write-Host "  ❌ Training data not found: $TrainData" -ForegroundColor Red
    $missingFiles += "Training"
} else {
    $trainSize = (Get-Item $TrainData).Length / 1GB
    Write-Host "  ✅ Training data found ($([math]::Round($trainSize, 2)) GB)" -ForegroundColor Green
}

if (-not (Test-Path $TestData)) {
    Write-Host "  ❌ Test data not found: $TestData" -ForegroundColor Red
    $missingFiles += "Test"
} else {
    $testSize = (Get-Item $TestData).Length / 1GB
    Write-Host "  ✅ Test data found ($([math]::Round($testSize, 2)) GB)" -ForegroundColor Green
}

if ($missingFiles.Count -gt 0) {
    Write-Host ""
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Yellow
    Write-Host "⚠️  Missing data files!" -ForegroundColor Yellow
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Please download the missing data first:" -ForegroundColor White
    Write-Host "  .\download_to_agg_data.ps1" -ForegroundColor Cyan
    Write-Host ""
    exit 1
}

Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan

# Global timer
$globalStart = Get-Date

# Step 1: Training
Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "📊 [Step 1/2] Training Model with GPU" -ForegroundColor Yellow
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host ""

$trainStart = Get-Date
python scripts/train_model_gpu.py --data "$TrainData" --model-name "$ModelName" --timeframe "$Timeframe"

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "❌ Training failed!" -ForegroundColor Red
    Write-Host ""
    exit 1
}

$trainEnd = Get-Date
$trainDuration = ($trainEnd - $trainStart).TotalSeconds

Write-Host ""
Write-Host "✅ Training completed: $([math]::Round($trainDuration, 2))s" -ForegroundColor Green
Write-Host ""

# Step 2: OOS Test
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "📈 [Step 2/2] Running OOS Test" -ForegroundColor Yellow
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host ""

$oosStart = Get-Date
python scripts/oos_test.py --model "$ModelName" --data "$TestData" --output "${ModelName}_oos"

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "❌ OOS test failed!" -ForegroundColor Red
    Write-Host ""
    exit 1
}

$oosEnd = Get-Date
$oosDuration = ($oosEnd - $oosStart).TotalSeconds

Write-Host ""
Write-Host "✅ OOS test completed: $([math]::Round($oosDuration, 2))s" -ForegroundColor Green
Write-Host ""

# Summary
$globalEnd = Get-Date
$totalDuration = ($globalEnd - $globalStart).TotalSeconds

Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "  ✨ Pipeline Complete!" -ForegroundColor Green
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host ""
Write-Host "⏱️  Training time    : $([math]::Round($trainDuration, 2))s" -ForegroundColor White
Write-Host "⏱️  OOS test time    : $([math]::Round($oosDuration, 2))s" -ForegroundColor White
Write-Host "⏱️  Total time       : $([math]::Round($totalDuration, 2))s" -ForegroundColor White
Write-Host ""
Write-Host "📁 Results saved in:" -ForegroundColor Cyan
Write-Host "   Training:" -ForegroundColor White
Write-Host "     - models/${ModelName}.txt" -ForegroundColor Gray
Write-Host "     - models/${ModelName}_metadata.json" -ForegroundColor Gray
Write-Host ""
Write-Host "   OOS Test:" -ForegroundColor White
Write-Host "     - results/${ModelName}_oos/backtest_results.json" -ForegroundColor Gray
Write-Host "     - results/${ModelName}_oos/trades.csv" -ForegroundColor Gray
Write-Host "     - results/${ModelName}_oos/equity_curve.csv" -ForegroundColor Gray
Write-Host ""
Write-Host "🎉 All done!" -ForegroundColor Green
Write-Host ""

