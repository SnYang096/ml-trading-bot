# Quick GPU training: January 2025 → Model → February 2025 OOS Test
# 训练1月数据生成模型，然后用2月数据做OOS测试

Write-Host ""
Write-Host "╔═══════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║                                                               ║" -ForegroundColor Cyan
Write-Host "║     🚀 GPU Training Pipeline: Jan 2025 → Feb 2025 OOS       ║" -ForegroundColor Cyan
Write-Host "║                                                               ║" -ForegroundColor Cyan
Write-Host "╚═══════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# 设置PYTHONPATH
$env:PYTHONPATH = "src"

# 检查数据文件是否存在
$janZip = "D:\GitHub\trading\rlbot\data\agg_data\BTCUSDT-aggTrades-2025-01.zip"
$febZip = "D:\GitHub\trading\rlbot\data\agg_data\BTCUSDT-aggTrades-2025-02.zip"

Write-Host "📋 Checking data files..." -ForegroundColor Yellow
Write-Host ""

$missingFiles = @()

if (-not (Test-Path $janZip)) {
    Write-Host "  ❌ January 2025 data not found" -ForegroundColor Red
    $missingFiles += "January"
} else {
    $janSize = (Get-Item $janZip).Length / 1GB
    Write-Host "  ✅ January 2025 data found ($([math]::Round($janSize, 2)) GB)" -ForegroundColor Green
}

if (-not (Test-Path $febZip)) {
    Write-Host "  ❌ February 2025 data not found" -ForegroundColor Red
    $missingFiles += "February"
} else {
    $febSize = (Get-Item $febZip).Length / 1GB
    Write-Host "  ✅ February 2025 data found ($([math]::Round($febSize, 2)) GB)" -ForegroundColor Green
}

if ($missingFiles.Count -gt 0) {
    Write-Host ""
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Yellow
    Write-Host "⚠️  Missing data files!" -ForegroundColor Yellow
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Please download the missing data first:" -ForegroundColor White
    Write-Host ""
    Write-Host "  .\download_to_agg_data.ps1" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Or use the Python script:" -ForegroundColor White
    Write-Host ""
    Write-Host "  python scripts/download_training_data.py --data-dir D:\GitHub\trading\rlbot\data\agg_data --symbols BTCUSDT --start-year 2025 --start-month 1 --end-month 2" -ForegroundColor Cyan
    Write-Host ""
    exit 1
}

Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan

# 全局计时
$globalStart = Get-Date

# Step 1: Training on January data
Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "📊 [Step 1/2] Training Model with GPU" -ForegroundColor Yellow
Write-Host "         Data: January 2025" -ForegroundColor Gray
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host ""

$trainStart = Get-Date
python scripts/train_january.py

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

# Step 2: OOS Test on February data
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "📈 [Step 2/2] Running OOS Test" -ForegroundColor Yellow
Write-Host "         Data: February 2025" -ForegroundColor Gray
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host ""

$oosStart = Get-Date
python scripts/oos_february.py

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
Write-Host "     - models/trained_model_january_2025.pkl" -ForegroundColor Gray
Write-Host "     - models/feature_scalers_january_2025.pkl" -ForegroundColor Gray
Write-Host "     - models/model_info_january_2025.json" -ForegroundColor Gray
Write-Host ""
Write-Host "   OOS Test:" -ForegroundColor White
Write-Host "     - results/february_2025_oos/5T_february_trades.csv" -ForegroundColor Gray
Write-Host "     - results/february_2025_oos/15T_february_trades.csv" -ForegroundColor Gray
Write-Host "     - results/february_2025_oos/60T_february_trades.csv" -ForegroundColor Gray
Write-Host "     - results/february_2025_oos/february_oos_summary.json" -ForegroundColor Gray
Write-Host ""
Write-Host "🎉 All done!" -ForegroundColor Green
Write-Host ""
Write-Host "📊 Next steps:" -ForegroundColor Cyan
Write-Host "   1. Review results in results/february_2025_oos/" -ForegroundColor White
Write-Host "   2. Analyze equity curves and trade history" -ForegroundColor White
Write-Host "   3. Compare performance across timeframes" -ForegroundColor White
Write-Host ""

