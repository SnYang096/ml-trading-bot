# Quick GPU training script - no menu, just train and OOS
# 快速运行：训练 + OOS

Write-Host ""
Write-Host "🚀 Starting GPU Training + OOS Pipeline..." -ForegroundColor Cyan
Write-Host ""

# 设置PYTHONPATH
$env:PYTHONPATH = "src"

# 全局计时
$globalStart = Get-Date

# Step 1: Training
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "📊 [1/2] Training Model with GPU (May 2025)" -ForegroundColor Yellow
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
$trainStart = Get-Date
python scripts/train_model_wavelet.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Training failed!" -ForegroundColor Red
    exit 1
}
$trainEnd = Get-Date
$trainDuration = ($trainEnd - $trainStart).TotalSeconds
Write-Host ""
Write-Host "✅ Training completed: $([math]::Round($trainDuration, 2))s" -ForegroundColor Green
Write-Host ""

# Step 2: OOS
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "📈 [2/2] Running OOS Test (June 2025)" -ForegroundColor Yellow
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
$oosStart = Get-Date
python scripts/oos_june.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ OOS failed!" -ForegroundColor Red
    exit 1
}
$oosEnd = Get-Date
$oosDuration = ($oosEnd - $oosStart).TotalSeconds
Write-Host ""
Write-Host "✅ OOS completed: $([math]::Round($oosDuration, 2))s" -ForegroundColor Green
Write-Host ""

# Summary
$globalEnd = Get-Date
$totalDuration = ($globalEnd - $globalStart).TotalSeconds

Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "  ✨ Pipeline Complete!" -ForegroundColor Green
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host ""
Write-Host "⏱️  Training time : $([math]::Round($trainDuration, 2))s" -ForegroundColor White
Write-Host "⏱️  OOS time      : $([math]::Round($oosDuration, 2))s" -ForegroundColor White
Write-Host "⏱️  Total time    : $([math]::Round($totalDuration, 2))s" -ForegroundColor White
Write-Host ""
Write-Host "📁 Results saved in:" -ForegroundColor Cyan
Write-Host "   - models/trained_model_wavelet_may_2025.pkl" -ForegroundColor Gray
Write-Host "   - results/june_2025_oos/" -ForegroundColor Gray
Write-Host ""
Write-Host "🎉 All done!" -ForegroundColor Green
Write-Host ""

