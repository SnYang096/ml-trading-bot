# Quarterly Rolling Re-training
# 季度滚动再训练：每季度用最新数据重新训练，测试下一季度

param(
    [string]$Symbols = "BTCUSDT",
    [int]$InitialQuarters = 8,  # 初始训练期：8个季度 = 2年
    [int]$StartYear = 2021,
    [int]$EndYear = 2025
)

Write-Host ""
Write-Host "╔═══════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║                                                               ║" -ForegroundColor Cyan
Write-Host "║     📊 Quarterly Rolling Re-training                         ║" -ForegroundColor Cyan
Write-Host "║                                                               ║" -ForegroundColor Cyan
Write-Host "╚═══════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

$env:PYTHONPATH = "src"

Write-Host "📋 Configuration:" -ForegroundColor Yellow
Write-Host "   Symbols            : $Symbols" -ForegroundColor White
Write-Host "   Period             : $StartYear-$EndYear" -ForegroundColor White
Write-Host "   Initial training   : $InitialQuarters quarters (2 years)" -ForegroundColor White
Write-Host ""

Write-Host "💡 Strategy:" -ForegroundColor Yellow
Write-Host "   1. Train on first $InitialQuarters quarters (e.g., 2021-2022)" -ForegroundColor White
Write-Host "   2. Test on next quarter (e.g., 2023Q1)" -ForegroundColor White
Write-Host "   3. Expand training set, retrain" -ForegroundColor White
Write-Host "   4. Test on next quarter" -ForegroundColor White
Write-Host "   5. Repeat until end" -ForegroundColor White
Write-Host ""

Write-Host "⏱️  Estimated time: 20-60 minutes (depends on data)" -ForegroundColor Yellow
Write-Host ""

$response = Read-Host "Continue? (Y/n)"
if ($response -eq 'n' -or $response -eq 'N') {
    Write-Host "Cancelled." -ForegroundColor Yellow
    exit 0
}

Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host ""

$startTime = Get-Date

python scripts/quarterly_rolling_retrain.py `
    --symbols $Symbols `
    --start-year $StartYear `
    --end-year $EndYear `
    --initial-train-quarters $InitialQuarters `
    --output "quarterly_rolling_${Symbols}"

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "❌ Rolling re-training failed!" -ForegroundColor Red
    exit 1
}

$endTime = Get-Date
$duration = ($endTime - $startTime).TotalSeconds

Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "  ✨ Quarterly Rolling Re-training Complete!" -ForegroundColor Green
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host ""
Write-Host "⏱️  Total time: $([math]::Round($duration, 1))s ($([math]::Round($duration/60, 1)) min)" -ForegroundColor White
Write-Host "📁 Results: results/quarterly_rolling_${Symbols}/" -ForegroundColor White
Write-Host ""
Write-Host "📊 Next steps:" -ForegroundColor Cyan
Write-Host "   1. Review quarterly_results.csv" -ForegroundColor White
Write-Host "   2. Compare with static model performance" -ForegroundColor White
Write-Host "   3. Analyze which quarters benefited from retraining" -ForegroundColor White
Write-Host ""


