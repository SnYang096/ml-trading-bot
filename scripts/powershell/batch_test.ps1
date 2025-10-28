# Batch OOS Testing Script
# 批量OOS测试脚本 - 支持通配符和正则表达式

param(
    [string]$Model = "model_btc",
    [string]$DataDir = "D:\GitHub\trading\rlbot\data\agg_data",
    [string]$Pattern = "BTCUSDT-aggTrades-2025-0[2-9]\.zip",  # 默认: 2月到9月
    [string]$Output = "batch_oos_results"
)

Write-Host ""
Write-Host "╔═══════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║                                                               ║" -ForegroundColor Cyan
Write-Host "║     📊 Batch OOS Testing: Multiple Months Evaluation        ║" -ForegroundColor Cyan
Write-Host "║                                                               ║" -ForegroundColor Cyan
Write-Host "╚═══════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# 设置PYTHONPATH
$env:PYTHONPATH = "src"

Write-Host "📋 Configuration:" -ForegroundColor Yellow
Write-Host "   Model       : $Model" -ForegroundColor White
Write-Host "   Data Dir    : $DataDir" -ForegroundColor White
Write-Host "   Pattern     : $Pattern" -ForegroundColor White
Write-Host "   Output      : $Output" -ForegroundColor White
Write-Host ""

# 显示匹配的文件
Write-Host "🔍 Preview matching files..." -ForegroundColor Yellow
$files = Get-ChildItem "$DataDir\*.zip" | Where-Object { $_.Name -match $Pattern.Replace('\', '') }
if ($files.Count -eq 0) {
    Write-Host "   ❌ No files match pattern: $Pattern" -ForegroundColor Red
    exit 1
}

Write-Host "   Found $($files.Count) files:" -ForegroundColor Green
foreach ($file in $files) {
    $size = [math]::Round($file.Length / 1GB, 2)
    Write-Host "     - $($file.Name) ($size GB)" -ForegroundColor Gray
}

Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host ""

# 确认
$response = Read-Host "Continue with batch testing? (Y/n)"
if ($response -eq 'n' -or $response -eq 'N') {
    Write-Host "Cancelled." -ForegroundColor Yellow
    exit 0
}

# 运行批量测试
$startTime = Get-Date

python scripts/oos_batch_test.py `
    --model "$Model" `
    --data-dir "$DataDir" `
    --pattern "$Pattern" `
    --output "$Output"

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "❌ Batch testing failed!" -ForegroundColor Red
    exit 1
}

$endTime = Get-Date
$duration = ($endTime - $startTime).TotalSeconds

Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "  ✨ Batch Testing Complete!" -ForegroundColor Green
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host ""
Write-Host "⏱️  Total time: $([math]::Round($duration, 2))s" -ForegroundColor White
Write-Host "📁 Results: results/$Output/" -ForegroundColor White
Write-Host ""

