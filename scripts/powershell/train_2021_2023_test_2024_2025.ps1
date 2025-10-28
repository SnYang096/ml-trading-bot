# Large-Scale Training: 2021-2023 Train → 2024-2025 Test
# 大规模训练：2021-2023年训练 → 2024-2025年测试

param(
    [string]$Symbols = "BTCUSDT,ETHUSDT,SOLUSDT",
    [float]$SampleRate = 0.3,  # 使用30%数据加速训练
    [int]$MaxFiles = $null,    # 限制文件数量（测试用）
    [string]$ModelName = "model_2021_2023_multi"
)

Write-Host ""
Write-Host "╔═══════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║                                                               ║" -ForegroundColor Cyan
Write-Host "║  🚀 Large-Scale Training & Testing Pipeline                 ║" -ForegroundColor Cyan
Write-Host "║     Train: 2021-2023 (3 years × 3 symbols)                  ║" -ForegroundColor Cyan
Write-Host "║     Test: 2024-2025 (2 years)                               ║" -ForegroundColor Cyan
Write-Host "║                                                               ║" -ForegroundColor Cyan
Write-Host "╚═══════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

$env:PYTHONPATH = "src"

Write-Host "📋 Configuration:" -ForegroundColor Yellow
Write-Host "   Symbols      : $Symbols" -ForegroundColor White
Write-Host "   Sample Rate  : $($SampleRate * 100)%" -ForegroundColor White
Write-Host "   Model Name   : $ModelName" -ForegroundColor White
if ($MaxFiles) {
    Write-Host "   Max Files    : $MaxFiles (Test mode)" -ForegroundColor Yellow
}
Write-Host ""

# Calculate data volume
$symbolList = $Symbols -split ","
$yearsPerSymbol = 3 * 12  # 2021-2023
$totalFiles = $symbolList.Count * $yearsPerSymbol
if ($MaxFiles) {
    $totalFiles = [Math]::Min($totalFiles, $MaxFiles)
}

Write-Host "📊 Data Volume Estimate:" -ForegroundColor Yellow
Write-Host "   Training files : $totalFiles files" -ForegroundColor White
Write-Host "   Test files     : ~$($symbolList.Count * 21) files (2024-2025)" -ForegroundColor White
Write-Host "   Sample rate    : $($SampleRate * 100)% (to speed up)" -ForegroundColor Gray
Write-Host ""

Write-Host "⚠️  WARNING:" -ForegroundColor Yellow
Write-Host "   This will process large amounts of data and may take 10-60 minutes." -ForegroundColor White
Write-Host "   GPU acceleration recommended." -ForegroundColor White
Write-Host ""

$response = Read-Host "Continue? (Y/n)"
if ($response -eq 'n' -or $response -eq 'N') {
    Write-Host "Cancelled." -ForegroundColor Yellow
    exit 0
}

Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan

# Step 1: Training
Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "📊 [Step 1/2] Training Model with 2021-2023 Data" -ForegroundColor Yellow
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host ""

$trainStart = Get-Date

$trainCmd = "python scripts/train_multi_year_multi_symbol.py --symbols $($Symbols -replace ',', ' ') --start-year 2021 --end-year 2023 --model-name $ModelName --sample-rate $SampleRate"
if ($MaxFiles) {
    $trainCmd += " --max-files $MaxFiles"
}

Invoke-Expression $trainCmd

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "❌ Training failed!" -ForegroundColor Red
    exit 1
}

$trainEnd = Get-Date
$trainDuration = ($trainEnd - $trainStart).TotalSeconds

Write-Host ""
Write-Host "✅ Training completed: $([math]::Round($trainDuration, 1))s" -ForegroundColor Green
Write-Host ""

# Step 2: Batch OOS Testing on 2024-2025
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "📈 [Step 2/2] Testing on 2024-2025 Data" -ForegroundColor Yellow
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host ""

$testStart = Get-Date

# Test on 2024-2025 for all symbols
foreach ($symbol in $symbolList) {
    Write-Host "Testing $symbol on 2024-2025..." -ForegroundColor Cyan
    
    python scripts/oos_batch_test.py `
        --model "$ModelName" `
        --data-dir "D:\GitHub\trading\rlbot\data\agg_data" `
        --pattern "$symbol-aggTrades-202[45]-.*\.zip" `
        --output "${ModelName}_${symbol}_2024_2025"
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "⚠️  Testing failed for $symbol" -ForegroundColor Yellow
    }
    Write-Host ""
}

$testEnd = Get-Date
$testDuration = ($testEnd - $testStart).TotalSeconds

Write-Host ""
Write-Host "✅ Testing completed: $([math]::Round($testDuration, 1))s" -ForegroundColor Green
Write-Host ""

# Summary
$globalEnd = Get-Date
$totalDuration = ($globalEnd - $trainStart).TotalSeconds

Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "  ✨ Pipeline Complete!" -ForegroundColor Green
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host ""
Write-Host "⏱️  Training time  : $([math]::Round($trainDuration, 1))s" -ForegroundColor White
Write-Host "⏱️  Testing time   : $([math]::Round($testDuration, 1))s" -ForegroundColor White
Write-Host "⏱️  Total time     : $([math]::Round($totalDuration, 1))s ($([math]::Round($totalDuration/60, 1)) min)" -ForegroundColor White
Write-Host ""
Write-Host "📁 Results saved in:" -ForegroundColor Cyan
Write-Host "   Training:" -ForegroundColor White
Write-Host "     - models/$ModelName.txt" -ForegroundColor Gray
Write-Host "     - models/${ModelName}_metadata.json" -ForegroundColor Gray
Write-Host ""
Write-Host "   Testing:" -ForegroundColor White
foreach ($symbol in $symbolList) {
    Write-Host "     - results/${ModelName}_${symbol}_2024_2025/" -ForegroundColor Gray
}
Write-Host ""
Write-Host "🎉 All done! Check the results folders for detailed analysis." -ForegroundColor Green
Write-Host ""

