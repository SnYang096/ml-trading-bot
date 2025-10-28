# 数据下载脚本 - 下载到 D:\GitHub\trading\rlbot\data\agg_data
# 下载BTC、ETH、SOL历史交易数据 (2021-2025)

Write-Host ""
Write-Host "╔════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║   Binance 历史交易数据下载器                      ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# 设置目标目录
$TARGET_DIR = "D:\GitHub\trading\rlbot\data\agg_data"

Write-Host "📂 目标目录: $TARGET_DIR" -ForegroundColor Cyan
Write-Host ""

# 检查目录是否存在，不存在则创建
if (-not (Test-Path $TARGET_DIR)) {
    Write-Host "创建目录: $TARGET_DIR" -ForegroundColor Yellow
    New-Item -ItemType Directory -Path $TARGET_DIR -Force | Out-Null
}

# 设置PYTHONPATH
$env:PYTHONPATH = "src"

# 进入项目目录
Set-Location $PSScriptRoot

# 菜单
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "  下载选项:" -ForegroundColor Yellow
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. 查看已下载的数据摘要" -ForegroundColor Green
Write-Host "2. 下载所有币种数据 (BTC + ETH + SOL)" -ForegroundColor Green
Write-Host "3. 只下载 BTC 数据" -ForegroundColor Green
Write-Host "4. 只下载 ETH 数据" -ForegroundColor Green
Write-Host "5. 只下载 SOL 数据" -ForegroundColor Green
Write-Host "6. 自定义下载（指定币种和时间范围）" -ForegroundColor Green
Write-Host "7. 退出" -ForegroundColor Yellow
Write-Host ""

$choice = Read-Host "请选择 (1-7)"

switch ($choice) {
    "1" {
        Write-Host ""
        Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
        Write-Host "  📊 查看数据摘要" -ForegroundColor Yellow
        Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
        Write-Host ""
        
        python scripts/download_training_data.py --data-dir $TARGET_DIR --summary
    }
    
    "2" {
        Write-Host ""
        Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
        Write-Host "  📥 下载所有币种 (BTC + ETH + SOL)" -ForegroundColor Yellow
        Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "时间范围: 2021-01 至 2025-09" -ForegroundColor Gray
        Write-Host "币种: BTCUSDT, ETHUSDT, SOLUSDT" -ForegroundColor Gray
        Write-Host "保存位置: $TARGET_DIR" -ForegroundColor Gray
        Write-Host ""
        Write-Host "⚠️  警告: 这将下载约 177 个文件，可能需要数小时和大量磁盘空间！" -ForegroundColor Yellow
        Write-Host ""
        
        python scripts/download_training_data.py --data-dir $TARGET_DIR
    }
    
    "3" {
        Write-Host ""
        Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
        Write-Host "  📥 下载 BTC 数据" -ForegroundColor Yellow
        Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "时间范围: 2021-01 至 2025-09" -ForegroundColor Gray
        Write-Host "币种: BTCUSDT" -ForegroundColor Gray
        Write-Host "保存位置: $TARGET_DIR" -ForegroundColor Gray
        Write-Host ""
        
        python scripts/download_training_data.py --data-dir $TARGET_DIR --symbols BTCUSDT
    }
    
    "4" {
        Write-Host ""
        Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
        Write-Host "  📥 下载 ETH 数据" -ForegroundColor Yellow
        Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "时间范围: 2021-01 至 2025-09" -ForegroundColor Gray
        Write-Host "币种: ETHUSDT" -ForegroundColor Gray
        Write-Host "保存位置: $TARGET_DIR" -ForegroundColor Gray
        Write-Host ""
        
        python scripts/download_training_data.py --data-dir $TARGET_DIR --symbols ETHUSDT
    }
    
    "5" {
        Write-Host ""
        Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
        Write-Host "  📥 下载 SOL 数据" -ForegroundColor Yellow
        Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "时间范围: 2021-01 至 2025-09" -ForegroundColor Gray
        Write-Host "币种: SOLUSDT" -ForegroundColor Gray
        Write-Host "保存位置: $TARGET_DIR" -ForegroundColor Gray
        Write-Host ""
        
        python scripts/download_training_data.py --data-dir $TARGET_DIR --symbols SOLUSDT
    }
    
    "6" {
        Write-Host ""
        Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
        Write-Host "  ⚙️  自定义下载" -ForegroundColor Yellow
        Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
        Write-Host ""
        
        Write-Host "币种选择:" -ForegroundColor Cyan
        Write-Host "  1 - BTC" -ForegroundColor White
        Write-Host "  2 - ETH" -ForegroundColor White
        Write-Host "  3 - SOL" -ForegroundColor White
        Write-Host "  4 - BTC + ETH" -ForegroundColor White
        Write-Host "  5 - 全部" -ForegroundColor White
        $symbolChoice = Read-Host "选择币种 (1-5)"
        
        $symbols = switch ($symbolChoice) {
            "1" { "BTCUSDT" }
            "2" { "ETHUSDT" }
            "3" { "SOLUSDT" }
            "4" { "BTCUSDT ETHUSDT" }
            "5" { "BTCUSDT ETHUSDT SOLUSDT" }
            default { "BTCUSDT" }
        }
        
        Write-Host ""
        $startYear = Read-Host "开始年份 (默认 2021)"
        if ([string]::IsNullOrWhiteSpace($startYear)) { $startYear = "2021" }
        
        $startMonth = Read-Host "开始月份 (默认 1)"
        if ([string]::IsNullOrWhiteSpace($startMonth)) { $startMonth = "1" }
        
        $endYear = Read-Host "结束年份 (默认 2025)"
        if ([string]::IsNullOrWhiteSpace($endYear)) { $endYear = "2025" }
        
        $endMonth = Read-Host "结束月份 (默认 9)"
        if ([string]::IsNullOrWhiteSpace($endMonth)) { $endMonth = "9" }
        
        Write-Host ""
        Write-Host "即将下载:" -ForegroundColor Cyan
        Write-Host "  币种: $symbols" -ForegroundColor White
        Write-Host "  时间: $startYear-$startMonth 至 $endYear-$endMonth" -ForegroundColor White
        Write-Host "  保存位置: $TARGET_DIR" -ForegroundColor White
        Write-Host ""
        
        $symbolArray = $symbols -split ' '
        python scripts/download_training_data.py --data-dir $TARGET_DIR --symbols @symbolArray --start-year $startYear --start-month $startMonth --end-year $endYear --end-month $endMonth
    }
    
    "7" {
        Write-Host "退出..." -ForegroundColor Yellow
        exit 0
    }
    
    default {
        Write-Host "无效选择！" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "  ✅ 完成！" -ForegroundColor Green
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host ""
Write-Host "💡 提示: 下载的数据保存在 $TARGET_DIR" -ForegroundColor Cyan
Write-Host ""


