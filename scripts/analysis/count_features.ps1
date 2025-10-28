# PowerShell脚本 - 统计特征数量

Write-Host ""
Write-Host "🔍 统计项目特征数量..." -ForegroundColor Cyan
Write-Host ""

# 切换到项目根目录
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Resolve-Path (Join-Path $scriptPath "..\..")
Set-Location $projectRoot

# 运行Python脚本
python scripts\analysis\count_features.py

Write-Host ""
Write-Host "✅ 特征统计完成！" -ForegroundColor Green
Write-Host "   报告: reports\feature_count_report.txt" -ForegroundColor Yellow
Write-Host "   数据: reports\feature_count_data.json" -ForegroundColor Yellow
Write-Host ""

