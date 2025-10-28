@echo off
REM Windows批处理脚本 - 统计特征数量

echo.
echo 🔍 统计项目特征数量...
echo.

cd /d "%~dp0..\.."
python scripts\analysis\count_features.py

echo.
echo ✅ 特征统计完成！
echo    报告: reports\feature_count_report.txt
echo    数据: reports\feature_count_data.json
echo.

pause

