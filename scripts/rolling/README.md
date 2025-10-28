# 滚动训练脚本 (Rolling Training)

⭐ **推荐使用** - 最佳实践，避免前视偏差

## 脚本

### quarterly_rolling_retrain.py
季度滚动训练 - 使用扩展窗口，每季度重新训练

```bash
python scripts/rolling/quarterly_rolling_retrain.py --gpu
```

### monthly_rolling_retrain.py
月度滚动训练 - 更频繁的重新训练

```bash
python scripts/rolling/monthly_rolling_retrain.py --year 2024
```

### monthly_rolling_2025.py
2025专用 - 使用2024 Q4训练，测试2025 H1

```bash
python scripts/rolling/monthly_rolling_2025.py
```

## 特点

- ✅ 使用 EnhancedFeatureEngineer (260+ 特征)
- ✅ 扩展窗口避免前视偏差
- ✅ 保存模型和特征重要性
- ✅ 详细的回测指标

