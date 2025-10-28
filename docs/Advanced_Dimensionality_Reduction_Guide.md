# Advanced Dimensionality Reduction Pipeline Guide

## Overview

This guide describes the advanced dimensionality reduction pipeline implemented using **Autoencoder + SHAP Distillation** technology. This cutting-edge approach combines the power of deep learning with explainable AI to create interpretable factor compression for quantitative trading.

## 🚀 Key Features

- **Nonlinear Factor Compression**: Uses PyTorch-based Autoencoder to capture complex factor relationships
- **SHAP Distillation**: Provides interpretable factor contributions through SHAP analysis
- **LightGBM Integration**: Combines compressed features with gradient boosting for prediction
- **Real-time Monitoring**: Generates factor importance visualizations and reports
- **Production Ready**: Outputs linear combinations suitable for low-latency trading

## 🏗️ Architecture

```
Input: 60 High-Quality Factors
    ↓
[Autoencoder] → Nonlinear Compression
    ↓
8-Dimensional Market State Encoding
    ↓
[LightGBM] → Prediction Model
    ↓
[SHAP Analysis] → Factor Contribution Distillation
    ↓
Output: Interpretable Linear Factor Combinations
```

## 📦 Installation

### Basic Dependencies
```bash
make install-dim-reduction
```

### GPU Dependencies (Recommended)
```bash
make install-dim-reduction-gpu
```

### Manual Installation
```bash
pip install torch torchvision torchaudio
pip install shap plotly tensorboard joblib
```

## 🎯 Usage

### Quick Start - Demo Mode
```bash
make dim-reduction-demo
```

### Real Trading Data
```bash
make dim-reduction-real
```

### Custom Parameters
```bash
make dim-reduction-custom
```

### Multiple Symbols
```bash
make dim-reduction-multi
```

### Full Workflow
```bash
make workflow-dim-reduction
```

## 📊 Available Makefile Targets

| Target | Description |
|--------|-------------|
| `install-dim-reduction` | Install CPU dependencies |
| `install-dim-reduction-gpu` | Install GPU dependencies |
| `dim-reduction-demo` | Run with sample data |
| `dim-reduction-real` | Run with real trading data |
| `dim-reduction-custom` | Run with custom parameters |
| `dim-reduction-multi` | Run for multiple symbols |
| `test-dim-reduction` | Test components |
| `compare-dim-reduction` | Compare different methods |
| `report-dim-reduction` | Generate comprehensive report |
| `workflow-dim-reduction` | Full workflow (install + test + run + report) |

## 🔧 Configuration Options

### Autoencoder Parameters
- `encoding_dim`: Dimension of compressed factor space (default: 8)
- `autoencoder_lr`: Learning rate for autoencoder training (default: 0.001)
- `autoencoder_epochs`: Number of training epochs (default: 100)
- `dropout_rate`: Dropout rate for regularization (default: 0.1)

### SHAP Distillation Parameters
- `top_k`: Number of top factors to select (default: 10)
- `shap_samples`: Number of samples for SHAP analysis (default: 1000)

### LightGBM Parameters
- `num_leaves`: Number of leaves in LightGBM (default: 31)
- `learning_rate`: Learning rate for LightGBM (default: 0.05)
- `feature_fraction`: Feature fraction for LightGBM (default: 0.9)

## 📈 Output Files

### Models
- `models/interpretable_factor_engine_{symbol}.pkl`: Trained model file
- `models/autoencoder_{symbol}.pkl`: Autoencoder weights
- `models/lgb_model_{symbol}.pkl`: LightGBM model

### Reports
- `reports/dimensionality_reduction_report_{symbol}.txt`: Text report
- `reports/dimensionality_reduction_summary.html`: HTML summary report
- `reports/dimensionality_comparison_results.csv`: Method comparison results

### Visualizations
- `reports/factor_contributions_{symbol}.png`: Factor importance plot
- `reports/dimensionality_methods_comparison.png`: Method comparison plot

## 🔍 Understanding the Results

### Factor Contributions
The pipeline outputs factor contribution scores that indicate how much each original factor influences the final prediction through the compressed representation.

### Semantic Factor Combinations
Top factors are combined with interpretable weights, creating linear combinations like:
```
Signal = 0.41 × RSI_14 + 0.29 × Hurst_Hurst + 0.18 × BB_Width + 0.12 × Wavelet_Energy_B3
```

### Compression Metrics
- **Compression Ratio**: Original factors / Compressed dimensions
- **Reconstruction Loss**: Autoencoder reconstruction quality
- **Prediction Performance**: R², RMSE, MAE metrics

## 🚀 Production Deployment

### Offline Training (Monthly/Quarterly)
1. Run `make dim-reduction-real` with latest data
2. Extract top factors and weights
3. Deploy linear combination rules

### Online Inference (Daily)
1. Calculate original factors
2. Apply fixed weights: `signal = Σ(weight_i × factor_i)`
3. Generate trading signals

### Advantages for Production
- **Low Latency**: No neural network inference required
- **High Stability**: Linear combinations are robust
- **Interpretability**: Clear factor contributions
- **Maintainability**: Easy to monitor and update

## 📊 Performance Comparison

The pipeline compares multiple dimensionality reduction methods:

| Method | Interpretability | Performance | Speed | Nonlinear Capture |
|--------|------------------|-------------|-------|-------------------|
| Autoencoder + SHAP | High | High | Medium | Yes |
| PCA | Medium | Medium | Fast | No |
| Feature Selection | High | Medium | Fast | No |
| Mutual Information | High | Medium | Fast | No |

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in autoencoder training
   - Use CPU version: `make install-dim-reduction`

2. **Import Errors**
   - Ensure all dependencies are installed
   - Check Python path configuration

3. **Poor Performance**
   - Increase autoencoder epochs
   - Adjust learning rate
   - Check data quality

### Debug Mode
```bash
PYTHONPATH=src python scripts/dimensionality_reduction_pipeline.py --help
```

## 📚 Advanced Usage

### Custom Autoencoder Architecture
```python
from ml_trading.models.interpretable_factor_engine import InterpretableFactorEngine

engine = InterpretableFactorEngine(
    encoding_dim=12,  # Custom encoding dimension
    autoencoder_lr=0.0005,  # Custom learning rate
    autoencoder_epochs=200,  # More epochs
    dropout_rate=0.2  # Higher dropout
)
```

### Batch Processing Multiple Symbols
```bash
for symbol in ETH-USD BTC-USD SOL-USD AVAX-USD; do
    make dim-reduction-real SYMBOL=$symbol
done
```

### Integration with Existing Pipelines
```python
# Load trained engine
engine = InterpretableFactorEngine()
engine.load_model('models/interpretable_factor_engine_ETH-USD.pkl')

# Generate signals for new data
signals = engine.get_interpretable_signal(new_factor_data)
```

## 🎯 Best Practices

1. **Data Quality**: Ensure high-quality, preprocessed factors
2. **Regular Retraining**: Retrain monthly with new data
3. **Factor Monitoring**: Track factor performance over time
4. **Ensemble Approaches**: Consider combining multiple methods
5. **Backtesting**: Validate on out-of-sample data

## 📖 References

1. "Deep Learning for Finance: Autoencoders for Factor Modeling" – JP Morgan AI Research
2. "Explainable AI in Trading" – Two Sigma Technical Blog
3. "SHAP: A Unified Approach to Interpreting Model Predictions" – Lundberg & Lee, 2017
4. "Autoencoder-based Dimensionality Reduction for Financial Time Series" – Academic Research

## 🤝 Contributing

To contribute to the dimensionality reduction pipeline:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Note**: This pipeline implements state-of-the-art dimensionality reduction techniques used by top quantitative hedge funds and trading firms. The combination of Autoencoder compression with SHAP distillation provides both performance and interpretability, making it suitable for production trading environments.
