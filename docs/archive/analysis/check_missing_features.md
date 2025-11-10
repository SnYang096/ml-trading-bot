# 基线模型 vs 增强模型特征对比

## 基线模型 (feature_engineering_wavelet.py)

### 方法列表：
1. `add_technical_indicators()` - 技术指标
2. `add_wavelet_features()` - 小波特征

### add_technical_indicators包含：
- RSI
- MACD (macd, macd_signal, macd_histogram)
- Bollinger Bands (bb_upper, bb_middle, bb_lower, bb_position)
- ATR (atr, atr_normalized)
- Price change & volatility
- Volume features (volume_sma, volume_ratio)
- Normalized features (rsi_normalized, macd_normalized)
- Momentum (momentum_5, momentum_10, momentum_20)
- SMA features (sma_5, sma_10, sma_20, sma_ratio_5_20, sma_ratio_10_20)

### add_wavelet_features包含：
- **Wavelet for close** (wavelet_energy, wavelet_entropy, etc.)
- **Wavelet for volume** (volume_wavelet_*)
- **Hilbert for close** (hilbert_amplitude, hilbert_phase, hilbert_frequency) ⚠️
- **Spectral for close** (spectral_centroid, spectral_bandwidth, spectral_rolloff) ⚠️

---

## 增强模型 (feature_engineering_enhanced.py)

### 方法列表：
1. `add_basic_features()` - 基础技术指标
2. `add_hurst_features()` - Hurst指数（5个信号源）✅
3. `add_wavelet_packet_features()` - WPT（5个信号源）✅
4. `add_hilbert_features()` - Hilbert变换（5个信号源）✅ 新加
5. `add_order_flow_features()` - 订单流特征 ✅ 新加

### add_basic_features包含：
- returns, log_returns, price_change
- SMA (5, 10, 20, 50)
- EMA (5, 10, 20, 50)
- volatility, atr, atr_normalized
- Momentum (5, 10, 20)
- ROC (5, 10, 20)
- RSI_14
- Bollinger Bands (bb_upper, bb_lower, bb_position)
- Volume features (volume_sma_20, volume_ratio)
- MACD (macd, macd_signal, macd_histogram)

---

## ⚠️ 缺失的特征

### 1. Spectral Features (光谱分析) - 缺失 ❌
基线有，增强没有：
- spectral_centroid
- spectral_bandwidth
- spectral_rolloff

### 2. 基线模型的高级衍生特征 - 大部分缺失 ❌
需要检查feature_importance_5T.csv中重要的特征，比如：
- cvd_divergence_strength
- compression_energy  
- structure_tension
- compression_duration
- slope_consistency_score
- momentum_persistence
- 等等...

---

## ✅ 已包含的特征

- Hilbert变换 ✅ (刚添加，5个信号源)
- 订单流特征 ✅ (刚添加)
- WPT ✅ (5个信号源)
- Hurst ✅ (5个信号源)
- 基础技术指标 ✅

---

## 🎯 需要添加的特征

### 高优先级（重要特征）：
1. **Spectral features** (对所有信号源)
2. **CVD高级衍生特征**
   - cvd_divergence_strength
   - cvd slope相关
3. **市场结构特征**
   - compression相关
   - structure_tension
   - slope_consistency
4. **时间特征**
   - hour_sin, hour_cos
   - day_sin, day_cos

### 中优先级：
5. **波动率高级特征**
   - bb_width相关
   - volatility相关
6. **成交量高级特征**
   - volume_anomaly
   - up_vol, down_vol


