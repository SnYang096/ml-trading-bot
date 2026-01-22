# FR/ET数据分布分析报告

## 实验元信息

- **实验时间**: 2026-01-22 03:00:30
- **实验目的**: 分析FR/ET交易的数据分布，找出适合做FR/ET的数据区域
- **数据时间范围**: 2025-05-01 到 2025-10-31

## 分析结果

### top_bottom_comparison

- **atr_percentile**: {'top_mean': 0.6089763374485597, 'top_median': 0.6545138888888888, 'top_std': 0.30510955847526555, 'bottom_mean': 0.5706190623709211, 'bottom_median': 0.5590277777777778, 'bottom_std': 0.30179763421625316}
- **fr_semantic_score**: {'top_mean': 0.1638345575660943, 'top_median': 0.16841971058838534, 'top_std': 0.11364424365103568, 'bottom_mean': 0.17132989139573798, 'bottom_median': 0.16841971058838534, 'bottom_std': 0.10754576114665983}
- **et_semantic_score**: {'top_mean': 0.1638345575660943, 'top_median': 0.16841971058838534, 'top_std': 0.11364424365103568, 'bottom_mean': 0.17132989139573798, 'bottom_median': 0.16841971058838534, 'bottom_std': 0.10754576114665983}
- **cvd_change_5**: {'top_mean': 73136.11934074086, 'top_median': -4938.121000000003, 'top_std': 3829554.0384388072, 'bottom_mean': -344362.5319776952, 'bottom_median': -4095.718999999999, 'bottom_std': 4082259.5436119465}
- **bb_width_normalized**: {'top_mean': 4.705043176785848, 'top_median': 4.11812931641958, 'top_std': 2.182515413893191, 'bottom_mean': 4.792468118215218, 'bottom_median': 4.1233583824751685, 'bottom_std': 2.2857919416832932}

### profitable_vs_unprofitable

- **atr_percentile**: {'profitable_mean': 0.5280308930425753, 'profitable_median': 0.5208333333333334, 'unprofitable_mean': 0.5482948908730159, 'unprofitable_median': 0.5347222222222222}
- **fr_semantic_score**: {'profitable_mean': 0.1584388819612434, 'profitable_median': 0.1669485801367903, 'unprofitable_mean': 0.16671501730920202, 'unprofitable_median': 0.16841971058838534}
- **et_semantic_score**: {'profitable_mean': 0.1584388819612434, 'profitable_median': 0.1669485801367903, 'unprofitable_mean': 0.16671501730920202, 'unprofitable_median': 0.16841971058838534}
- **cvd_change_5**: {'profitable_mean': -168428.39674205607, 'profitable_median': -1757.5635000000007, 'unprofitable_mean': -1458939.8907869046, 'unprofitable_median': -6129.684000000001}
- **bb_width_normalized**: {'profitable_mean': 4.738863923367951, 'profitable_median': 4.155762679906921, 'unprofitable_mean': 4.817521832405092, 'unprofitable_median': 4.37956916838289}

### regime_performance

- **MEAN_REGIME**: {'count': 2, 'mean_return': 0.0036859947442690055, 'std_return': 0.0, 'win_rate': 1.0}
- **NO_TRADE**: {'count': 2946, 'mean_return': -0.0004155245996038053, 'std_return': 0.00972530189843082, 'win_rate': 0.40393754243041413}
- **TC_REGIME**: {'count': 1288, 'mean_return': -0.0006883640630513281, 'std_return': 0.010720252540181444, 'win_rate': 0.37888198757763975}
- **TE_REGIME**: {'count': 1624, 'mean_return': -0.0005298816297847117, 'std_return': 0.00942671024743526, 'win_rate': 0.34236453201970446}

### gate_filter_analysis

- **total_candidates**: 5860
- **passed_count**: 5860
- **failed_count**: 0
- **pass_rate**: 1.0
- **passed_mean_return**: -0.0005057856929905725
- **passed_win_rate**: 0.38156996587030717
- **passed_std_return**: 0.009870097651361426

### golden_ranges

