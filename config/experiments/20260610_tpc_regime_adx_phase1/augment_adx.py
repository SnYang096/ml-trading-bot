#!/usr/bin/env python3
"""Phase 0: 为已有 features_labeled.parquet 追加 ADX 列。

用法:
  PYTHONPATH=src python config/experiments/20260610_tpc_regime_adx_phase1/augment_adx.py \
    --input results/train_final/tpc/train_final_20260604_rd_rerun/tpc/features_labeled.parquet \
    --output results/train_final/tpc/train_final_20260610_adx/tpc/features_labeled.parquet
"""
import argparse, os
import pandas as pd, numpy as np
import talib

def augment_parquet_with_adx(input_path: str, output_path: str):
    df = pd.read_parquet(input_path)
    syms = df['symbol'].unique()
    frames = []
    for sym in syms:
        sub = df[df['symbol'] == sym].copy()
        sub = sub.sort_values('datetime')
        # Read OHLCV for this symbol
        ohlcv_path = f'cache/timeframes/{sym}_120T.parquet'
        if not os.path.exists(ohlcv_path):
            print(f"⚠️  No OHLCV for {sym}, skipping")
            continue
        ohlcv = pd.read_parquet(ohlcv_path)
        ohlcv.index = pd.to_datetime(ohlcv.index)
        # Compute ADX variants
        for period in [14, 50, 100]:
            ohlcv[f'adx_{period}'] = talib.ADX(
                ohlcv['high'].values, ohlcv['low'].values, ohlcv['close'].values,
                timeperiod=period
            )
        # Merge
        sub['ts'] = pd.to_datetime(sub['datetime'])
        sub = sub.set_index('ts')
        common = ohlcv.index.intersection(sub.index)
        for period in [14, 50, 100]:
            col = f'adx_{period}'
            sub[col] = np.nan
            sub.loc[common, col] = ohlcv.loc[common, col].values
        sub = sub.reset_index(drop=True)
        frames.append(sub)
    
    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values(['symbol', 'datetime'])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.to_parquet(output_path)
    print(f"✅ Saved: {output_path} ({len(result)} rows, columns: {list(result.columns[-5:])})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    augment_parquet_with_adx(args.input, args.output)
