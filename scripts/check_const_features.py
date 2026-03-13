#!/usr/bin/env python3
"""扫描 FeatureStore layer 中的常量特征（每月 nunique==1）"""
import pandas as pd
import glob
import collections
import sys
import os

layer = sys.argv[1] if len(sys.argv) > 1 else "features_me_60T_87c2c8c784"
root = sys.argv[2] if len(sys.argv) > 2 else "feature_store"

paths = sorted(
    glob.glob(f"{root}/{layer}/*/60T/*.parquet")
    + glob.glob(f"{root}/{layer}/*/240T/*.parquet")
    + glob.glob(f"{root}/{layer}/*/15T/*.parquet")
)

SKIP = {"trade_count"}

col_const = collections.Counter()
col_total = collections.Counter()
col_val = {}
errs = []

for p in paths:
    try:
        df = pd.read_parquet(p)
        for c in df.select_dtypes(include="number").columns:
            if c in SKIP:
                continue
            s = df[c].dropna()
            if len(s) == 0:
                continue
            col_total[c] += 1
            if s.nunique() == 1:
                col_const[c] += 1
                col_val[c] = s.iloc[0]
    except Exception as e:
        errs.append(f"{p}: {e}")

print(f"Layer : {layer}")
print(f"Files : {len(paths)}  |  Errors: {len(errs)}")
if errs:
    for e in errs[:5]:
        print(f"  ERR {e}")
print()

header = f'{"Column":<55} {"const":>7}/{"total":<7} {"ratio":>7}  {"value":>16}  flag'
print(header)
print("-" * len(header))

found = False
for c, cnt in sorted(col_const.items(), key=lambda x: -x[1]):
    tot = col_total[c]
    r = cnt / tot
    if r < 0.05:
        continue
    found = True
    flag = "🔴 BUG" if r >= 0.95 else ("🟡 CHECK" if r >= 0.5 else "🟢 warmup?")
    print(f'{c:<55} {cnt:>7}/{tot:<7} {r:>7.1%}  {str(col_val.get(c, "")):>16}  {flag}')

if not found:
    print("✅ No constant features found (ratio >= 5%)")
