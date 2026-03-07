#!/usr/bin/env python3
with open("/home/yin/trading/ml_trading_bot/README_CN.md", "r", encoding="utf-8") as f:
    lines = f.readlines()
out = "/home/yin/trading/ml_trading_bot/_tmp_readme.txt"
with open(out, "w", encoding="utf-8") as o:
    o.write(f"TOTAL: {len(lines)}\n")
    for i in range(95, min(120, len(lines))):
        o.write(f"L{i+1}: {lines[i]}")
