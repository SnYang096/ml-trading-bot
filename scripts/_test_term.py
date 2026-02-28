#!/usr/bin/env python3
"""Quick terminal test"""
import os

out_path = "/home/yin/trading/ml_trading_bot/scripts/_term_test_output.txt"
with open(out_path, "w") as f:
    f.write("alive\n")
print("Terminal is working")
