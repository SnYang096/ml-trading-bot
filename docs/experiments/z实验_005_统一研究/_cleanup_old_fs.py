#!/usr/bin/env python3
"""Delete feature_store directories older than today."""
import os, datetime, shutil

fs = "/home/yin/trading/ml_trading_bot/feature_store"
today_start = datetime.datetime.now().replace(hour=0, minute=0, second=0).timestamp()

old_dirs = []
old_metas = []
for name in sorted(os.listdir(fs)):
    path = os.path.join(fs, name)
    if os.path.isdir(path) and os.path.getmtime(path) < today_start:
        old_dirs.append(name)
    elif name.endswith(".meta.json") and name != "meta.json" and os.path.getmtime(path) < today_start:
        old_metas.append(name)

print(f"Found {len(old_dirs)} old directories, {len(old_metas)} old meta files")
for d in old_dirs:
    print(f"  rm -rf {d}")
for m in old_metas:
    print(f"  rm {m}")

confirm = input("Delete? [y/N] ")
if confirm.strip().lower() == "y":
    for d in old_dirs:
        shutil.rmtree(os.path.join(fs, d))
    for m in old_metas:
        os.remove(os.path.join(fs, m))
    print("Done.")
else:
    print("Cancelled.")
