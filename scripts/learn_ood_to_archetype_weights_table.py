#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.time_series_model.diagnostics.ood_to_archetype_table import (  # noqa: E402
    build_conditional_survival_table,
    export_weights_yaml,
    load_any,
    load_ood_to_archetype_table_config,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Learn OOD->archetype weights via Conditional Survival Table baseline."
    )
    p.add_argument(
        "--logs",
        required=True,
        help="logs_3action.parquet (must contain ood_score + active_archetype)",
    )
    p.add_argument(
        "--labels",
        required=True,
        help="labels.parquet from extinction-replay-3action (y_surv)",
    )
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument(
        "--config",
        default="config/ood/ood_to_archetype_table_v1.yaml",
        help="Config YAML for bins/archetypes/temperature/min_samples",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_ood_to_archetype_table_config(args.config)
    df_logs = load_any(args.logs)
    df_labels = load_any(args.labels)

    # join logs + labels on (symbol,timestamp)
    df_logs[cfg.timestamp_col] = pd.to_datetime(
        df_logs[cfg.timestamp_col], utc=True, errors="coerce"
    )
    df_labels[cfg.timestamp_col] = pd.to_datetime(
        df_labels[cfg.timestamp_col], utc=True, errors="coerce"
    )
    df = df_logs.merge(
        df_labels[[cfg.symbol_col, cfg.timestamp_col, cfg.label_col]],
        on=[cfg.symbol_col, cfg.timestamp_col],
        how="inner",
    )

    table, meta = build_conditional_survival_table(df, cfg=cfg)
    weights_yaml = export_weights_yaml(table, cfg=cfg)

    table.to_csv(out_dir / "survival_table.csv", index=False)
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # We keep YAML as YAML (human-editable); JSON is kept for debug.
    import yaml

    (out_dir / "weights.yaml").write_text(
        yaml.safe_dump(weights_yaml, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(f"[ok] wrote: {out_dir}/survival_table.csv, meta.json, weights.yaml")


if __name__ == "__main__":
    main()
