from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from .counterfactual_eval_3action import (
    CounterfactualEvalConfig,
    train_and_counterfactual_eval_bc3,
)
from .regime_embedding import RegimeEmbeddingConfig, add_regime_onehot
from .shadow_eval_3action import ShadowEvalConfig, train_and_shadow_eval_bc3_from_logs


@dataclass(frozen=True)
class RouterEmbedEvalConfig:
    """
    A/B evaluation:
      A) baseline state_keys (no regime)
      B) augmented state_keys (+regime one-hot)
    """

    regime_cfg: RegimeEmbeddingConfig = RegimeEmbeddingConfig()

    # We reuse default configs but allow overriding state_keys downstream.
    shadow_cfg: ShadowEvalConfig = ShadowEvalConfig()
    cf_cfg: CounterfactualEvalConfig = CounterfactualEvalConfig()


def run_router_embed_eval(
    df_logs: pd.DataFrame,
    *,
    cfg: RouterEmbedEvalConfig = RouterEmbedEvalConfig(),
    out_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Returns a summary dict; writes artifacts if out_dir is provided.
    """
    base_keys = list(cfg.shadow_cfg.state_keys)

    df_aug, onehot_cols = add_regime_onehot(df_logs, cfg=cfg.regime_cfg)
    aug_keys = base_keys + onehot_cols

    # 1) shadow eval
    _, meta_shadow_base, met_shadow_base = train_and_shadow_eval_bc3_from_logs(
        df_logs,
        cfg=ShadowEvalConfig(
            mode_col=cfg.shadow_cfg.mode_col,
            timestamp_col=cfg.shadow_cfg.timestamp_col,
            symbol_col=cfg.shadow_cfg.symbol_col,
            state_keys=tuple(base_keys),
            split_cfg=cfg.shadow_cfg.split_cfg,
            bc_cfg=cfg.shadow_cfg.bc_cfg,
        ),
        out_dir=(str(Path(out_dir) / "baseline_shadow") if out_dir else None),
    )
    _, meta_shadow_aug, met_shadow_aug = train_and_shadow_eval_bc3_from_logs(
        df_aug,
        cfg=ShadowEvalConfig(
            mode_col=cfg.shadow_cfg.mode_col,
            timestamp_col=cfg.shadow_cfg.timestamp_col,
            symbol_col=cfg.shadow_cfg.symbol_col,
            state_keys=tuple(aug_keys),
            split_cfg=cfg.shadow_cfg.split_cfg,
            bc_cfg=cfg.shadow_cfg.bc_cfg,
        ),
        out_dir=(str(Path(out_dir) / "embed_shadow") if out_dir else None),
    )

    # 2) counterfactual eval
    meta_cf_base, met_cf_base, _ = train_and_counterfactual_eval_bc3(
        df_logs,
        cfg=CounterfactualEvalConfig(
            mode_col=cfg.cf_cfg.mode_col,
            timestamp_col=cfg.cf_cfg.timestamp_col,
            symbol_col=cfg.cf_cfg.symbol_col,
            state_keys=tuple(base_keys),
            split_cfg=cfg.cf_cfg.split_cfg,
            bc_cfg=cfg.cf_cfg.bc_cfg,
            sim_cfg=cfg.cf_cfg.sim_cfg,
            score_lambda=cfg.cf_cfg.score_lambda,
            score_mu=cfg.cf_cfg.score_mu,
        ),
        out_dir=(str(Path(out_dir) / "baseline_counterfactual") if out_dir else None),
    )
    meta_cf_aug, met_cf_aug, _ = train_and_counterfactual_eval_bc3(
        df_aug,
        cfg=CounterfactualEvalConfig(
            mode_col=cfg.cf_cfg.mode_col,
            timestamp_col=cfg.cf_cfg.timestamp_col,
            symbol_col=cfg.cf_cfg.symbol_col,
            state_keys=tuple(aug_keys),
            split_cfg=cfg.cf_cfg.split_cfg,
            bc_cfg=cfg.cf_cfg.bc_cfg,
            sim_cfg=cfg.cf_cfg.sim_cfg,
            score_lambda=cfg.cf_cfg.score_lambda,
            score_mu=cfg.cf_cfg.score_mu,
        ),
        out_dir=(str(Path(out_dir) / "embed_counterfactual") if out_dir else None),
    )

    summary = {
        "regime": {
            "n_buckets": int(cfg.regime_cfg.n_buckets),
            "bucket_col": cfg.regime_cfg.out_bucket_col,
            "onehot_cols": onehot_cols,
        },
        "baseline": {
            "state_keys": base_keys,
            "shadow_metrics": met_shadow_base,
            "counterfactual_metrics": met_cf_base,
        },
        "embed": {
            "state_keys": aug_keys,
            "shadow_metrics": met_shadow_aug,
            "counterfactual_metrics": met_cf_aug,
        },
    }

    if out_dir:
        p = Path(out_dir)
        p.mkdir(parents=True, exist_ok=True)
        (p / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        # lightweight top-level report
        html = []
        html.append("<!doctype html><html><head><meta charset='utf-8'>")
        html.append(
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        )
        html.append("<title>Router regime embedding A/B</title>")
        html.append(
            "<style>body{font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial;margin:24px;color:#111}"
            "code{background:#f4f4f5;padding:2px 6px;border-radius:6px}"
            "table{border-collapse:collapse;width:100%;font-size:12px}"
            "th,td{border-bottom:1px solid #eee;text-align:left;padding:6px 8px;vertical-align:top}"
            "th{background:#fafafa}</style></head><body>"
        )
        html.append("<h1>Router regime embedding A/B</h1>")
        html.append("<p>Baseline vs +regime(one-hot) state features.</p>")
        html.append("<h2>Regime config</h2>")
        html.append(
            f"<pre>{json.dumps(asdict(cfg.regime_cfg), ensure_ascii=False, indent=2)}</pre>"
        )

        def _tbl(title: str, a: Dict[str, Any], b: Dict[str, Any]) -> str:
            keys = sorted(set(a.keys()) | set(b.keys()))
            rows = []
            for k in keys:
                rows.append(
                    f"<tr><td><code>{k}</code></td><td>{a.get(k)}</td><td>{b.get(k)}</td></tr>"
                )
            return (
                f"<h2>{title}</h2>"
                "<table><thead><tr><th>metric</th><th>baseline</th><th>embed</th></tr></thead>"
                f"<tbody>{''.join(rows)}</tbody></table>"
            )

        html.append(_tbl("Shadow metrics", met_shadow_base, met_shadow_aug))
        html.append(_tbl("Counterfactual metrics", met_cf_base, met_cf_aug))
        html.append(
            "<p>See subfolders for full reports: baseline_shadow/, embed_shadow/, baseline_counterfactual/, embed_counterfactual/.</p>"
        )
        html.append("</body></html>")
        (p / "report.html").write_text("\n".join(html), encoding="utf-8")

        # Store meta for traceability
        meta = {
            "cfg": {
                "regime_cfg": asdict(cfg.regime_cfg),
                "shadow_cfg": {
                    "state_keys": list(cfg.shadow_cfg.state_keys),
                    "split_cfg": asdict(cfg.shadow_cfg.split_cfg),
                    "bc_cfg": asdict(cfg.shadow_cfg.bc_cfg),
                },
                "counterfactual_cfg": {
                    "state_keys": list(cfg.cf_cfg.state_keys),
                    "split_cfg": asdict(cfg.cf_cfg.split_cfg),
                    "bc_cfg": asdict(cfg.cf_cfg.bc_cfg),
                    "sim_cfg": asdict(cfg.cf_cfg.sim_cfg),
                },
            },
            "meta_shadow_base": meta_shadow_base,
            "meta_shadow_embed": meta_shadow_aug,
            "meta_cf_base": meta_cf_base,
            "meta_cf_embed": meta_cf_aug,
        }
        (p / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    return summary
