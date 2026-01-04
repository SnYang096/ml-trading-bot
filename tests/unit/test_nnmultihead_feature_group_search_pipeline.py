from __future__ import annotations


def test_nn_pipeline_halving_then_beam_then_prune(tmp_path):
    """
    Pipeline behavior (nn):
    - Prefilter keeps B,C and drops A
    - Beam finds synergy B+C
    - SFFS removes C if B alone is better (simulated by stubs)
    """
    from src.time_series_model.diagnostics import nn_feature_group_search as nfgs

    base_dir = tmp_path / "base_cfg"
    base_dir.mkdir(parents=True)
    (base_dir / "features.yaml").write_text(
        "description: base\nfeature_pipeline:\n  requested_features: []\n",
        encoding="utf-8",
    )
    (base_dir / "labels.yaml").write_text(
        "target_column: dummy\nlabel_generator: {}\n", encoding="utf-8"
    )
    (base_dir / "model.yaml").write_text("trainer: {}\n", encoding="utf-8")

    cfg = nfgs.NNFeatureSearchConfig(
        base_config_dir=base_dir,
        symbols="BTCUSDT",
        timeframe="240T",
        start_date="2023-01-01",
        end_date="2025-10-31",
        features_store_root="feature_store",
        features_store_layer="features_x",
        output_dir=tmp_path / "out",
        no_docker=True,
        epochs=10,
    )

    groups = {"A": ["fa"], "B": ["fb"], "C": ["fc"]}

    def _stub_eval(*, cfg, temp_config_dir, objective, run_id, epochs):
        if run_id == "baseline":
            return 1.0, True, None, {}

        # Prefilter
        if run_id.startswith("prefilter_add_") and run_id.endswith("__e3"):
            if "prefilter_add_A" in run_id:
                return 0.5, True, None, {}
            if "prefilter_add_B" in run_id:
                return 1.4, True, None, {}
            if "prefilter_add_C" in run_id:
                return 1.3, True, None, {}
        if run_id.startswith("prefilter_add_") and run_id.endswith("__e10"):
            if "prefilter_add_B" in run_id:
                return 1.45, True, None, {}
            if "prefilter_add_C" in run_id:
                return 1.35, True, None, {}

        # Beam
        if run_id.startswith("beam_step1_sel_"):
            if run_id.endswith("B"):
                return 1.4, True, None, {}
            if run_id.endswith("C"):
                return 1.3, True, None, {}
        if run_id.startswith("beam_step2_sel_") and run_id.endswith("B__C"):
            return 2.0, True, None, {}

        # Prune stage: make B alone better than B+C
        if run_id.startswith("prune_init__"):
            return 2.0, True, None, {}
        if run_id.startswith("prune_try__keep_") and "__rm_C" in run_id:
            return 2.1, True, None, {}
        if run_id.startswith("prune_try__keep_") and "__rm_B" in run_id:
            return 1.2, True, None, {}

        return 0.0, True, None, {}

    res = nfgs.pipeline_sh_beam_sffs(
        cfg=cfg,
        base_features=[],
        groups=groups,
        max_steps=2,
        objective="dir_auc",
        stages=[3, 10],
        top_fraction=0.67,
        min_survivors=2,
        target_survivors=2,
        beam_width=2,
        sffs_max_backward_steps=2,
        evaluator=_stub_eval,
    )
    assert res["search_algo"] == "pipeline_sh_beam_sffs"
    assert res["selected_groups"] == ["B"]
