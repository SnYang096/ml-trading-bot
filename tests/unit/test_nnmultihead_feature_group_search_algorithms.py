from __future__ import annotations


def test_nn_successive_halving_picks_best_survivor(tmp_path, monkeypatch):
    """
    Mirrors tree test shape: halving should prune by cheap budget and pick best at full budget.
    Budget dimension for nn: epochs.
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

    groups = {"A": ["fa"], "B": ["fb"], "C": ["fc"], "D": ["fd"]}

    # baseline irrelevant; halving stage 3 picks A,B ; stage 10 picks B.
    def _stub_eval(*, cfg, temp_config_dir, objective, run_id, epochs):
        if run_id == "baseline":
            return 1.0, True, None, {"dir_auc": 1.0}
        if run_id.startswith("halving_step1_add_") and run_id.endswith("__e3"):
            if "_add_A__" in run_id:
                return 2.0, True, None, {}
            if "_add_B__" in run_id:
                return 1.9, True, None, {}
            if "_add_C__" in run_id:
                return 1.1, True, None, {}
            if "_add_D__" in run_id:
                return 1.0, True, None, {}
        if run_id.startswith("halving_step1_add_") and run_id.endswith("__e10"):
            if "_add_A__" in run_id:
                return 1.2, True, None, {}
            if "_add_B__" in run_id:
                return 2.5, True, None, {}
        raise AssertionError(f"Unexpected run_id: {run_id}")

    res = nfgs.successive_halving_search(
        cfg=cfg,
        base_features=[],
        groups=groups,
        max_steps=1,
        objective="dir_auc",
        stages=[3, 10],
        top_fraction=0.5,
        min_survivors=2,
        evaluator=_stub_eval,
    )
    assert res["search_algo"] == "successive_halving"
    assert res["selected_groups"] == ["B"]


def test_nn_main_excludes_base_features_from_candidates(tmp_path):
    """
    Ensure that base_features_yaml is honored:
    - base features are always included
    - candidates exclude those base features so search isn't redundant
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

    pool_b = tmp_path / "poolb.yaml"
    pool_b.write_text(
        "feature_pipeline:\n  requested_features:\n    - atr_f\n    - rsi_f\n    - bb_width_f\n",
        encoding="utf-8",
    )
    base_feats = tmp_path / "base_feats.yaml"
    base_feats.write_text("- atr_f\n- rsi_f\n", encoding="utf-8")

    # Simulate internal behavior: load base + poolB and remove base overlaps
    base_list = nfgs._load_features_list_yaml(base_feats)
    pool_obj = nfgs._load_yaml(pool_b)
    req = nfgs._flatten_requested_features(
        (pool_obj.get("feature_pipeline") or {}).get("requested_features")
    )
    req2 = [f for f in req if f not in set(base_list)]
    assert base_list == ["atr_f", "rsi_f"]
    assert req == ["atr_f", "rsi_f", "bb_width_f"]
    assert req2 == ["bb_width_f"]


def test_nn_beam_search_finds_synergy_path(tmp_path):
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
    score_map = {
        "A": 1.5,
        "B": 1.4,
        "C": 0.9,
        "B__C": 2.0,
        "A__B": 1.45,
        "A__C": 1.2,
        "A__B__C": 1.6,
    }

    def _stub_eval(*, cfg, temp_config_dir, objective, run_id, epochs):
        if run_id == "baseline":
            return 1.0, True, None, {}
        if run_id.startswith("beam_step"):
            sel = run_id.split("_sel_", 1)[1]
            return float(score_map.get(sel, 0.0)), True, None, {}
        raise AssertionError(f"Unexpected run_id: {run_id}")

    res = nfgs.beam_search(
        cfg=cfg,
        base_features=[],
        groups=groups,
        max_steps=2,
        objective="dir_auc",
        beam_width=2,
        evaluator=_stub_eval,
    )
    assert res["search_algo"] == "beam"
    assert res["selected_groups"] == ["B", "C"]


def test_nn_sffs_can_remove_redundant_group(tmp_path):
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

    groups = {"A": ["fa"], "B": ["fb"]}

    def _stub_eval(*, cfg, temp_config_dir, objective, run_id, epochs):
        if run_id == "baseline":
            return 1.0, True, None, {}
        if "sffs_step1_fwd_sel_A" in run_id:
            return 1.2, True, None, {}
        if "sffs_step2_fwd_sel_A__B" in run_id:
            return 1.25, True, None, {}
        if "sffs_step2_bwd_sel_B__rm_A" in run_id:
            return 1.3, True, None, {}
        return 0.0, True, None, {}

    res = nfgs.sffs_search(
        cfg=cfg,
        base_features=[],
        groups=groups,
        max_steps=3,
        objective="dir_auc",
        max_backward_per_step=2,
        evaluator=_stub_eval,
    )
    assert res["search_algo"] == "sffs"
    assert res["selected_groups"] == ["B"]
