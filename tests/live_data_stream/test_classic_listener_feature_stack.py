from __future__ import annotations

import os

from src.live_data_stream.classic_listener_feature_stack import (
    bar_minutes_from_tf,
    build_extra_feature_computers_for_symbol,
    make_primary_feature_computer_factory,
)


def test_bar_minutes_from_tf() -> None:
    assert bar_minutes_from_tf("240T") == 240
    assert bar_minutes_from_tf("15min") == 15
    assert bar_minutes_from_tf("2h") == 120


def test_build_extra_groups_by_timeframe(tmp_path) -> None:
    strategies = tmp_path / "strategies"
    for name, tf in (("bpc", "120T"), ("me", "60T"), ("tpc", "120T")):
        base = strategies / name
        (base / "archetypes").mkdir(parents=True)
        (base / "meta.yaml").write_text(
            f"strategy:\n  timeframe: {tf}\n", encoding="utf-8"
        )
    sr = str(strategies)
    me_pkg = "me"
    tf_bpc = "120T"
    reg = {"bpc": tf_bpc, "me": "60T", "tpc": "120T"}
    extras = build_extra_feature_computers_for_symbol(
        strategies_root=sr,
        registry_tf_map=reg,
        me_pkg=me_pkg,
        tf_bpc=tf_bpc,
        fer_feat=set(),
        fer_nodes=[],
    )
    assert "60T" in extras
    assert "120T" not in extras


def test_primary_factory_merges_same_tf_dirs(tmp_path) -> None:
    strategies = tmp_path / "strategies"
    for name, tf in (("bpc", "120T"), ("srb", "120T")):
        base = strategies / name
        ad = base / "archetypes"
        ad.mkdir(parents=True)
        (base / "meta.yaml").write_text(
            f"strategy:\n  timeframe: {tf}\n", encoding="utf-8"
        )
        (ad / "features.yaml").write_text("nodes: []\ncolumns: []\n", encoding="utf-8")
    bpc_ad = str(strategies / "bpc" / "archetypes")
    srb_ad = str(strategies / "srb" / "archetypes")
    fac = make_primary_feature_computer_factory(
        strategies_root=str(strategies),
        tf_bpc="120T",
        bar_minutes_bpc=120,
        bpc_archetypes_dir=bpc_ad,
        fer_feat=set(),
        fer_nodes=[],
        same_tf_other_dirs=[srb_ad] if os.path.isdir(srb_ad) else [],
    )
    fc = fac("BTCUSDT")
    assert fc.primary_timeframe == "120T"
