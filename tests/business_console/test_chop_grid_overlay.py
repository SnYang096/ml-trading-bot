"""Chop grid ladder overlay for Trade Map."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from mlbot_console.services.chop_grid_overlay import (
    load_chop_grid_map_overlay,
    load_chop_regime_regions,
)


@pytest.fixture
def engine_data_root(tmp_path: Path) -> Path:
    state_dir = tmp_path / "multi_leg_live" / "state"
    state_dir.mkdir(parents=True)
    group = "BNBUSDT_2026-05-19 08:40:00+00:00"
    state_dir.joinpath("chop_grid_BNBUSDT.json").write_text(
        json.dumps(
            {
                "grid_id": group,
                "center": 643.55,
                "spacing": 6.44,
                "active": True,
                "inventory": [
                    {
                        "leg_id": f"{group}_S1",
                        "side": "SHORT",
                        "entry_price": 653.205,
                        "quantity": 0.62,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


def test_chop_grid_overlay_from_engine_state(multi_leg_db, engine_data_root):
    from src.order_management.multi_leg_storage import MultiLegStorage

    storage = MultiLegStorage(str(multi_leg_db))
    run_id = storage.create_run(
        mode="testnet",
        strategies=["chop_grid"],
        symbols=["BNBUSDT"],
        run_id="overlay_bnb",
    )
    group = "BNBUSDT_2026-05-19 08:40:00+00:00"
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "chop_grid",
            "local_order_id": f"{group}_S1_tp",
            "leg_id": f"{group}_S1",
            "symbol": "BNBUSDT",
            "side": "BUY",
            "purpose": "take_profit",
            "price": 643.5545,
            "status": "open",
        }
    )
    out = load_chop_grid_map_overlay(
        multi_leg_db=multi_leg_db,
        engine_data_root=engine_data_root,
        symbol="BNBUSDT",
    )
    assert out["batches"]
    batch = out["batches"][0]
    assert batch["center"] == pytest.approx(643.55)
    assert batch["spacing"] == pytest.approx(6.44)
    levels = {lv["leg"]: lv for lv in batch["levels"]}
    assert levels["S1"]["entry_status"] == "filled"
    assert levels["S1"]["entry_price"] == pytest.approx(653.205)
    assert levels["S1"]["tp_price"] == pytest.approx(643.5545)
    assert levels["S1"]["tp_status"] == "open"
    assert levels["L1"]["grid_price"] == pytest.approx(643.55 - 6.44)
    assert levels["S2"]["grid_price"] == pytest.approx(643.55 + 6.44 * 2)


def test_s2_short_tp_below_entry_when_tp_order_mispriced(
    multi_leg_db, engine_data_root
):
    """Short S2 TP must plot below entry; coerce if DB TP is above entry."""
    from src.order_management.multi_leg_storage import MultiLegStorage

    storage = MultiLegStorage(str(multi_leg_db))
    run_id = storage.create_run(
        mode="testnet",
        strategies=["chop_grid"],
        symbols=["BNBUSDT"],
        run_id="overlay_s2_tp",
    )
    group = "BNBUSDT_2026-05-19 08:40:00+00:00"
    state_path = (
        engine_data_root / "multi_leg_live" / "state" / "chop_grid_BNBUSDT.json"
    )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["inventory"] = [
        {
            "leg_id": f"{group}_S2",
            "side": "SHORT",
            "entry_price": 656.285,
            "quantity": 0.31,
        }
    ]
    state_path.write_text(json.dumps(state), encoding="utf-8")
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "chop_grid",
            "local_order_id": f"{group}_S2_tp",
            "leg_id": f"{group}_S2",
            "symbol": "BNBUSDT",
            "side": "BUY",
            "purpose": "take_profit",
            "price": 662.0,
            "status": "open",
        }
    )
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "chop_grid",
            "local_order_id": f"{group}_S1_tp",
            "leg_id": f"{group}_S1",
            "symbol": "BNBUSDT",
            "side": "BUY",
            "purpose": "take_profit",
            "price": 643.5545,
            "status": "open",
        }
    )
    out = load_chop_grid_map_overlay(
        multi_leg_db=multi_leg_db,
        engine_data_root=engine_data_root,
        symbol="BNBUSDT",
    )
    s2 = next(lv for lv in out["batches"][0]["levels"] if lv["leg"] == "S2")
    assert s2["entry_price"] == pytest.approx(656.285)
    assert s2["tp_price"] == pytest.approx(656.285 - 6.44)
    assert s2["tp_price"] < s2["entry_price"]
    assert s2["tp_price"] < s2["grid_price"]


def test_s2_short_tp_below_grid_when_tp_above_grid_only(multi_leg_db, engine_data_root):
    """Short TP above S2 grid (but below entry) still coerces below grid line."""
    from src.order_management.multi_leg_storage import MultiLegStorage

    storage = MultiLegStorage(str(multi_leg_db))
    run_id = storage.create_run(
        mode="testnet",
        strategies=["chop_grid"],
        symbols=["BNBUSDT"],
        run_id="overlay_s2_tp_grid",
    )
    group = "BNBUSDT_2026-05-19 08:40:00+00:00"
    state_path = (
        engine_data_root / "multi_leg_live" / "state" / "chop_grid_BNBUSDT.json"
    )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["inventory"] = [
        {
            "leg_id": f"{group}_S2",
            "side": "SHORT",
            "entry_price": 670.0,
            "quantity": 0.31,
        }
    ]
    state_path.write_text(json.dumps(state), encoding="utf-8")
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "chop_grid",
            "local_order_id": f"{group}_S2_tp",
            "leg_id": f"{group}_S2",
            "symbol": "BNBUSDT",
            "side": "BUY",
            "purpose": "take_profit",
            "price": 658.0,
            "status": "open",
        }
    )
    out = load_chop_grid_map_overlay(
        multi_leg_db=multi_leg_db,
        engine_data_root=engine_data_root,
        symbol="BNBUSDT",
    )
    s2 = next(lv for lv in out["batches"][0]["levels"] if lv["leg"] == "S2")
    assert s2["tp_price"] == pytest.approx(s2["grid_price"] - 6.44)
    assert s2["tp_price"] < s2["grid_price"]


def test_s2_does_not_use_s1_tp(multi_leg_db, engine_data_root):
    from src.order_management.multi_leg_storage import MultiLegStorage

    storage = MultiLegStorage(str(multi_leg_db))
    run_id = storage.create_run(
        mode="testnet",
        strategies=["chop_grid"],
        symbols=["BNBUSDT"],
        run_id="overlay_s2_isolate",
    )
    group = "BNBUSDT_2026-05-19 08:40:00+00:00"
    state_path = (
        engine_data_root / "multi_leg_live" / "state" / "chop_grid_BNBUSDT.json"
    )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["inventory"] = [
        {
            "leg_id": f"{group}_S2",
            "side": "SHORT",
            "entry_price": 656.285,
            "quantity": 0.31,
        }
    ]
    state_path.write_text(json.dumps(state), encoding="utf-8")
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "chop_grid",
            "local_order_id": f"{group}_S2_tp",
            "leg_id": f"{group}_S2",
            "symbol": "BNBUSDT",
            "side": "BUY",
            "purpose": "take_profit",
            "price": 649.845,
            "status": "open",
        }
    )
    storage.upsert_order(
        {
            "run_id": run_id,
            "strategy": "chop_grid",
            "local_order_id": f"{group}_S1_tp",
            "leg_id": f"{group}_S1",
            "symbol": "BNBUSDT",
            "side": "BUY",
            "purpose": "take_profit",
            "price": 643.5545,
            "status": "open",
        }
    )
    out = load_chop_grid_map_overlay(
        multi_leg_db=multi_leg_db,
        engine_data_root=engine_data_root,
        symbol="BNBUSDT",
    )
    s2 = next(lv for lv in out["batches"][0]["levels"] if lv["leg"] == "S2")
    assert s2["tp_price"] == pytest.approx(649.845)
    assert s2["tp_price"] != pytest.approx(643.5545)


def test_chop_regime_regions_from_feature_bus(bus_root):
    sym = "ETHUSDT"
    feat_dir = bus_root / "features" / "120T"
    df = pd.read_parquet(feat_dir / f"{sym}.parquet")
    df["bpc_semantic_chop"] = 0.2
    df.loc[df.index[2:5], "bpc_semantic_chop"] = 0.65
    df.to_parquet(feat_dir / f"{sym}.parquet", index=False)

    regions = load_chop_regime_regions(bus_root, sym, "2h", entry_min=0.5)
    assert regions
    assert regions[0]["start"] <= regions[0]["end"]
    assert regions[0]["feature"] == "bpc_semantic_chop"


def test_trade_map_bundle_includes_chop_overlay(
    client, engine_data_root, console_settings, monkeypatch
):
    from dataclasses import replace

    from mlbot_console.routers import trade_map as tm

    new_settings = replace(console_settings, engine_data_root=engine_data_root)
    monkeypatch.setattr(tm, "SETTINGS", new_settings)
    r = client.get(
        "/api/trade-map/bundle",
        params={
            "symbol": "BNBUSDT",
            "timeframe": "2h",
            "scopes": "multi_leg",
            "include_ohlcv": "none",
        },
    )
    assert r.status_code == 200
    payload = r.json()
    data = payload.get("data", payload)
    assert data.get("chop_grid_overlay", {}).get("batches")
