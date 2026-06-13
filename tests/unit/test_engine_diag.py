"""Component 3 — engine-diag one-shot JSON snapshot (TDD).

Tests written before implementation, exercising:
- Snapshot structure: top-level keys (generated_at_ns, data_dir, slots).
- Per-slot fields: status, positions, open_orders, feed, flags, rejects,
  last_decision, config_fingerprint.
- true_pnl = fills_realized + settlement (realized_pnl_since already includes
  settlement, so we verify the decomposed math).
- Rejects are filtered to each slot's alias (no cross-slot double-counting).
- Graceful degradation with empty events table and missing state.db.
- Flag files: presence + mtime propagate; missing files return null.
- Config fingerprint is stable and includes key inline params.
- --alias filter returns only the requested slot.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from hlanalysis.engine.state import Fill, OpenOrder, Position, Settlement, StateDAL
from hlanalysis.engine.config import (
    AllowlistEntry,
    DeployConfig,
    GlobalRiskConfig,
    StrategiesConfig,
    StrategyConfig,
    load_deploy_config,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_allowlist_entry(**kwargs) -> AllowlistEntry:
    defaults = dict(
        match={"class": "priceBinary"},
        max_position_usd=100.0,
        stop_loss_pct=None,
        tte_min_seconds=0,
        tte_max_seconds=7200,
        price_extreme_threshold=0.95,
        distance_from_strike_usd_min=0.0,
        vol_max=1.0,
    )
    defaults.update(kwargs)
    return AllowlistEntry(**defaults)


def _minimal_global_risk() -> GlobalRiskConfig:
    return GlobalRiskConfig(
        max_total_inventory_usd=500.0,
        max_concurrent_positions=5,
        daily_loss_cap_usd=200.0,
        max_strike_distance_pct=10.0,
        min_recent_volume_usd=1000.0,
        stale_data_halt_seconds=30,
        reconcile_interval_seconds=60,
    )


def _make_strategy_config(alias: str, strategy_type: str = "late_resolution") -> StrategyConfig:
    return StrategyConfig(
        name=strategy_type,
        paper_mode=True,
        account_alias=alias,
        strategy_type=strategy_type,
        allowlist=[_minimal_allowlist_entry()],
        defaults=_minimal_allowlist_entry(exit_safety_d=1.0),
        **{"global": _minimal_global_risk()},
    )


def _make_deploy_config(tmp_path: Path, aliases: list[str]) -> DeployConfig:
    """Write a minimal deploy.yaml with one HL account per alias, then load it."""
    accounts_yaml = "\n".join(
        f"    {alias}:\n"
        f"      venue: hyperliquid\n"
        f"      account_address: '0xabc'\n"
        f"      api_secret_key: '0xdef'\n"
        f"      base_url: https://api.hyperliquid.xyz"
        for alias in aliases
    )
    deploy_yaml = tmp_path / "deploy.yaml"
    deploy_yaml.write_text(f"""
deploy:
  env: test
  accounts:
{accounts_yaml}
  alerts:
    telegram:
      bot_token: T
      chat_id: C
  state_db_path: {tmp_path}/engine/state.db
  kill_switch_path: {tmp_path}/engine/halt
""")
    return load_deploy_config(deploy_yaml)


def _seed_dal(deploy_cfg: DeployConfig, alias: str) -> StateDAL:
    """Create and migrate a state.db at the path the deploy config resolves to."""
    db_path = Path(deploy_cfg.state_db_path_for(alias))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    dal = StateDAL(db_path)
    dal.run_migrations()
    return dal


def _flag_dir(deploy_cfg: DeployConfig, alias: str) -> Path:
    """Return the directory containing halt/restart_blocked for this alias."""
    return Path(deploy_cfg.kill_switch_path_for(alias)).parent


# ---------------------------------------------------------------------------
# Import helper (avoids top-level import failing before implementation exists)
# ---------------------------------------------------------------------------


def _import_diag():
    from hlanalysis.engine import diag  # noqa: PLC0415

    return diag


# ---------------------------------------------------------------------------
# 1. Top-level snapshot structure
# ---------------------------------------------------------------------------


def test_snapshot_top_level_keys(tmp_path):
    diag = _import_diag()
    deploy_cfg = _make_deploy_config(tmp_path, ["v1"])
    strategies = StrategiesConfig(strategies=[_make_strategy_config("v1")])
    _seed_dal(deploy_cfg, "v1")

    snapshot = diag.build_snapshot(deploy_cfg, strategies)

    assert "generated_at_ns" in snapshot
    assert "data_dir" in snapshot
    assert "slots" in snapshot
    assert isinstance(snapshot["slots"], dict)
    assert "v1" in snapshot["slots"]


def test_snapshot_generated_at_is_recent(tmp_path):
    diag = _import_diag()
    deploy_cfg = _make_deploy_config(tmp_path, ["v1"])
    strategies = StrategiesConfig(strategies=[_make_strategy_config("v1")])
    _seed_dal(deploy_cfg, "v1")

    before_ns = time.time_ns()
    snapshot = diag.build_snapshot(deploy_cfg, strategies)
    after_ns = time.time_ns()

    assert before_ns <= snapshot["generated_at_ns"] <= after_ns


# ---------------------------------------------------------------------------
# 2. Per-slot structure: expected keys present
# ---------------------------------------------------------------------------


def test_slot_has_required_keys(tmp_path):
    diag = _import_diag()
    deploy_cfg = _make_deploy_config(tmp_path, ["v1"])
    strategies = StrategiesConfig(strategies=[_make_strategy_config("v1")])
    _seed_dal(deploy_cfg, "v1")

    snapshot = diag.build_snapshot(deploy_cfg, strategies)
    slot = snapshot["slots"]["v1"]

    required = {"status", "positions", "open_orders", "feed", "flags", "rejects", "last_decision", "config_fingerprint"}
    assert required.issubset(set(slot.keys())), f"Missing keys: {required - set(slot.keys())}"


# ---------------------------------------------------------------------------
# 3. true_pnl = fills_realized + settlement
# ---------------------------------------------------------------------------


def test_true_pnl_equals_fills_plus_settlement(tmp_path):
    """Seed fills + settlement; verify snapshot true_pnl = fills_closed_pnl + settle."""
    diag = _import_diag()
    deploy_cfg = _make_deploy_config(tmp_path, ["v1"])
    strategies = StrategiesConfig(strategies=[_make_strategy_config("v1")])

    dal = _seed_dal(deploy_cfg, "v1")

    # Insert a fill with positive closed_pnl
    dal.append_fill(
        Fill(
            fill_id="f-1",
            cloid="hla-v1-1",
            question_idx=42,
            symbol="@30",
            side="buy",
            price=0.95,
            size=10.0,
            fee=0.05,
            ts_ns=1000,
            closed_pnl=5.0,
        )
    )
    # Insert settlement
    dal.record_settlement(question_idx=99, symbol="@40", realized_pnl=15.0, ts_ns=2000)

    snapshot = diag.build_snapshot(deploy_cfg, strategies)
    slot = snapshot["slots"]["v1"]

    true_pnl = slot["positions"]["true_pnl"]
    fills_realized = slot["positions"]["fills_realized_pnl"]
    settlement = slot["positions"]["settlement_pnl"]

    assert abs(true_pnl - (fills_realized + settlement)) < 1e-9
    # Settlement must be non-zero (15.0)
    assert abs(settlement - 15.0) < 1e-9
    # fills_realized = closed_pnl - fee = 5.0 - 0.05 = 4.95
    assert abs(fills_realized - 4.95) < 1e-9


# ---------------------------------------------------------------------------
# 4. Rejects filtered by alias (no cross-slot double-counting)
# ---------------------------------------------------------------------------


def test_rejects_filtered_by_alias(tmp_path):
    """Events for alias v31 must NOT appear in v1's rejects section."""
    diag = _import_diag()
    deploy_cfg = _make_deploy_config(tmp_path, ["v1", "v31"])
    strategies = StrategiesConfig(
        strategies=[
            _make_strategy_config("v1"),
            _make_strategy_config("v31", strategy_type="theta_harvester"),
        ]
    )

    # Seed each alias's DB at its own path
    dal_v1 = _seed_dal(deploy_cfg, "v1")
    dal_v31 = _seed_dal(deploy_cfg, "v31")

    now = time.time_ns()
    # v1: 3 order_rejected events
    for i in range(3):
        dal_v1.append_event(
            ts_ns=now - i * 1000,
            alias="v1",
            kind="order_rejected",
            question_idx=i,
            reason="bad_token",
            payload_json=f'{{"cloid":"c{i}"}}',
        )
    # v31: 2 order_rejected events with different reason
    for i in range(2):
        dal_v31.append_event(
            ts_ns=now - i * 1000,
            alias="v31",
            kind="order_rejected",
            question_idx=100 + i,
            reason="min_notional",
            payload_json=f'{{"cloid":"c31-{i}"}}',
        )

    snapshot = diag.build_snapshot(deploy_cfg, strategies, reject_window_hours=24)

    v1_rejects = snapshot["slots"]["v1"]["rejects"]
    v31_rejects = snapshot["slots"]["v31"]["rejects"]

    # v1 should only have its own bad_token rejects
    v1_reasons = {r["reason"] for r in v1_rejects}
    v31_reasons = {r["reason"] for r in v31_rejects}

    assert "bad_token" in v1_reasons
    assert "min_notional" not in v1_reasons  # v31's events must not bleed in
    assert "min_notional" in v31_reasons
    assert "bad_token" not in v31_reasons

    # Counts must match exactly
    bad_token_row = next(r for r in v1_rejects if r["reason"] == "bad_token")
    assert bad_token_row["count"] == 3
    min_notional_row = next(r for r in v31_rejects if r["reason"] == "min_notional")
    assert min_notional_row["count"] == 2


# ---------------------------------------------------------------------------
# 5. Graceful degradation: empty events table
# ---------------------------------------------------------------------------


def test_degrades_gracefully_empty_events_table(tmp_path):
    """Empty events table → rejects=[], last_decision=None, feed has no heartbeat."""
    diag = _import_diag()
    deploy_cfg = _make_deploy_config(tmp_path, ["v1"])
    strategies = StrategiesConfig(strategies=[_make_strategy_config("v1")])

    _seed_dal(deploy_cfg, "v1")  # migrations run, events table is empty

    snapshot = diag.build_snapshot(deploy_cfg, strategies)
    slot = snapshot["slots"]["v1"]

    assert slot["rejects"] == []
    assert slot["last_decision"] is None
    # feed section exists but heartbeat_ts_ns is None (no heartbeat event)
    assert "feed" in slot
    assert slot["feed"]["last_heartbeat_ts_ns"] is None


# ---------------------------------------------------------------------------
# 6. Graceful degradation: missing state.db
# ---------------------------------------------------------------------------


def test_degrades_gracefully_missing_db(tmp_path):
    """Missing state.db → slot sections are empty/null, no crash."""
    diag = _import_diag()
    deploy_cfg = _make_deploy_config(tmp_path, ["v1"])
    strategies = StrategiesConfig(strategies=[_make_strategy_config("v1")])

    # Do NOT seed — no state.db exists
    snapshot = diag.build_snapshot(deploy_cfg, strategies)
    slot = snapshot["slots"]["v1"]

    # Must not crash; sections degrade gracefully
    assert slot["rejects"] == []
    assert slot["last_decision"] is None
    assert slot["positions"]["open_count"] == 0


# ---------------------------------------------------------------------------
# 7. Flags: presence + mtime
# ---------------------------------------------------------------------------


def test_flags_restart_blocked_detected(tmp_path):
    diag = _import_diag()
    deploy_cfg = _make_deploy_config(tmp_path, ["v1"])
    strategies = StrategiesConfig(strategies=[_make_strategy_config("v1")])
    _seed_dal(deploy_cfg, "v1")

    # Create restart_blocked flag in the correct engine dir for this alias
    flag_dir = _flag_dir(deploy_cfg, "v1")
    flag_dir.mkdir(parents=True, exist_ok=True)
    (flag_dir / "restart_blocked").write_text("blocked")

    snapshot = diag.build_snapshot(deploy_cfg, strategies)
    flags = snapshot["slots"]["v1"]["flags"]

    assert flags["restart_blocked"]["present"] is True
    assert flags["restart_blocked"]["mtime_ns"] is not None
    assert flags["halt"]["present"] is False
    assert flags["halt"]["mtime_ns"] is None


def test_flags_halt_detected(tmp_path):
    diag = _import_diag()
    deploy_cfg = _make_deploy_config(tmp_path, ["v1"])
    strategies = StrategiesConfig(strategies=[_make_strategy_config("v1")])
    _seed_dal(deploy_cfg, "v1")

    # Halt flag is the kill_switch_path itself
    halt_path = Path(deploy_cfg.kill_switch_path_for("v1"))
    halt_path.parent.mkdir(parents=True, exist_ok=True)
    halt_path.write_text("halted")

    snapshot = diag.build_snapshot(deploy_cfg, strategies)
    flags = snapshot["slots"]["v1"]["flags"]

    assert flags["halt"]["present"] is True
    assert flags["halt"]["mtime_ns"] is not None


def test_flags_missing_returns_null_mtime(tmp_path):
    diag = _import_diag()
    deploy_cfg = _make_deploy_config(tmp_path, ["v1"])
    strategies = StrategiesConfig(strategies=[_make_strategy_config("v1")])
    _seed_dal(deploy_cfg, "v1")

    snapshot = diag.build_snapshot(deploy_cfg, strategies)
    flags = snapshot["slots"]["v1"]["flags"]

    assert flags["restart_blocked"]["present"] is False
    assert flags["restart_blocked"]["mtime_ns"] is None
    assert flags["halt"]["present"] is False
    assert flags["halt"]["mtime_ns"] is None


# ---------------------------------------------------------------------------
# 8. Feed: heartbeat event
# ---------------------------------------------------------------------------


def test_feed_last_heartbeat_populated(tmp_path):
    diag = _import_diag()
    deploy_cfg = _make_deploy_config(tmp_path, ["v1"])
    strategies = StrategiesConfig(strategies=[_make_strategy_config("v1")])

    dal = _seed_dal(deploy_cfg, "v1")
    hb_ts = time.time_ns() - 5_000_000_000  # 5s ago

    # Heartbeat events carry alias="" (cross-slot); stored with alias=None in the DB
    dal.append_event(
        ts_ns=hb_ts,
        alias=None,
        kind="engine_heartbeat",
        question_idx=None,
        reason=None,
        payload_json=json.dumps({"events_ingested": 500, "d_events": 10, "n_questions": 3}),
    )

    snapshot = diag.build_snapshot(deploy_cfg, strategies)
    feed = snapshot["slots"]["v1"]["feed"]

    assert feed["last_heartbeat_ts_ns"] == hb_ts
    assert feed["events_ingested"] == 500


def test_feed_stale_state_populated(tmp_path):
    diag = _import_diag()
    deploy_cfg = _make_deploy_config(tmp_path, ["v1"])
    strategies = StrategiesConfig(strategies=[_make_strategy_config("v1")])

    dal = _seed_dal(deploy_cfg, "v1")
    now = time.time_ns()

    dal.append_event(
        ts_ns=now - 2_000_000_000,
        alias=None,
        kind="feed_stale",
        question_idx=None,
        reason=None,
        payload_json=json.dumps({"d_events": 0, "interval_seconds": 30.0}),
    )
    dal.append_event(
        ts_ns=now - 1_000_000_000,
        alias=None,
        kind="feed_recovered",
        question_idx=None,
        reason=None,
        payload_json=None,
    )

    snapshot = diag.build_snapshot(deploy_cfg, strategies)
    feed = snapshot["slots"]["v1"]["feed"]

    # Most recent feed status — recovered after stale
    assert "last_feed_status" in feed
    # last_feed_status should reflect the most recent feed event kind
    assert feed["last_feed_status"] in ("feed_recovered", "feed_stale", "feed_down", None)


# ---------------------------------------------------------------------------
# 9. Open orders with age
# ---------------------------------------------------------------------------


def test_open_orders_with_age(tmp_path):
    diag = _import_diag()
    deploy_cfg = _make_deploy_config(tmp_path, ["v1"])
    strategies = StrategiesConfig(strategies=[_make_strategy_config("v1")])

    dal = _seed_dal(deploy_cfg, "v1")
    now_ns = time.time_ns()
    placed_ns = now_ns - 30_000_000_000  # 30 seconds ago

    dal.upsert_order(
        OpenOrder(
            cloid="hla-v1-1",
            venue_oid=None,
            question_idx=42,
            symbol="@30",
            side="buy",
            price=0.95,
            size=10.0,
            status="open",
            placed_ts_ns=placed_ns,
            last_update_ts_ns=placed_ns,
            strategy_id="late_resolution",
        )
    )

    snapshot = diag.build_snapshot(deploy_cfg, strategies)
    orders = snapshot["slots"]["v1"]["open_orders"]

    assert len(orders) == 1
    order = orders[0]
    assert order["cloid"] == "hla-v1-1"
    assert order["age_seconds"] >= 29.0  # at least 29s
    assert order["age_seconds"] < 60.0  # not unreasonably old


# ---------------------------------------------------------------------------
# 10. Status: paper / halted / blocked from flags
# ---------------------------------------------------------------------------


def test_status_paper_mode(tmp_path):
    diag = _import_diag()
    deploy_cfg = _make_deploy_config(tmp_path, ["v1"])
    # paper_mode=True
    strategies = StrategiesConfig(strategies=[_make_strategy_config("v1")])
    _seed_dal(deploy_cfg, "v1")

    snapshot = diag.build_snapshot(deploy_cfg, strategies)
    assert snapshot["slots"]["v1"]["status"] == "paper"


def test_status_blocked_when_flag_present(tmp_path):
    diag = _import_diag()
    deploy_cfg = _make_deploy_config(tmp_path, ["v1"])
    strategies = StrategiesConfig(strategies=[_make_strategy_config("v1")])
    _seed_dal(deploy_cfg, "v1")

    flag_dir = _flag_dir(deploy_cfg, "v1")
    flag_dir.mkdir(parents=True, exist_ok=True)
    (flag_dir / "restart_blocked").write_text("blocked")

    snapshot = diag.build_snapshot(deploy_cfg, strategies)
    assert snapshot["slots"]["v1"]["status"] == "blocked"


def test_status_halted_when_halt_flag_present(tmp_path):
    diag = _import_diag()
    deploy_cfg = _make_deploy_config(tmp_path, ["v1"])
    # paper_mode=False for this test
    strat = StrategyConfig(
        name="late_resolution",
        paper_mode=False,
        account_alias="v1",
        strategy_type="late_resolution",
        allowlist=[_minimal_allowlist_entry()],
        defaults=_minimal_allowlist_entry(),
        **{"global": _minimal_global_risk()},
    )
    strategies = StrategiesConfig(strategies=[strat])
    _seed_dal(deploy_cfg, "v1")

    halt_path = Path(deploy_cfg.kill_switch_path_for("v1"))
    halt_path.parent.mkdir(parents=True, exist_ok=True)
    halt_path.write_text("halted")

    snapshot = diag.build_snapshot(deploy_cfg, strategies)
    assert snapshot["slots"]["v1"]["status"] == "halted"


# ---------------------------------------------------------------------------
# 11. Config fingerprint: stable hash + inline params
# ---------------------------------------------------------------------------


def test_config_fingerprint_stable(tmp_path):
    diag = _import_diag()
    deploy_cfg = _make_deploy_config(tmp_path, ["v1"])
    strategies = StrategiesConfig(strategies=[_make_strategy_config("v1")])
    _seed_dal(deploy_cfg, "v1")

    snap1 = diag.build_snapshot(deploy_cfg, strategies)
    snap2 = diag.build_snapshot(deploy_cfg, strategies)

    assert snap1["slots"]["v1"]["config_fingerprint"]["hash"] == snap2["slots"]["v1"]["config_fingerprint"]["hash"]


def test_config_fingerprint_has_inline_params(tmp_path):
    diag = _import_diag()
    deploy_cfg = _make_deploy_config(tmp_path, ["v1"])
    strategies = StrategiesConfig(strategies=[_make_strategy_config("v1")])
    _seed_dal(deploy_cfg, "v1")

    snapshot = diag.build_snapshot(deploy_cfg, strategies)
    fp = snapshot["slots"]["v1"]["config_fingerprint"]

    assert "hash" in fp
    assert "strategy_type" in fp
    # Key safety params must be inline for quick config-vs-runtime skew detection
    assert "exit_safety_d" in fp
    assert "paper_mode" in fp


def test_config_fingerprint_changes_with_params(tmp_path):
    diag = _import_diag()
    deploy_cfg = _make_deploy_config(tmp_path, ["v1"])

    strat_a = _make_strategy_config("v1")
    strat_b = StrategyConfig(
        name="late_resolution",
        paper_mode=True,
        account_alias="v1",
        strategy_type="late_resolution",
        allowlist=[_minimal_allowlist_entry()],
        defaults=_minimal_allowlist_entry(exit_safety_d=2.0),  # different exit_safety_d
        **{"global": _minimal_global_risk()},
    )

    _seed_dal(deploy_cfg, "v1")

    snap_a = diag.build_snapshot(deploy_cfg, StrategiesConfig(strategies=[strat_a]))
    snap_b = diag.build_snapshot(deploy_cfg, StrategiesConfig(strategies=[strat_b]))

    assert snap_a["slots"]["v1"]["config_fingerprint"]["hash"] != snap_b["slots"]["v1"]["config_fingerprint"]["hash"]


# ---------------------------------------------------------------------------
# 12. last_decision: most recent decision/terminal event per slot
# ---------------------------------------------------------------------------


def test_last_decision_populated(tmp_path):
    diag = _import_diag()
    deploy_cfg = _make_deploy_config(tmp_path, ["v1"])
    strategies = StrategiesConfig(strategies=[_make_strategy_config("v1")])

    dal = _seed_dal(deploy_cfg, "v1")
    now = time.time_ns()

    dal.append_event(
        ts_ns=now - 1_000_000_000,
        alias="v1",
        kind="entry",
        question_idx=42,
        reason=None,
        payload_json='{"symbol":"@30"}',
    )
    dal.append_event(
        ts_ns=now - 500_000_000,
        alias="v1",
        kind="exit",
        question_idx=42,
        reason="exit_safety_d",
        payload_json='{"realized_pnl":5.0}',
    )

    snapshot = diag.build_snapshot(deploy_cfg, strategies)
    last = snapshot["slots"]["v1"]["last_decision"]

    assert last is not None
    assert last["kind"] == "exit"
    assert last["reason"] == "exit_safety_d"
    assert last["ts_ns"] == now - 500_000_000


# ---------------------------------------------------------------------------
# 13. --alias filter
# ---------------------------------------------------------------------------


def test_alias_filter_returns_only_requested_slot(tmp_path):
    diag = _import_diag()
    deploy_cfg = _make_deploy_config(tmp_path, ["v1", "v31"])
    strategies = StrategiesConfig(
        strategies=[
            _make_strategy_config("v1"),
            _make_strategy_config("v31", strategy_type="theta_harvester"),
        ]
    )

    _seed_dal(deploy_cfg, "v1")
    _seed_dal(deploy_cfg, "v31")

    snapshot = diag.build_snapshot(deploy_cfg, strategies, alias_filter="v1")
    assert "v1" in snapshot["slots"]
    assert "v31" not in snapshot["slots"]


# ---------------------------------------------------------------------------
# 14. JSON output: build_snapshot result is JSON-serialisable
# ---------------------------------------------------------------------------


def test_snapshot_is_json_serialisable(tmp_path):
    diag = _import_diag()
    deploy_cfg = _make_deploy_config(tmp_path, ["v1"])
    strategies = StrategiesConfig(strategies=[_make_strategy_config("v1")])
    _seed_dal(deploy_cfg, "v1")

    snapshot = diag.build_snapshot(deploy_cfg, strategies)
    # Should not raise
    serialised = json.dumps(snapshot)
    parsed = json.loads(serialised)
    assert "slots" in parsed


# ---------------------------------------------------------------------------
# 15. Multi-slot: both slots present
# ---------------------------------------------------------------------------


def test_multi_slot_both_present(tmp_path):
    diag = _import_diag()
    deploy_cfg = _make_deploy_config(tmp_path, ["v1", "v31"])
    strategies = StrategiesConfig(
        strategies=[
            _make_strategy_config("v1"),
            _make_strategy_config("v31", strategy_type="theta_harvester"),
        ]
    )

    _seed_dal(deploy_cfg, "v1")
    _seed_dal(deploy_cfg, "v31")

    snapshot = diag.build_snapshot(deploy_cfg, strategies)
    assert set(snapshot["slots"].keys()) == {"v1", "v31"}
