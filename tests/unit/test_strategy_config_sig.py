# tests/unit/test_strategy_config_sig.py
"""R5 — shared strategy_config_sig TDD tests.

Guards that:
1. strategy_config_sig is deterministic (same cfg → same sig, twice).
2. The sig is param-sensitive: mutating any of defaults, allowlist, global_,
   theta changes the hash.
3. Engine parity: _build_config_fingerprint(alias, cfg)["hash"] equals
   strategy_config_sig(cfg) for the same StrategyConfig.
4. Backtest report parity: for a slot loaded from config/strategy.yaml,
   the sig stamped in the report equals strategy_config_sig(cfg).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from hlanalysis.engine.config import (
    AllowlistEntry,
    GlobalRiskConfig,
    StrategyConfig,
    ThetaParams,
    load_strategies_config,
    strategy_config_sig,
)
from hlanalysis.engine.diag import _build_config_fingerprint


# ---------------------------------------------------------------------------
# Minimal fixture builders (mirrors test_engine_diag.py patterns)
# ---------------------------------------------------------------------------


def _minimal_allowlist_entry(**kwargs: Any) -> AllowlistEntry:
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


def _minimal_global_risk(**kwargs: Any) -> GlobalRiskConfig:
    defaults = dict(
        max_total_inventory_usd=500.0,
        max_concurrent_positions=5,
        daily_loss_cap_usd=200.0,
        max_strike_distance_pct=10.0,
        min_recent_volume_usd=1000.0,
        stale_data_halt_seconds=30,
        reconcile_interval_seconds=60,
    )
    defaults.update(kwargs)
    return GlobalRiskConfig(**defaults)


def _make_late_resolution_cfg(**kwargs: Any) -> StrategyConfig:
    """A minimal late_resolution StrategyConfig with no theta block."""
    base: dict[str, Any] = dict(
        name="late_resolution",
        paper_mode=True,
        account_alias="test_v1",
        strategy_type="late_resolution",
        allowlist=[_minimal_allowlist_entry()],
        defaults=_minimal_allowlist_entry(exit_safety_d=1.0),
        **{"global": _minimal_global_risk()},
    )
    base.update(kwargs)
    return StrategyConfig(**base)


def _make_theta_cfg(**kwargs: Any) -> StrategyConfig:
    """A minimal theta_harvester StrategyConfig."""
    base: dict[str, Any] = dict(
        name="theta_harvester",
        paper_mode=False,
        account_alias="test_v31",
        strategy_type="theta_harvester",
        allowlist=[_minimal_allowlist_entry()],
        defaults=_minimal_allowlist_entry(exit_safety_d=0.0),
        **{"global": _minimal_global_risk()},
        theta=ThetaParams(exit_safety_d=1.0, vol_estimator="bipower"),
    )
    base.update(kwargs)
    return StrategyConfig(**base)


# ---------------------------------------------------------------------------
# 1. Determinism
# ---------------------------------------------------------------------------


def test_strategy_config_sig_is_deterministic_late_resolution() -> None:
    cfg = _make_late_resolution_cfg()
    assert strategy_config_sig(cfg) == strategy_config_sig(cfg)


def test_strategy_config_sig_is_deterministic_theta() -> None:
    cfg = _make_theta_cfg()
    assert strategy_config_sig(cfg) == strategy_config_sig(cfg)


def test_strategy_config_sig_is_16_hex_chars() -> None:
    cfg = _make_late_resolution_cfg()
    sig = strategy_config_sig(cfg)
    assert len(sig) == 16
    assert all(c in "0123456789abcdef" for c in sig)


# ---------------------------------------------------------------------------
# 2. Param-sensitivity
# ---------------------------------------------------------------------------


def test_sig_changes_when_defaults_field_changes() -> None:
    cfg_a = _make_late_resolution_cfg()
    # Mutate a defaults field — pydantic models are frozen so we build a new one.
    new_defaults = _minimal_allowlist_entry(exit_safety_d=0.5)  # was 1.0
    # StrategyConfig is frozen; build a new one with altered defaults.
    cfg_b = StrategyConfig(
        name=cfg_a.name,
        paper_mode=cfg_a.paper_mode,
        account_alias=cfg_a.account_alias,
        strategy_type=cfg_a.strategy_type,
        allowlist=list(cfg_a.allowlist),
        defaults=new_defaults,
        **{"global": cfg_a.global_},
    )
    assert strategy_config_sig(cfg_a) != strategy_config_sig(cfg_b), "sig must change when a defaults field changes"


def test_sig_changes_when_allowlist_entry_changes() -> None:
    cfg_a = _make_late_resolution_cfg()
    new_entry = _minimal_allowlist_entry(max_position_usd=999.0)  # was 100
    cfg_b = StrategyConfig(
        name=cfg_a.name,
        paper_mode=cfg_a.paper_mode,
        account_alias=cfg_a.account_alias,
        strategy_type=cfg_a.strategy_type,
        allowlist=[new_entry],
        defaults=cfg_a.defaults,
        **{"global": cfg_a.global_},
    )
    assert strategy_config_sig(cfg_a) != strategy_config_sig(cfg_b), "sig must change when an allowlist entry changes"


def test_sig_changes_when_global_risk_field_changes() -> None:
    cfg_a = _make_late_resolution_cfg()
    new_global = _minimal_global_risk(daily_loss_cap_usd=9999.0)  # was 200
    cfg_b = StrategyConfig(
        name=cfg_a.name,
        paper_mode=cfg_a.paper_mode,
        account_alias=cfg_a.account_alias,
        strategy_type=cfg_a.strategy_type,
        allowlist=list(cfg_a.allowlist),
        defaults=cfg_a.defaults,
        **{"global": new_global},
    )
    assert strategy_config_sig(cfg_a) != strategy_config_sig(cfg_b), "sig must change when a global_ field changes"


def test_sig_changes_when_theta_field_changes() -> None:
    cfg_a = _make_theta_cfg()
    # Change a theta field: exit_safety_d 1.0 → 0.5
    new_theta = ThetaParams(exit_safety_d=0.5, vol_estimator="bipower")
    cfg_b = StrategyConfig(
        name=cfg_a.name,
        paper_mode=cfg_a.paper_mode,
        account_alias=cfg_a.account_alias,
        strategy_type=cfg_a.strategy_type,
        allowlist=list(cfg_a.allowlist),
        defaults=cfg_a.defaults,
        **{"global": cfg_a.global_},
        theta=new_theta,
    )
    assert strategy_config_sig(cfg_a) != strategy_config_sig(cfg_b), "sig must change when a theta field changes"


def test_sig_changes_when_theta_added_vs_absent() -> None:
    cfg_no_theta = _make_late_resolution_cfg()
    cfg_with_theta = StrategyConfig(
        name=cfg_no_theta.name,
        paper_mode=cfg_no_theta.paper_mode,
        account_alias=cfg_no_theta.account_alias,
        strategy_type="theta_harvester",
        allowlist=list(cfg_no_theta.allowlist),
        defaults=cfg_no_theta.defaults,
        **{"global": cfg_no_theta.global_},
        theta=ThetaParams(),
    )
    assert strategy_config_sig(cfg_no_theta) != strategy_config_sig(cfg_with_theta), (
        "sig must differ when theta block is present vs absent"
    )


# ---------------------------------------------------------------------------
# 3. Engine parity: strategy_config_sig == _build_config_fingerprint["hash"]
# ---------------------------------------------------------------------------


def test_engine_parity_late_resolution() -> None:
    """strategy_config_sig must equal _build_config_fingerprint[\"hash\"] for a
    late_resolution slot."""
    cfg = _make_late_resolution_cfg()
    fp = _build_config_fingerprint("test_v1", cfg)
    assert fp["hash"] == strategy_config_sig(cfg), (
        f"engine fingerprint hash {fp['hash']!r} != strategy_config_sig {strategy_config_sig(cfg)!r}"
    )


def test_engine_parity_theta() -> None:
    """strategy_config_sig must equal _build_config_fingerprint[\"hash\"] for a
    theta_harvester slot."""
    cfg = _make_theta_cfg()
    fp = _build_config_fingerprint("test_v31", cfg)
    assert fp["hash"] == strategy_config_sig(cfg), (
        f"engine fingerprint hash {fp['hash']!r} != strategy_config_sig {strategy_config_sig(cfg)!r}"
    )


def test_engine_parity_v31_from_live_yaml() -> None:
    """For the live v31 slot, engine fingerprint hash must equal strategy_config_sig.

    This is the key parity test: engine diag and backtest --slot must produce
    the SAME hash for the same slot config. A mismatch means 'is this sim
    comparable to the live slot?' cannot be answered.
    """
    cfgs = load_strategies_config(Path("config/strategy.yaml"))
    by_alias = {c.account_alias: c for c in cfgs.strategies}
    assert "v31" in by_alias, "expected v31 slot in config/strategy.yaml for parity test"
    cfg = by_alias["v31"]
    fp = _build_config_fingerprint("v31", cfg)
    sig = strategy_config_sig(cfg)
    assert fp["hash"] == sig, (
        f"v31 engine hash {fp['hash']!r} != strategy_config_sig {sig!r}; "
        "engine diag and backtest --slot would show different hashes for the same slot"
    )


# ---------------------------------------------------------------------------
# 4. Backtest report parity: the slot_config_sig line in report.md matches
#    strategy_config_sig(cfg) for the same slot.
# ---------------------------------------------------------------------------


def test_report_writes_slot_config_sig_for_slot_run(tmp_path: Path) -> None:
    """write_single_run_report must stamp a slot_config_sig line when
    slot_strategy_cfg is provided, and that line's hex must equal
    strategy_config_sig(cfg)."""
    from hlanalysis.backtest.report import write_single_run_report
    from hlanalysis.backtest.runner.result import RunSummary

    cfg = _make_theta_cfg()
    expected_sig = strategy_config_sig(cfg)

    summary = RunSummary(
        n_markets=1,
        n_trades=0,
        total_pnl_usd=0.0,
        sharpe=0.0,
        hit_rate=0.0,
        max_drawdown_usd=0.0,
    )

    report_path = write_single_run_report(
        out_dir=tmp_path,
        strategy_name="v3_theta_harvester",
        config_summary={"some": "params"},
        summary=summary,
        descriptors=[],
        per_question_pnl=[],
        outcomes=[],
        slot_strategy_cfg=cfg,
    )

    text = report_path.read_text()
    assert f"`{expected_sig}`" in text, f"expected slot_config_sig `{expected_sig}` in report.md, got:\n{text}"
    assert "slot config_sig" in text, "expected 'slot config_sig' label in report.md"


def test_report_omits_slot_config_sig_when_no_slot_cfg(tmp_path: Path) -> None:
    """write_single_run_report must NOT emit a slot_config_sig line when no
    slot_strategy_cfg is provided (non-slot --strategy/--config run)."""
    from hlanalysis.backtest.report import write_single_run_report
    from hlanalysis.backtest.runner.result import RunSummary

    summary = RunSummary(
        n_markets=0,
        n_trades=0,
        total_pnl_usd=0.0,
        sharpe=0.0,
        hit_rate=0.0,
        max_drawdown_usd=0.0,
    )

    report_path = write_single_run_report(
        out_dir=tmp_path,
        strategy_name="v3_theta_harvester",
        config_summary={"some": "params"},
        summary=summary,
        descriptors=[],
        per_question_pnl=[],
        outcomes=[],
        # slot_strategy_cfg omitted — non-slot run
    )

    text = report_path.read_text()
    assert "slot config_sig" not in text, "expected no slot_config_sig in report.md for non-slot run"
