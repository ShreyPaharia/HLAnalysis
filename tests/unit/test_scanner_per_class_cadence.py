# tests/unit/test_scanner_per_class_cadence.py
"""Scanner reads σ-history at the per-question-class cadence: a v31 slot with a
priceBucket theta_override of vol_sampling_dt_seconds=2 must read the dt=2 bar
series for bucket questions while priceBinary reads the slot default dt=5.

Default path (no dt override) stays bit-identical: dt-less read, legacy n."""
from __future__ import annotations

from hlanalysis.engine.config import (
    AllowlistEntry, GlobalRiskConfig, StrategyConfig, ThetaParams,
)
from hlanalysis.engine.scanner import Scanner


def _global() -> GlobalRiskConfig:
    return GlobalRiskConfig(
        max_total_inventory_usd=1100, max_concurrent_positions=5,
        daily_loss_cap_usd=100, max_strike_distance_pct=50,
        min_recent_volume_usd=100, stale_data_halt_seconds=30,
        reconcile_interval_seconds=15,
    )


def _entry(klass: str) -> AllowlistEntry:
    return AllowlistEntry(
        match={"class": klass, "underlying": "BTC"}, max_position_usd=500,
        stop_loss_pct=None, tte_min_seconds=0, tte_max_seconds=43200,
        price_extreme_threshold=0.0, distance_from_strike_usd_min=0, vol_max=100,
    )


def _cfg(theta_overrides: dict | None = None) -> StrategyConfig:
    defaults = AllowlistEntry(
        match={}, max_position_usd=500, stop_loss_pct=None, tte_min_seconds=0,
        tte_max_seconds=43200, price_extreme_threshold=0.0,
        distance_from_strike_usd_min=0, vol_max=100,
    )
    kwargs: dict = dict(
        name="theta_harvester", account_alias="v31", paper_mode=False,
        strategy_type="theta_harvester",
        allowlist=[_entry("priceBinary"), _entry("priceBucket")],
        blocklist_question_idxs=[], defaults=defaults,
        theta=ThetaParams(vol_lookback_seconds=3600, vol_sampling_dt_seconds=5),
        **{"global": _global()},
    )
    if theta_overrides is not None:
        kwargs["theta_overrides"] = theta_overrides
    return StrategyConfig(**kwargs)


def test_cadence_by_class_only_contains_dt_overriding_classes() -> None:
    cfg = _cfg(theta_overrides={"priceBucket": {"vol_sampling_dt_seconds": 2}})
    m = Scanner.cadence_by_class(cfg)
    assert set(m) == {"priceBucket"}
    assert m["priceBucket"][0] == 2                       # (dt_seconds, n_bars)
    assert m["priceBucket"][1] > Scanner._required_returns_n(cfg)


def test_non_dt_override_creates_no_cadence_entry() -> None:
    cfg = _cfg(theta_overrides={"priceBucket": {"favorite_threshold": 0.80}})
    assert Scanner.cadence_by_class(cfg) == {}


def test_no_overrides_empty_cadence_map() -> None:
    assert Scanner.cadence_by_class(_cfg()) == {}


def test_default_returns_n_unchanged() -> None:
    assert Scanner._required_returns_n(_cfg()) == 720  # ceil(3600/5)
