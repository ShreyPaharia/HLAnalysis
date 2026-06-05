# tests/unit/test_theta_per_class_overrides.py
"""Per-question.klass theta overrides for the theta_harvester strategy.

The independent HL bucket tune (summeries/v31_bucket_independent_tune_2026_06_05)
proved priceBucket wants the OPPOSITE of priceBinary on several theta axes
(favorite_threshold, vol_lookback_seconds, exit_safety_d, edge_buffer). Today
those knobs live in the shared `theta:` block and cannot diverge by class.

These tests pin the new `theta_overrides:` path:
  * no override block  → bit-identical to today (empty by-class map)
  * a priceBucket override applies to buckets while priceBinary keeps defaults
  * resolution order is per-class override > instance theta defaults
  * a partial override only touches the fields the operator explicitly set
    (pydantic model_fields_set) — even when the value equals the dataclass default
  * vol_sampling_dt_seconds is shared-feed-coupled and rejected per-class
  * unknown knobs still fail loud (extra='forbid' inherited from ThetaParams)
"""
from __future__ import annotations

import dataclasses
import textwrap
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from hlanalysis.engine.config import (
    AllowlistEntry,
    GlobalRiskConfig,
    StrategyConfig,
    ThetaParams,
    load_strategies_config,
)
from hlanalysis.engine.runtime import (
    build_theta_harvester_config,
    build_theta_harvester_configs_by_class,
    reference_vol_lookback_seconds,
)
from hlanalysis.strategy.theta_harvester import ThetaHarvesterStrategy
from hlanalysis.strategy.types import BookState, QuestionView


def _global() -> GlobalRiskConfig:
    return GlobalRiskConfig(
        max_total_inventory_usd=1100, max_concurrent_positions=5,
        daily_loss_cap_usd=100, max_strike_distance_pct=50,
        min_recent_volume_usd=100, stale_data_halt_seconds=30,
        reconcile_interval_seconds=15,
    )


def _entry(klass: str) -> AllowlistEntry:
    return AllowlistEntry(
        match={"class": klass, "underlying": "BTC"},
        max_position_usd=500, stop_loss_pct=None, tte_min_seconds=0,
        tte_max_seconds=43200, price_extreme_threshold=0.0,
        distance_from_strike_usd_min=0, vol_max=100,
    )


def _theta_cfg(theta_overrides: dict | None = None) -> StrategyConfig:
    """A v31-like theta_harvester slot with binary+bucket allowlist entries and
    a shared `theta:` block (favorite_threshold=0.85, exit_safety_d=1.0)."""
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
        theta=ThetaParams(
            vol_lookback_seconds=3600, vol_sampling_dt_seconds=5,
            favorite_threshold=0.85, exit_safety_d=1.0, edge_buffer=0.02,
        ),
        **{"global": _global()},
    )
    if theta_overrides is not None:
        kwargs["theta_overrides"] = theta_overrides
    return StrategyConfig(**kwargs)


def _question(klass: str) -> QuestionView:
    return QuestionView(
        question_idx=1, yes_symbol="", no_symbol="", strike=100_000.0,
        expiry_ns=10**18, underlying="BTC", klass=klass, period="24h",
    )


# --- no override → bit-identical to today -----------------------------------

def test_no_overrides_yields_empty_by_class_map() -> None:
    """With no `theta_overrides:` block the by-class map is empty, so the
    strategy falls through to the single default config for every class —
    bit-identical to today's behavior."""
    cfg = _theta_cfg()
    assert build_theta_harvester_configs_by_class(cfg) == {}


def test_live_strategy_yaml_has_no_overrides_today() -> None:
    """Guard: the shipped config introduces no per-class divergence yet (this is
    plumbing only). Every theta slot must build an empty by-class map so live
    behavior is unchanged until the operator flips values."""
    cfgs = load_strategies_config(Path("config/strategy.yaml"))
    theta_slots = [c for c in cfgs.strategies if c.strategy_type == "theta_harvester"]
    assert theta_slots
    for c in theta_slots:
        assert build_theta_harvester_configs_by_class(c) == {}


# --- per-class override applies, other classes keep defaults -----------------

def test_bucket_override_applies_binary_keeps_default() -> None:
    """A priceBucket favorite_threshold=0.80 override applies to buckets while
    priceBinary keeps the instance default 0.85 — in the same instance."""
    cfg = _theta_cfg(theta_overrides={"priceBucket": {"favorite_threshold": 0.80}})
    by_class = build_theta_harvester_configs_by_class(cfg)
    base = build_theta_harvester_config(cfg)

    assert by_class["priceBucket"].favorite_threshold == 0.80
    assert base.favorite_threshold == 0.85  # binary (default) unchanged
    assert "priceBinary" not in by_class      # falls through to base


def test_override_only_touches_set_fields() -> None:
    """Resolution order = per-class override > instance theta defaults. Fields
    the override does NOT set keep the instance theta value verbatim."""
    cfg = _theta_cfg(theta_overrides={"priceBucket": {
        "favorite_threshold": 0.80,
        "vol_lookback_seconds": 2700,
        "edge_buffer": 0.005,
    }})
    base = build_theta_harvester_config(cfg)
    bucket = build_theta_harvester_configs_by_class(cfg)["priceBucket"]

    # overridden
    assert bucket.favorite_threshold == 0.80
    assert bucket.vol_lookback_seconds == 2700
    assert bucket.edge_buffer == 0.005
    # NOT overridden → identical to base on every other field
    changed = {"favorite_threshold", "vol_lookback_seconds", "edge_buffer"}
    for f in dataclasses.fields(base):
        if f.name in changed:
            continue
        assert getattr(bucket, f.name) == getattr(base, f.name), f.name


def test_override_to_dataclass_default_value_still_applies() -> None:
    """exit_safety_d=0.0 is both the dataclass default AND a meaningful bucket
    target. An explicit override to 0.0 must win over the instance theta 1.0 —
    proving we apply model_fields_set, not a default-diff."""
    cfg = _theta_cfg(theta_overrides={"priceBucket": {"exit_safety_d": 0.0}})
    base = build_theta_harvester_config(cfg)
    bucket = build_theta_harvester_configs_by_class(cfg)["priceBucket"]

    assert base.exit_safety_d == 1.0       # instance default
    assert bucket.exit_safety_d == 0.0     # explicit override wins


# --- strategy wiring: the swap selects per-class config ----------------------

def test_strategy_cfg_for_selects_per_class() -> None:
    """ThetaHarvesterStrategy resolves the per-class config by question.klass,
    falling back to the default for unmapped classes."""
    cfg = _theta_cfg(theta_overrides={"priceBucket": {"favorite_threshold": 0.80}})
    strat = ThetaHarvesterStrategy(
        build_theta_harvester_config(cfg),
        cfg_by_class=build_theta_harvester_configs_by_class(cfg),
    )
    assert strat._cfg_for(_question("priceBucket")).favorite_threshold == 0.80
    assert strat._cfg_for(_question("priceBinary")).favorite_threshold == 0.85


def test_strategy_restores_cfg_after_evaluate() -> None:
    """The per-class swap inside evaluate must restore self.cfg afterwards so
    external readers (diagnostics, tests) see the stable default."""
    cfg = _theta_cfg(theta_overrides={"priceBucket": {"favorite_threshold": 0.80}})
    base = build_theta_harvester_config(cfg)
    strat = ThetaHarvesterStrategy(
        base, cfg_by_class=build_theta_harvester_configs_by_class(cfg),
    )
    # A settled bucket question returns immediately but still runs the swap.
    q = _question("priceBucket")
    q = dataclasses.replace(q, settled=True)
    strat.evaluate(
        question=q, books={}, reference_price=100_000.0, recent_returns=(),
        recent_volume_usd=0.0, position=None, now_ns=10**17,
    )
    assert strat.cfg is base
    assert strat.cfg.favorite_threshold == 0.85


# --- guards ------------------------------------------------------------------

def test_vol_sampling_dt_override_rejected() -> None:
    """vol_sampling_dt_seconds couples to the shared reference-feed bucketing
    (all slots on a symbol move in lockstep). It MUST NOT diverge per class."""
    cfg = _theta_cfg(theta_overrides={"priceBucket": {"vol_sampling_dt_seconds": 2}})
    with pytest.raises(ValueError, match="vol_sampling_dt_seconds"):
        build_theta_harvester_configs_by_class(cfg)


def test_unknown_override_knob_fails_loud() -> None:
    """A typo'd knob in a per-class override fails at load (extra='forbid'),
    never silently dropped — same strictness as the shared theta block."""
    with pytest.raises(ValidationError):
        _theta_cfg(theta_overrides={"priceBucket": {"not_a_real_knob": 1.23}})


# --- bit-identical replay: overrides for one class don't perturb another -----

def _book(symbol: str, *, bid: float, ask: float) -> BookState:
    return BookState(
        symbol=symbol, bid_px=bid, bid_sz=100.0, ask_px=ask, ask_sz=100.0,
        last_trade_ts_ns=0, last_l2_ts_ns=0,
    )


def _binary_question() -> QuestionView:
    return QuestionView(
        question_idx=0, yes_symbol="YES", no_symbol="NO", strike=100_000.0,
        expiry_ns=3600 * 10**9, underlying="BTC", klass="priceBinary", period="1d",
    )


def test_binary_evaluate_bit_identical_with_and_without_bucket_override() -> None:
    """A priceBucket override must not change a priceBinary evaluation. The
    Decision built with a bucket override present is byte-for-byte identical to
    the Decision built with no overrides at all (legacy path)."""
    cfg_plain = _theta_cfg()
    cfg_over = _theta_cfg(theta_overrides={"priceBucket": {
        "favorite_threshold": 0.80, "exit_safety_d": 0.0, "edge_buffer": 0.005,
    }})
    base_plain = build_theta_harvester_config(cfg_plain)
    base_over = build_theta_harvester_config(cfg_over)

    strat_plain = ThetaHarvesterStrategy(base_plain)  # legacy 1-arg construction
    strat_over = ThetaHarvesterStrategy(
        base_over, cfg_by_class=build_theta_harvester_configs_by_class(cfg_over),
    )

    qv = _binary_question()
    books = {
        "YES": _book("YES", bid=0.49, ask=0.50),
        "NO": _book("NO", bid=0.49, ask=0.50),
    }
    rets = tuple([0.0001] * 120)
    call = dict(
        question=qv, books=books, reference_price=120_000.0, recent_returns=rets,
        recent_volume_usd=1000.0, position=None, now_ns=0,
    )
    d_plain = strat_plain.evaluate(**call)
    d_over = strat_over.evaluate(**call)

    assert d_plain == d_over


def test_reference_vol_lookback_accounts_for_overrides() -> None:
    """MarketState history sizing must consider per-class override lookbacks so
    a class requesting a longer σ window isn't truncated."""
    cfg = _theta_cfg(theta_overrides={"priceBucket": {"vol_lookback_seconds": 9000}})
    assert reference_vol_lookback_seconds(cfg) >= 9000


# --- YAML round-trip ---------------------------------------------------------

_YAML = textwrap.dedent("""
    name: theta_harvester
    account_alias: v31
    strategy_type: theta_harvester
    paper_mode: false
    allowlist:
      - match: {class: priceBinary, underlying: BTC}
        max_position_usd: 500
        stop_loss_pct: null
        tte_min_seconds: 0
        tte_max_seconds: 43200
        price_extreme_threshold: 0.0
        distance_from_strike_usd_min: 0
        vol_max: 100
      - match: {class: priceBucket, underlying: BTC}
        max_position_usd: 500
        stop_loss_pct: null
        tte_min_seconds: 0
        tte_max_seconds: 28800
        price_extreme_threshold: 0.0
        distance_from_strike_usd_min: 0
        vol_max: 100
    blocklist_question_idxs: []
    defaults:
      match: {}
      max_position_usd: 500
      stop_loss_pct: null
      tte_min_seconds: 0
      tte_max_seconds: 43200
      price_extreme_threshold: 0.0
      distance_from_strike_usd_min: 0
      vol_max: 100
    global:
      max_total_inventory_usd: 1100
      max_concurrent_positions: 5
      daily_loss_cap_usd: 100
      max_strike_distance_pct: 50
      min_recent_volume_usd: 100
      stale_data_halt_seconds: 30
      reconcile_interval_seconds: 15
    theta:
      vol_lookback_seconds: 3600
      vol_sampling_dt_seconds: 5
      favorite_threshold: 0.85
      edge_buffer: 0.02
      exit_safety_d: 1.0
    theta_overrides:
      priceBucket:
        favorite_threshold: 0.80
        vol_lookback_seconds: 2700
        exit_safety_d: 0.0
        edge_buffer: 0.005
""")


def test_yaml_round_trip_per_class_theta_override() -> None:
    """End-to-end YAML acceptance: a `theta_overrides:` sub-block parses and
    builds a divergent bucket config (the target HL bucket tune) while binary
    keeps the shared theta defaults."""
    cfg = StrategyConfig(**yaml.safe_load(_YAML))
    base = build_theta_harvester_config(cfg)
    bucket = build_theta_harvester_configs_by_class(cfg)["priceBucket"]

    # binary keeps shared theta defaults
    assert base.favorite_threshold == 0.85
    assert base.vol_lookback_seconds == 3600
    assert base.exit_safety_d == 1.0
    assert base.edge_buffer == 0.02
    # bucket diverges to the independent-tune target
    assert bucket.favorite_threshold == 0.80
    assert bucket.vol_lookback_seconds == 2700
    assert bucket.exit_safety_d == 0.0
    assert bucket.edge_buffer == 0.005
    # shared cadence unchanged on both
    assert bucket.vol_sampling_dt_seconds == base.vol_sampling_dt_seconds == 5
