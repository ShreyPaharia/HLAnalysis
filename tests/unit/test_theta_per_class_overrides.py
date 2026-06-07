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


def test_live_strategy_yaml_bucket_override_matches_tune() -> None:
    """Guard the shipped HL v31 per-class config (C3, 2026-06-06 overfit
    rollback): priceBucket keeps only the mechanical σ/timing tilt
    (vol_lookback 2700 / dt 2; tte 8h on the allowlist) and RESTORES the three
    risk gates to the binary baseline (fav 0.85 / eb 0.02 / esd 1.0). The
    2026-06-05 "best sim" (fav0.80/eb0.005/esd0) was overfit to a zero-loss
    sample — a loss-injection stress collapsed it to ~$0 expected / 50% chance of
    a net loss; C3 is strictly better risk-adjusted. priceBinary keeps the shared
    theta defaults EXCEPT vol_lookback_seconds=900 (binary @dt5). The PM (v31_pm)
    slot carries NO per-class override.

    If someone edits config/strategy.yaml's theta config, this pins the
    intended live values so a typo/regression fails loudly."""
    cfgs = load_strategies_config(Path("config/strategy.yaml"))
    theta = {c.account_alias: c for c in cfgs.strategies
             if c.strategy_type == "theta_harvester"}

    v31 = theta["v31"]
    base = build_theta_harvester_config(v31)
    bucket = build_theta_harvester_configs_by_class(v31)["priceBucket"]
    # bucket: mechanical tilt only; risk gates restored to the binary baseline
    assert bucket.vol_lookback_seconds == 2700
    assert bucket.vol_sampling_dt_seconds == 2
    assert bucket.favorite_threshold == 0.85   # restored (was 0.80 overfit)
    assert bucket.edge_buffer == 0.02          # restored (was 0.005 overfit)
    assert bucket.exit_safety_d == 1.0         # restored (was 0.0 overfit)
    # binary (the instance default): only vol_lookback diverges to the binary
    # tune (3600→900 @dt5); fav/dt/esd/eb stay at prod (eb=0 does not stack).
    assert base.favorite_threshold == 0.85
    assert base.vol_lookback_seconds == 900
    assert base.vol_sampling_dt_seconds == 5
    assert base.exit_safety_d == 1.0
    assert base.edge_buffer == 0.02
    # only priceBucket diverges; binary falls through to the default
    assert set(build_theta_harvester_configs_by_class(v31)) == {"priceBucket"}

    # v31_pm now folds in BTC multi-strike buckets → carries a priceBucket
    # override (PR #12 tuned cell). v1_pm (late_resolution) has none.
    assert set(build_theta_harvester_configs_by_class(theta["v31_pm"])) == {"priceBucket"}
    pm_bucket = build_theta_harvester_configs_by_class(theta["v31_pm"])["priceBucket"]
    assert pm_bucket.favorite_threshold == 0.75
    assert pm_bucket.vol_lookback_seconds == 1800
    assert pm_bucket.exit_safety_d == 0.5


def test_bucket_override_risk_gates_not_below_binary() -> None:
    """Guardrail (2026-06-06): a per-class override must NEVER loosen a defensive
    risk gate below the binary baseline. The 2026-06-05 bucket tune did exactly
    that (favorite_threshold 0.85→0.80, edge_buffer 0.02→0.005, exit_safety_d
    1.0→0.0) and a loss-injection stress showed it was overfit to a tailless
    sample. A looser gate is only justified by out-of-sample evidence with
    adverse settlements; until then, gates track binary. This makes "kill the
    safety gate because the backtest said so" fail CI.

    Higher = more protective for all three: favorite_threshold (more extreme
    favorites only), edge_buffer (wider entry margin), exit_safety_d (mid-hold
    stop-out engaged).

    EXEMPTIONS (`_RISK_GATE_EXEMPT`): a per-class override is exempt only when it
    is a SEPARATELY-VALIDATED tune for a market that is genuinely a different
    instrument than the slot's binary leg, so "track the binary baseline" does
    not apply. The guardrail's premise — bucket and binary are the same
    underlying daily market (HL HIP-4) — does not hold there.
      - (v31_pm, priceBucket): v31_pm trades BTC up/down DAILY binaries AND BTC
        multi-strike WEEKLY buckets (folded onto one wallet). Different
        instrument + horizon; the bucket cell is PR #12's walk-forward tune
        (fav0.75/eb0.02/vlb1800/esd0.5; +$870, worst split −$24, maxDD $49,
        ~1100 trades / 18 splits) — OOS evidence with non-zero DDs, which is
        exactly the justification this guardrail's docstring demands. Operator-
        approved 2026-06-08. HL v31/priceBucket is NOT exempt (must still track
        its binary baseline — the original overfit case)."""
    # (alias, klass) cells exempt from the not-below-binary check; see docstring.
    _RISK_GATE_EXEMPT = {("v31_pm", "priceBucket")}
    cfgs = load_strategies_config(Path("config/strategy.yaml"))
    theta = {c.account_alias: c for c in cfgs.strategies
             if c.strategy_type == "theta_harvester"}
    for alias, cfg in theta.items():
        base = build_theta_harvester_config(cfg)
        for klass, override in build_theta_harvester_configs_by_class(cfg).items():
            if (alias, klass) in _RISK_GATE_EXEMPT:
                continue
            assert override.favorite_threshold >= base.favorite_threshold, (
                f"{alias}/{klass} favorite_threshold "
                f"{override.favorite_threshold} < binary {base.favorite_threshold}")
            assert override.edge_buffer >= base.edge_buffer, (
                f"{alias}/{klass} edge_buffer "
                f"{override.edge_buffer} < binary {base.edge_buffer}")
            assert override.exit_safety_d >= base.exit_safety_d, (
                f"{alias}/{klass} exit_safety_d "
                f"{override.exit_safety_d} < binary {base.exit_safety_d}")


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

def test_vol_sampling_dt_override_now_allowed() -> None:
    """Post (symbol, dt) refactor: a per-class vol_sampling_dt_seconds override
    is legal — the engine maintains an independent bar series per cadence, so the
    σ sampling cadence can diverge by question class."""
    cfg = _theta_cfg(theta_overrides={"priceBucket": {"vol_sampling_dt_seconds": 2}})
    by_class = build_theta_harvester_configs_by_class(cfg)
    assert by_class["priceBucket"].vol_sampling_dt_seconds == 2
    base = build_theta_harvester_config(cfg)
    assert base.vol_sampling_dt_seconds == 5  # binary keeps the shared default


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
