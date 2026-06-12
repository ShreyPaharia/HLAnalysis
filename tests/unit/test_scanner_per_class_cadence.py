# tests/unit/test_scanner_per_class_cadence.py
"""Scanner reads σ-history at the per-question-class cadence: a v31 slot with a
priceBucket theta_override of vol_sampling_dt_seconds=2 must read the dt=2 bar
series for bucket questions while priceBinary reads the slot default dt=5.

Default path (no dt override) stays bit-identical: dt-less read, legacy n."""
from __future__ import annotations

from pathlib import Path

import pytest

from hlanalysis.engine.config import (
    AllowlistEntry, GlobalRiskConfig, StrategyConfig, ThetaParams,
)
from hlanalysis.engine.market_state import MarketState
from hlanalysis.engine.scanner import Scanner
from hlanalysis.engine.state import StateDAL
from hlanalysis.events import (
    BboEvent, MarkEvent, Mechanism, ProductType, QuestionMetaEvent,
)
from hlanalysis.strategy.late_resolution import (
    LateResolutionConfig, LateResolutionStrategy,
)
from hlanalysis.strategy.types import Action, Decision


def _global() -> GlobalRiskConfig:
    return GlobalRiskConfig(
        max_total_inventory_usd=1100, max_concurrent_positions=5,
        daily_loss_cap_usd=100, max_strike_distance_pct=50,
        min_recent_volume_usd=0, stale_data_halt_seconds=30,
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


# --- Fix 2: per-class lookback sizing ---


def test_cadence_by_class_widens_n_for_per_class_vol_lookback() -> None:
    """A class override that ALSO sets vol_lookback_seconds must produce n_bars
    sized for the LARGER of the base lookback and the per-class override,
    not silently truncated to the base window."""
    # Base slot: 3600s lookback at dt=5 → base_secs=3600, base_n=720.
    # priceBucket override: dt=2, vol_lookback_seconds=7200
    # Expected n = ceil(7200/2) = 3600 > base_n=720.
    cfg = _cfg(theta_overrides={"priceBucket": {
        "vol_sampling_dt_seconds": 2,
        "vol_lookback_seconds": 7200,
    }})
    m = Scanner.cadence_by_class(cfg)
    assert m["priceBucket"][0] == 2
    # n must cover the 7200s window at dt=2 → 3600 bars
    assert m["priceBucket"][1] >= 3600, (
        f"expected n>=3600 to cover 7200s at dt=2, got {m['priceBucket'][1]}"
    )


def test_cadence_by_class_widens_n_for_per_class_drift_lookback() -> None:
    """A class override that sets drift_lookback_seconds beyond the base
    lookback must also widen n so the drift window is not truncated."""
    cfg = _cfg(theta_overrides={"priceBucket": {
        "vol_sampling_dt_seconds": 2,
        "drift_lookback_seconds": 5400,
    }})
    m = Scanner.cadence_by_class(cfg)
    # Base secs=3600, drift override=5400 → secs=5400; n=ceil(5400/2)=2700
    assert m["priceBucket"][1] >= 2700, (
        f"expected n>=2700 to cover 5400s drift at dt=2, got {m['priceBucket'][1]}"
    )


def test_cadence_by_class_base_lookback_not_inflated() -> None:
    """_lookback_secs (and therefore _required_returns_n) MUST NOT be affected
    by per-class vol/drift overrides — the default-path n must stay bit-identical.
    """
    cfg_no_override = _cfg()
    cfg_with_wide = _cfg(theta_overrides={"priceBucket": {
        "vol_sampling_dt_seconds": 2,
        "vol_lookback_seconds": 9000,
    }})
    assert Scanner._required_returns_n(cfg_no_override) == Scanner._required_returns_n(cfg_with_wide), (
        "per-class override must NOT inflate _required_returns_n (the default-path n)"
    )


# --- Fix 3: scan()-level wiring test ---


class _RecordingMarketState(MarketState):
    """Engine MarketState sub-type kept for market-seeding; the scanner now
    calls ``build_decision_inputs(self.ms._core, ...)`` directly (R1), so
    interception is done at the ``build_decision_inputs`` level in the test
    rather than overriding the wrapper's ``recent_returns`` / ``recent_hl_bars``
    methods.  The lists below are populated by the monkeypatch in
    ``test_scan_wiring_bucket_uses_dt2_binary_uses_default``."""

    def __init__(self) -> None:
        super().__init__()
        # list of (dt,) in call order — one entry per scan()-level evaluation
        # populated by the monkeypatch in the test function below (not by
        # overriding recent_returns, which is no longer called by the scanner
        # since R1 routes through build_decision_inputs directly).
        self.recent_returns_calls: list[tuple[int, int | None]] = []
        self.recent_hl_bars_calls: list[tuple[int, int | None]] = []


class _ClassRecordingStrategy(LateResolutionStrategy):
    """Records (question.klass, received recent_returns, received recent_hl_bars)
    for each evaluate() call so the test can map class → dt used."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.calls: list[tuple[str, tuple, tuple]] = []

    def evaluate(self, *, question, recent_returns=(), recent_hl_bars=(), **kw):
        self.calls.append((question.klass, recent_returns, recent_hl_bars))
        return super().evaluate(
            question=question,
            recent_returns=recent_returns,
            recent_hl_bars=recent_hl_bars,
            **kw,
        )


def _seed_two_class_market(now_ns: int, ms: MarketState) -> None:
    """Seed a priceBinary question (qidx=1) and a priceBucket question (qidx=2)
    into ``ms``, plus enough BTC marks at both dt=5s and dt=2s to produce bars."""
    from datetime import datetime, timezone

    expiry_str = datetime.fromtimestamp(
        (now_ns + 10 * 60 * 1_000_000_000) / 1e9, tz=timezone.utc
    ).strftime("%Y%m%d-%H%M")

    # priceBinary question — legs #30 / #31
    ms.apply(QuestionMetaEvent(
        venue="hyperliquid", product_type=ProductType.PREDICTION_BINARY,
        mechanism=Mechanism.CLOB, symbol="qmeta",
        exchange_ts=now_ns - 120_000_000_000,
        local_recv_ts=now_ns - 120_000_000_000,
        question_idx=1, named_outcome_idxs=[3],
        keys=["class", "underlying", "period", "expiry", "strike"],
        values=["priceBinary", "BTC", "1h", expiry_str, "80000"],
    ))
    # priceBucket question — 2 outcomes → legs #00, #01, #10, #11
    ms.apply(QuestionMetaEvent(
        venue="hyperliquid", product_type=ProductType.PREDICTION_CATEGORICAL,
        mechanism=Mechanism.CLOB, symbol="qmeta_bucket",
        exchange_ts=now_ns - 120_000_000_000,
        local_recv_ts=now_ns - 120_000_000_000,
        question_idx=2, named_outcome_idxs=[0, 1],
        keys=["class", "underlying", "period", "expiry", "strike"],
        values=["priceBucket", "BTC", "1d", expiry_str, "80000"],
    ))

    # BTC marks: 40 ticks spaced 2s apart — covers both dt=5s and dt=2s series
    for i in range(40):
        ts = now_ns - (40 - i) * 2_000_000_000
        ms.apply(MarkEvent(
            venue="hyperliquid", product_type=ProductType.PERP,
            mechanism=Mechanism.CLOB, symbol="BTC",
            exchange_ts=ts, local_recv_ts=ts,
            mark_px=80_000.0 + i * 1.0,
        ))

    # Books for priceBinary legs
    for sym, bid, ask in (("#30", 0.94, 0.95), ("#31", 0.04, 0.05)):
        ms.apply(BboEvent(
            venue="hyperliquid", product_type=ProductType.PREDICTION_BINARY,
            mechanism=Mechanism.CLOB, symbol=sym,
            exchange_ts=now_ns, local_recv_ts=now_ns,
            bid_px=bid, bid_sz=10.0, ask_px=ask, ask_sz=10.0,
        ))
    # Books for priceBucket legs
    for sym, bid, ask in (("#0", 0.20, 0.21), ("#1", 0.04, 0.05),
                          ("#10", 0.70, 0.71), ("#11", 0.04, 0.05)):
        ms.apply(BboEvent(
            venue="hyperliquid", product_type=ProductType.PREDICTION_BINARY,
            mechanism=Mechanism.CLOB, symbol=sym,
            exchange_ts=now_ns, local_recv_ts=now_ns,
            bid_px=bid, bid_sz=10.0, ask_px=ask, ask_sz=10.0,
        ))


def test_scan_wiring_bucket_uses_dt2_binary_uses_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """scan() must route per-class cadence overrides to the correct dt arg on
    build_decision_inputs calls (R1: scanner now calls build_decision_inputs
    directly instead of ms.recent_returns / ms.recent_hl_bars).

    - priceBucket has vol_sampling_dt_seconds=2 in its theta_override → scan()
      must call build_decision_inputs(..., dt=2).
    - priceBinary has no dt override → scan() must call build_decision_inputs(...)
      with dt=None (the dt-less / default-cadence read).

    Approach: monkeypatch hlanalysis.engine.scanner.build_decision_inputs to
    record (dt,) per call; _ClassRecordingStrategy links klass → which inputs
    were received by evaluate().
    """
    import hlanalysis.engine.scanner as _scanner_mod
    from hlanalysis.marketdata.decision_kernel import build_decision_inputs as _real_bdi

    now = 1_700_000_000_000_000_000
    cfg = _cfg(theta_overrides={"priceBucket": {"vol_sampling_dt_seconds": 2}})

    ms = _RecordingMarketState()
    # Register both cadences so MarketState buckets marks correctly for dt=5 AND dt=2.
    ms.set_reference_cadence("BTC", sampling_dt_seconds=5, lookback_seconds=3600)
    ms.set_reference_cadence("BTC", sampling_dt_seconds=2, lookback_seconds=3600)
    _seed_two_class_market(now, ms)

    dal = StateDAL(tmp_path / "state.db")
    dal.run_migrations()

    rcfg = LateResolutionConfig(
        tte_min_seconds=0, tte_max_seconds=86400,
        price_extreme_threshold=0.0, distance_from_strike_usd_min=0.0,
        vol_max=100.0, max_position_usd=500.0, stop_loss_pct=None,
        max_strike_distance_pct=100.0, min_recent_volume_usd=0.0,
        stale_data_halt_seconds=30,
    )
    strat = _ClassRecordingStrategy(rcfg)
    scanner = Scanner(
        strategy=strat, cfg=cfg, market_state=ms, dal=dal,
        kill_switch_path=tmp_path / "halt", last_reconcile_ns=now,
    )

    # Confirm _cadence_by_class is wired correctly before running scan()
    assert scanner._cadence_by_class == {"priceBucket": (2, scanner._cadence_by_class["priceBucket"][1])}
    assert "priceBinary" not in scanner._cadence_by_class

    # Monkeypatch build_decision_inputs in the scanner module so we can record
    # the (dt,) argument the scanner passes without touching the real logic.
    # R1: the scanner now calls build_decision_inputs(self.ms._core, ..., dt=...)
    # directly instead of ms.recent_returns / ms.recent_hl_bars.
    bdi_calls: list[int | None] = []

    def _recording_bdi(core, *, ref_symbol, now_ns, lookback_seconds, dt=None):
        bdi_calls.append(dt)
        return _real_bdi(core, ref_symbol=ref_symbol, now_ns=now_ns,
                         lookback_seconds=lookback_seconds, dt=dt)

    monkeypatch.setattr(_scanner_mod, "build_decision_inputs", _recording_bdi)

    scanner.scan(now_ns=now)

    # We must have seen at least one evaluate() call per question class to
    # verify the routing (if both lists are empty the market seeding failed).
    klasses_evaluated = {klass for klass, _, _ in strat.calls}
    assert "priceBinary" in klasses_evaluated, (
        "priceBinary question was never evaluated — check market seeding"
    )
    assert "priceBucket" in klasses_evaluated, (
        "priceBucket question was never evaluated — check market seeding"
    )

    # build_decision_inputs is called once per evaluated question.
    # Correlate bdi_calls order with strat.calls order to verify per-class dt.
    assert len(bdi_calls) == len(strat.calls), (
        f"expected one build_decision_inputs call per evaluate call, "
        f"got {len(bdi_calls)} vs {len(strat.calls)}"
    )
    for (klass, _rets, _hl), dt_arg in zip(strat.calls, bdi_calls):
        if klass == "priceBucket":
            assert dt_arg == 2, (
                f"priceBucket build_decision_inputs must use dt=2, got dt={dt_arg}"
            )
        elif klass == "priceBinary":
            # R9: the default path now passes the slot's explicit dt (5) rather
            # than dt=None, so reads always land on the correct cadence buffer
            # even when sibling slots have registered a different dt first.
            assert dt_arg == 5, (
                f"priceBinary build_decision_inputs must use dt=5 (slot's explicit dt), "
                f"got dt={dt_arg!r}"
            )


# --- R9: scanner passes explicit dt for default path to avoid cadences[0] aliasing ---


def test_scanner_default_path_passes_explicit_dt_not_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R9 capability: when a slot has a fixed vol_sampling_dt_seconds (say 5),
    the scanner's default path (no per-class override) must call
    build_decision_inputs with dt=5 explicitly, NOT dt=None.

    This ensures that if a sibling slot registers a different default cadence on
    the same symbol first (making cadences[0] != this slot's dt), the scanner
    still reads its own cadence buffer — not the first-registered one.

    Concretely: we have two slots on BTC — slot A at dt=60 (registered first,
    so cadences[0]=60) and slot B at dt=5. Slot B's scanner must call
    build_decision_inputs(..., dt=5), NOT dt=None (which would resolve to
    cadences[0]=60 and read the wrong buffer).
    """
    import hlanalysis.engine.scanner as _scanner_mod
    from hlanalysis.marketdata.decision_kernel import build_decision_inputs as _real_bdi

    now = 1_700_000_000_000_000_000
    # Slot B has dt=5 (via theta block).
    cfg = _cfg()  # default uses ThetaParams(vol_sampling_dt_seconds=5)

    ms = _RecordingMarketState()
    # Register dt=60 FIRST (simulating slot A having registered before slot B),
    # then dt=5 for slot B. This makes cadences[0]=60 for BTC.
    ms.set_reference_cadence("BTC", sampling_dt_seconds=60, lookback_seconds=3600)
    ms.set_reference_cadence("BTC", sampling_dt_seconds=5, lookback_seconds=3600)
    # Confirm the ordering: cadences[0] is dt=60, NOT dt=5.
    assert ms._cadences_by_symbol["BTC"][0] == 60 * 1_000_000_000, (
        "test precondition: dt=60 must be first so cadences[0] != slot B's dt=5"
    )
    _seed_two_class_market(now, ms)

    dal = StateDAL(tmp_path / "state.db")
    dal.run_migrations()

    from hlanalysis.strategy.late_resolution import (
        LateResolutionConfig, LateResolutionStrategy,
    )
    rcfg = LateResolutionConfig(
        tte_min_seconds=0, tte_max_seconds=86400,
        price_extreme_threshold=0.0, distance_from_strike_usd_min=0.0,
        vol_max=100.0, max_position_usd=500.0, stop_loss_pct=None,
        max_strike_distance_pct=100.0, min_recent_volume_usd=0.0,
        stale_data_halt_seconds=30,
    )
    strat = LateResolutionStrategy(rcfg)
    scanner = Scanner(
        strategy=strat, cfg=cfg, market_state=ms, dal=dal,
        kill_switch_path=tmp_path / "halt", last_reconcile_ns=now,
    )

    # Record dt args on build_decision_inputs calls.
    bdi_calls: list[int | None] = []

    def _recording_bdi(core, *, ref_symbol, now_ns, lookback_seconds, dt=None):
        bdi_calls.append(dt)
        return _real_bdi(core, ref_symbol=ref_symbol, now_ns=now_ns,
                         lookback_seconds=lookback_seconds, dt=dt)

    monkeypatch.setattr(_scanner_mod, "build_decision_inputs", _recording_bdi)

    scanner.scan(now_ns=now)

    assert bdi_calls, "scanner.scan() made no build_decision_inputs calls — check market seeding"
    # Every default-path call (no per-class override in this cfg) must use
    # the slot's own dt=5 explicitly, never dt=None — because dt=None would
    # resolve to cadences[0]=60 and silently read the wrong buffer.
    for i, dt_arg in enumerate(bdi_calls):
        assert dt_arg == 5, (
            f"R9: scanner default path call #{i} used dt={dt_arg!r} instead of dt=5 "
            f"(cadences[0]=60 would be returned by dt=None, aliasing the wrong buffer)"
        )


def test_late_resolution_scanner_derives_default_dt_from_allowlist(
    tmp_path: Path,
) -> None:
    """Regression (R9 review): a late_resolution slot has ``theta=None`` and
    carries ``vol_sampling_dt_seconds`` on its allowlist defaults. The scanner's
    default cadence MUST come from the canonical ``reference_sampling_dt_seconds``
    (→ defaults.vol_sampling_dt_seconds), NOT a bare ``else 60``. A ``60`` here
    would make the LIVE engine read a dt=60 OHLC buffer never registered/fed for
    a dt=5 late_resolution slot (v1 / v1_pm) → wrong σ. No backtest catches this
    (the sim path doesn't use the scanner), so this unit assertion is the guard.
    """
    from hlanalysis.strategy.late_resolution import (
        LateResolutionConfig, LateResolutionStrategy,
    )

    defaults = AllowlistEntry(
        match={}, max_position_usd=500, stop_loss_pct=None, tte_min_seconds=0,
        tte_max_seconds=43200, price_extreme_threshold=0.0,
        distance_from_strike_usd_min=0, vol_max=100,
        vol_sampling_dt_seconds=5,            # late_res carries dt here, theta=None
    )
    cfg = StrategyConfig(
        name="late_resolution", account_alias="v1", paper_mode=False,
        strategy_type="late_resolution",
        allowlist=[_entry("priceBinary")], blocklist_question_idxs=[],
        defaults=defaults, theta=None, **{"global": _global()},
    )
    assert cfg.theta is None, "test precondition: late_resolution has no theta block"

    ms = _RecordingMarketState()
    ms.set_reference_cadence("BTC", sampling_dt_seconds=5, lookback_seconds=3600)
    dal = StateDAL(tmp_path / "state.db")
    dal.run_migrations()
    rcfg = LateResolutionConfig(
        tte_min_seconds=0, tte_max_seconds=86400,
        price_extreme_threshold=0.0, distance_from_strike_usd_min=0.0,
        vol_max=100.0, max_position_usd=500.0, stop_loss_pct=None,
        max_strike_distance_pct=100.0, min_recent_volume_usd=0.0,
        stale_data_halt_seconds=30,
    )
    scanner = Scanner(
        strategy=LateResolutionStrategy(rcfg), cfg=cfg, market_state=ms, dal=dal,
        kill_switch_path=tmp_path / "halt", last_reconcile_ns=0,
    )
    assert scanner._default_dt_seconds == 5, (
        f"late_resolution scanner must derive dt=5 from allowlist defaults, got "
        f"{scanner._default_dt_seconds} (a bare `else 60` would read an unregistered buffer)"
    )
