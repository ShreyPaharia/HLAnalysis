"""SHR-97 gate: engine path ≡ backtest path on the same recorded tick stream.

Two acceptance tests per the spec §4:

1. **Unit parity**: `from_engine(cfg)` and `from_backtest_params(params)` agree on
   all four knobs (source, dt, lookback, ticks) for a representative HL slot.

2. **Bit-identical replay gate**: the same recorded HL reference tick stream fed
   through the *engine* MarketState (NormalizedEvent → apply) and the *backtest*
   runner MarketState (raw ReferenceEvent → apply_reference_tick), both configured
   from the SAME `DecisionInputConfig` resolver, produces bit-identical outputs at
   a grid of sampled `now_ns`:
     - `recent_returns`, `recent_hl_bars`, `last_mark`
     - `sigma` for each estimator (stdev, bipower, parkinson)
   Covers dt ∈ {5, 60} × source ∈ {mark, bbo}.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixture corpus loader — reuses the hl_hip4 fixture the SHR-87 test uses.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE_ROOT = _REPO_ROOT / "tests" / "fixtures" / "hl_hip4"

_S = 1_000_000_000  # ns per second


def _load_hl_reference_ticks(
    source: str = "mark",
) -> list[tuple[int, float]]:
    """Return a list of (ts_ns, price) tuples from the HL fixture corpus.

    ``source="mark"`` reads MarkEvent mark_px (the live HL path).
    ``source="bbo"`` reads BboEvent bid/ask mid.

    The ticker stream used here is the same corpus the SHR-87 gate loads, but
    we extract only the reference-price ticks (skipping L2 / trade / meta
    events) so the two MarketState adapters can be fed an identical scalar
    (ts_ns, price) sequence.
    """
    import duckdb

    base = _REPO_ROOT / "tests" / "fixtures" / "hl_hip4"
    con = duckdb.connect()
    ticks: list[tuple[int, float]] = []

    # Find the event directory using Python rglob (duckdb multi-** glob broken).
    event_dirs = {d.name.split("=", 1)[1]: d for d in base.rglob("event=*") if d.is_dir()}

    if source == "mark":
        event_dir = event_dirs.get("mark")
        if event_dir is None:
            return ticks
        glob = str(event_dir / "**" / "*.parquet")
        try:
            rows = con.execute(
                f"SELECT exchange_ts, local_recv_ts, mark_px "
                f"FROM read_parquet('{glob}', union_by_name=true) "
                f"ORDER BY exchange_ts NULLS LAST"
            ).fetchall()
        except Exception:
            rows = []
        for ts_ex, ts_local, px in rows:
            ts = ts_ex or ts_local
            if ts and px and float(px) > 0:
                ticks.append((int(ts), float(px)))

    elif source == "bbo":
        event_dir = event_dirs.get("bbo")
        if event_dir is None:
            return ticks
        glob = str(event_dir / "**" / "*.parquet")
        try:
            rows = con.execute(
                f"SELECT exchange_ts, local_recv_ts, bid_px, ask_px "
                f"FROM read_parquet('{glob}', union_by_name=true) "
                f"ORDER BY exchange_ts NULLS LAST"
            ).fetchall()
        except Exception:
            rows = []
        for ts_ex, ts_local, bid, ask in rows:
            ts = ts_ex or ts_local
            if ts and bid and ask and float(bid) > 0 and float(ask) > 0:
                mid = (float(bid) + float(ask)) / 2.0
                ticks.append((int(ts), mid))

    ticks.sort(key=lambda x: x[0])
    return ticks


# ---------------------------------------------------------------------------
# Helper: build engine-side and backtest-side MarketState from the SAME
# DecisionInputConfig, feed them the identical tick sequence, assert parity.
# ---------------------------------------------------------------------------


def _run_parity_check(*, source: str, dt: int, lookback: int = 3600) -> None:
    """Feed the same recorded tick stream to both adapters, configured from one
    DecisionInputConfig, and assert bit-identical outputs at sampled now_ns."""
    from hlanalysis.marketdata.decision_input import DecisionInputConfig

    # Build the resolver.
    cfg = DecisionInputConfig(
        reference_source=source,
        sampling_dt_seconds=dt,
        vol_lookback_seconds=lookback,
        reference_ticks="raw",
    )

    # --- Engine path ---
    from hlanalysis.engine.market_state import MarketState as EngineMS
    from hlanalysis.events import (
        BboEvent,
        MarkEvent,
        Mechanism,
        NormalizedEvent,
        ProductType,
    )

    engine_ms = EngineMS()
    engine_ms.set_reference_cadence(
        "BTC",
        sampling_dt_seconds=cfg.sampling_dt_seconds,
        lookback_seconds=cfg.vol_lookback_seconds,
    )
    engine_ms.set_reference_source("BTC", cfg.reference_source)

    # --- Backtest path ---
    from hlanalysis.backtest.runner.market_state import (
        MarketState as BacktestMS,
        _REFERENCE_KEY,
    )
    from hlanalysis.backtest.core.events import ReferenceEvent

    bt_ms = BacktestMS()
    bt_ms.set_reference_cadence(cfg.sampling_dt_seconds)

    # Load ticks.
    ticks = _load_hl_reference_ticks(source=source)
    assert len(ticks) > 100, f"Too few {source!r} ticks in HL fixture ({len(ticks)}); is the fixture corpus present?"

    # Feed all ticks to both adapters, sample at every 250th tick.
    sym = "BTC"
    sample_every = 250
    comparison_points = 0

    for i, (ts_ns, price) in enumerate(ticks):
        # Engine ingest: wrap as NormalizedEvent.
        if source == "mark":
            ev: NormalizedEvent = MarkEvent(
                venue="hyperliquid",
                product_type=ProductType.PERP,
                mechanism=Mechanism.CLOB,
                symbol=sym,
                exchange_ts=ts_ns,
                local_recv_ts=ts_ns,
                mark_px=price,
            )
        else:  # bbo
            ev = BboEvent(
                venue="binance",
                product_type=ProductType.PERP,
                mechanism=Mechanism.CLOB,
                symbol=sym,
                exchange_ts=ts_ns,
                local_recv_ts=ts_ns,
                bid_px=price - 1.0,
                bid_sz=1.0,
                ask_px=price + 1.0,
                ask_sz=1.0,
            )
        engine_ms.apply(ev)

        # Backtest ingest: raw ReferenceEvent (H=L=C=price).
        ref_ev = ReferenceEvent(ts_ns=ts_ns, symbol=sym, high=price, low=price, close=price)
        bt_ms.apply_reference_tick(ref_ev)

        if i % sample_every != 0:
            continue

        now_ns = ts_ns
        ctx = f"source={source!r} dt={dt} tick#{i}"

        # --- last_mark ---
        eng_mark = engine_ms.last_mark(sym)
        bt_mark = bt_ms.latest_btc_close()
        assert eng_mark == bt_mark, f"last_mark mismatch @ {ctx}: engine={eng_mark} bt={bt_mark}"

        # --- recent_returns and recent_hl_bars ---
        eng_rets = engine_ms._core.recent_returns(sym, now_ns=now_ns, lookback_seconds=lookback, dt=dt)
        bt_rets = bt_ms.recent_returns(now_ns=now_ns, lookback_seconds=lookback)
        assert np.array_equal(eng_rets, bt_rets), (
            f"recent_returns mismatch @ {ctx}: engine {eng_rets.shape} bt {bt_rets.shape}"
        )

        eng_hl = engine_ms._core.recent_hl_bars(sym, now_ns=now_ns, lookback_seconds=lookback, dt=dt)
        bt_hl = bt_ms.recent_hl_bars(now_ns=now_ns, lookback_seconds=lookback)
        assert np.array_equal(eng_hl, bt_hl), f"recent_hl_bars mismatch @ {ctx}: engine {eng_hl.shape} bt {bt_hl.shape}"

        # --- sigma (all three estimators) ---
        # Engine reads its core keyed by the real symbol; the backtest adapter
        # keys reference data under _REFERENCE_KEY and resolves dt from its single
        # registered cadence (no explicit dt — matches the recent_returns read
        # above). These are two DISTINCT cores fed independently — the comparison
        # is meaningful only because each side reads its own core.
        for estimator in ("stdev", "bipower", "parkinson"):
            eng_sig = engine_ms._core.sigma(sym, estimator=estimator, now_ns=now_ns, lookback_seconds=lookback, dt=dt)
            bt_sig = bt_ms._core.sigma(_REFERENCE_KEY, estimator=estimator, now_ns=now_ns, lookback_seconds=lookback)
            assert eng_sig == bt_sig, f"sigma[{estimator}] mismatch @ {ctx}: engine={eng_sig} bt={bt_sig}"

        comparison_points += 1

    assert comparison_points >= 5, f"Too few comparison points for source={source!r} dt={dt}: {comparison_points}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_unit_from_engine_and_from_backtest_params_agree():
    """from_engine(cfg) and from_backtest_params(params) must agree on all four
    knobs for a representative HL slot config — they are the SAME config path."""
    from hlanalysis.marketdata.decision_input import (
        DecisionInputConfig,
        from_engine,
        from_backtest_params,
    )
    from hlanalysis.engine.config import (
        AllowlistEntry,
        GlobalRiskConfig,
        StrategyConfig,
        ThetaParams,
    )

    # Build a representative HL v31 (theta) slot config at dt=5.
    _global = GlobalRiskConfig(
        max_total_inventory_usd=500,
        max_concurrent_positions=5,
        daily_loss_cap_usd=200,
        max_strike_distance_pct=50,
        min_recent_volume_usd=100,
        stale_data_halt_seconds=30,
        reconcile_interval_seconds=60,
    )
    entry = AllowlistEntry(
        match={"class": "priceBinary", "underlying": "BTC"},
        max_position_usd=100,
        stop_loss_pct=None,
        tte_min_seconds=0,
        tte_max_seconds=43200,
        price_extreme_threshold=0.0,
        distance_from_strike_usd_min=0,
        vol_max=100,
        vol_lookback_seconds=3600,
    )
    theta = ThetaParams(vol_sampling_dt_seconds=5, vol_lookback_seconds=3600)
    cfg = StrategyConfig(
        name="theta_harvester",
        account_alias="v31",
        paper_mode=True,
        strategy_type="theta_harvester",
        reference_symbol="BTC",
        reference_sigma_source="mark",
        allowlist=[entry],
        blocklist_question_idxs=[],
        defaults=entry,
        theta=theta,
        **{"global": _global},
    )

    # Equivalent backtest params dict.
    params = {
        "vol_sampling_dt_seconds": 5,
        "vol_lookback_seconds": 3600,
        # reference_sigma_source absent → resolved from track_default_source
    }

    engine_resolved = from_engine(cfg)
    bt_resolved = from_backtest_params(params, track_default_source="mark")

    assert engine_resolved.reference_source == bt_resolved.reference_source, (
        f"source mismatch: engine={engine_resolved.reference_source!r} bt={bt_resolved.reference_source!r}"
    )
    assert engine_resolved.sampling_dt_seconds == bt_resolved.sampling_dt_seconds, (
        f"dt mismatch: engine={engine_resolved.sampling_dt_seconds} bt={bt_resolved.sampling_dt_seconds}"
    )
    assert engine_resolved.vol_lookback_seconds == bt_resolved.vol_lookback_seconds, (
        f"lookback mismatch: engine={engine_resolved.vol_lookback_seconds} bt={bt_resolved.vol_lookback_seconds}"
    )
    assert engine_resolved.reference_ticks == bt_resolved.reference_ticks == "raw", (
        f"ticks: engine={engine_resolved.reference_ticks!r} bt={bt_resolved.reference_ticks!r}"
    )


@pytest.mark.parametrize(
    "source,dt",
    [
        ("mark", 5),
        ("mark", 60),
        ("bbo", 5),
        ("bbo", 60),
    ],
)
def test_engine_backtest_bit_identical_on_recorded_hl_ticks(source: str, dt: int) -> None:
    """Engine MarketState and backtest runner MarketState, both configured from
    the same DecisionInputConfig, produce bit-identical sigma/returns/hl/last_mark
    on the same recorded HL tick stream. Exact equality (not allclose)."""
    _run_parity_check(source=source, dt=dt)


def test_from_engine_returns_live_faithful_values():
    """from_engine must return reference_ticks='raw' and the exact same source/dt
    the engine config specifies."""
    from hlanalysis.marketdata.decision_input import from_engine
    from hlanalysis.engine.config import (
        AllowlistEntry,
        GlobalRiskConfig,
        StrategyConfig,
        ThetaParams,
    )

    _global = GlobalRiskConfig(
        max_total_inventory_usd=500,
        max_concurrent_positions=5,
        daily_loss_cap_usd=200,
        max_strike_distance_pct=50,
        min_recent_volume_usd=100,
        stale_data_halt_seconds=30,
        reconcile_interval_seconds=60,
    )
    entry = AllowlistEntry(
        match={"class": "priceBinary"},
        max_position_usd=100,
        stop_loss_pct=None,
        tte_min_seconds=0,
        tte_max_seconds=43200,
        price_extreme_threshold=0.0,
        distance_from_strike_usd_min=0,
        vol_max=100,
        vol_lookback_seconds=3600,
    )
    cfg = StrategyConfig(
        name="theta_harvester",
        account_alias="v31",
        paper_mode=True,
        strategy_type="theta_harvester",
        reference_symbol="BTC",
        reference_sigma_source="bbo",
        allowlist=[entry],
        blocklist_question_idxs=[],
        defaults=entry,
        theta=ThetaParams(vol_sampling_dt_seconds=5, vol_lookback_seconds=3600),
        **{"global": _global},
    )
    resolved = from_engine(cfg)
    assert resolved.reference_source == "bbo"
    assert resolved.sampling_dt_seconds == 5
    assert resolved.reference_ticks == "raw"


def test_from_backtest_params_explicit_source_override():
    """When params contains reference_sigma_source, from_backtest_params uses it
    (the explicit params override wins over track_default_source)."""
    from hlanalysis.marketdata.decision_input import from_backtest_params

    params = {
        "vol_sampling_dt_seconds": 60,
        "vol_lookback_seconds": 900,
        "reference_sigma_source": "bbo",
    }
    resolved = from_backtest_params(params, track_default_source="mark")
    assert resolved.reference_source == "bbo"
    assert resolved.sampling_dt_seconds == 60
    assert resolved.vol_lookback_seconds == 900
    assert resolved.reference_ticks == "raw"


def test_from_backtest_params_nested_per_class():
    """Per-class (binary/bucket) nested params: max lookback across classes."""
    from hlanalysis.marketdata.decision_input import from_backtest_params

    params = {
        "binary": {"vol_sampling_dt_seconds": 5, "vol_lookback_seconds": 3600},
        "bucket": {"vol_sampling_dt_seconds": 2, "vol_lookback_seconds": 900},
    }
    # The first class's dt should be returned (or we just check that it parses
    # without error and returns sensible values).
    resolved = from_backtest_params(params, track_default_source="mark")
    # Max lookback should be 3600 (max across both classes).
    assert resolved.vol_lookback_seconds == 3600
    assert resolved.reference_ticks == "raw"
    assert resolved.reference_source == "mark"
