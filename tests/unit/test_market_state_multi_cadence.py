# tests/unit/test_market_state_multi_cadence.py
"""(symbol, dt) bucketing: one reference-tick stream, two cadence series.

Lets a single slot run different vol_sampling_dt_seconds per question class
(v31 buckets dt=2 vs v31 binary/v1 dt=5) off the SAME feed. Single-cadence
reads must stay bit-identical to the legacy per-symbol path.
"""
from __future__ import annotations

import math

from hlanalysis.engine.market_state import MarketState
from hlanalysis.events import MarkEvent, Mechanism, ProductType


def _mark(symbol: str, px: float, ts_s: float) -> MarkEvent:
    ts = int(ts_s * 1_000_000_000)
    return MarkEvent(
        venue="hyperliquid", product_type=ProductType.PERP,
        mechanism=Mechanism.CLOB, symbol=symbol,
        exchange_ts=ts, local_recv_ts=ts, mark_px=px,
    )


def _feed(ms: MarketState, symbol: str, ticks: list[tuple[float, float]]) -> None:
    for ts_s, px in ticks:
        ms.apply(_mark(symbol, px, ts_s))


def test_two_cadences_bucket_same_stream_independently() -> None:
    """A symbol registered at dt=2 and dt=5 maintains two independent bar
    series from one tick stream: dt=2 closes every 2s, dt=5 every 5s."""
    ms = MarketState()
    ms.set_reference_cadence("BTC", sampling_dt_seconds=2)
    ms.set_reference_cadence("BTC", sampling_dt_seconds=5)
    # ticks at t=0,1,2,3,4,5,6 → prices 100..106
    _feed(ms, "BTC", [(float(t), 100.0 + t) for t in range(7)])

    # dt=5 buckets: [0,5)->close@t4=104, [5,10)->close@t6=106 → 1 return
    rets5 = ms.recent_returns("BTC", n=10, dt=5)
    assert len(rets5) == 1
    assert math.isclose(rets5[0], math.log(106.0 / 104.0), rel_tol=1e-12)

    # dt=2 buckets: [0,2)->101, [2,4)->103, [4,6)->105, [6,8)->106 → closes
    # 101,103,105,106 → 3 returns. Independent series off the same ticks.
    rets2 = ms.recent_returns("BTC", n=10, dt=2)
    assert len(rets2) == 3
    assert math.isclose(rets2[0], math.log(103.0 / 101.0), rel_tol=1e-12)


def test_single_cadence_read_is_bit_identical_to_legacy() -> None:
    """A symbol with exactly one registered cadence yields identical
    recent_returns whether or not dt is passed (default resolves to the sole
    registered cadence)."""
    ms = MarketState()
    ms.set_reference_cadence("BTC", sampling_dt_seconds=5)
    _feed(ms, "BTC", [(float(t), 100.0 + (t % 3)) for t in range(20)])
    assert ms.recent_returns("BTC", n=8) == ms.recent_returns("BTC", n=8, dt=5)
    assert ms.recent_hl_bars("BTC", n=8) == ms.recent_hl_bars("BTC", n=8, dt=5)


def test_same_cadence_reregistration_is_idempotent() -> None:
    """Re-registering the SAME (symbol, dt) only grows history sizing; it never
    raises (two slots can share class+cadence)."""
    ms = MarketState()
    ms.set_reference_cadence("BTC", sampling_dt_seconds=5, lookback_seconds=1800)
    ms.set_reference_cadence("BTC", sampling_dt_seconds=5, lookback_seconds=3600)
    assert ms.mark_bucket_ns_for("BTC", dt=5) == 5 * 1_000_000_000
    assert ms._mark_history_by_key[("BTC", 5 * 1_000_000_000)] >= 3600 // 5 + 2
