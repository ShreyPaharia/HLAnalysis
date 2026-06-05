# tests/unit/test_replay_cadence_parity.py
"""ReplayRunner single-cadence reads stay bit-identical after the (symbol, dt)
refactor: a dt-less read resolves to the sole registered cadence."""
from __future__ import annotations

from hlanalysis.engine.market_state import MarketState
from hlanalysis.events import MarkEvent, Mechanism, ProductType


def test_dtless_read_equals_explicit_dt_for_single_cadence() -> None:
    ms = MarketState()
    ms.set_reference_cadence("BTC", sampling_dt_seconds=5, lookback_seconds=1800)
    for t in range(50):
        ts = t * 1_000_000_000
        ms.apply(MarkEvent(
            venue="hyperliquid", product_type=ProductType.PERP,
            mechanism=Mechanism.CLOB, symbol="BTC",
            exchange_ts=ts, local_recv_ts=ts, mark_px=100.0 + (t % 7) * 0.5,
        ))
    assert ms.recent_returns("BTC", n=32) == ms.recent_returns("BTC", n=32, dt=5)
    assert ms.recent_hl_bars("BTC", n=32) == ms.recent_hl_bars("BTC", n=32, dt=5)
