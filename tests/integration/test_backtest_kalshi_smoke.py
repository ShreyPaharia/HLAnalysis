"""End-to-end: KalshiDataSource → discover → events → settlement against the
small synthetic fixture. Catches integration breakage between the adapter
and the rest of the backtest stack.
"""
from __future__ import annotations

from pathlib import Path

from hlanalysis.backtest.data.kalshi import KalshiDataSource
from hlanalysis.backtest.core.events import (
    BookSnapshot, ReferenceEvent, SettlementEvent, TradeEvent,
)

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "kalshi"


def test_kalshi_fixture_discovers_one_question():
    ds = KalshiDataSource(cache_root=FIXTURE_ROOT)
    descs = ds.discover(start="2023-11-01", end="2024-01-01")
    assert len(descs) == 1
    d = descs[0]
    assert d.klass == "priceBucket"
    assert d.underlying == "BTC"
    assert len(d.leg_symbols) == 6  # 3 markets × 2 sides


def test_kalshi_fixture_events_emit_in_order_and_include_settlement():
    ds = KalshiDataSource(cache_root=FIXTURE_ROOT)
    d = ds.discover(start="2023-11-01", end="2024-01-01")[0]
    evs = list(ds.events(d))
    # Monotone non-decreasing.
    for a, b in zip(evs, evs[1:]):
        assert a.ts_ns <= b.ts_ns
    # At least one trade, one book snapshot, and the six settlement events.
    assert any(isinstance(e, TradeEvent) for e in evs)
    assert any(isinstance(e, BookSnapshot) for e in evs)
    settles = [e for e in evs if isinstance(e, SettlementEvent)]
    assert len(settles) == 6
    by_sym = {s.symbol: s.outcome for s in settles}
    # M1 wins.
    assert by_sym["M1|yes"] == "yes"
    assert by_sym["M1|no"] == "no"
    assert by_sym["M0|yes"] == "no"
    assert by_sym["M0|no"] == "yes"


def test_kalshi_fixture_question_view_carries_priceThresholds():
    ds = KalshiDataSource(cache_root=FIXTURE_ROOT)
    d = ds.discover(start="2023-11-01", end="2024-01-01")[0]
    qv = ds.question_view(d, now_ns=d.end_ts_ns + 1, settled=True)
    assert dict(qv.kv)["priceThresholds"] == "79000,80000"
    assert qv.klass == "priceBucket"
