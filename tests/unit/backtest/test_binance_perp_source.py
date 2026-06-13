from __future__ import annotations

import json
from pathlib import Path

from hlanalysis.backtest.core.events import BookSnapshot
from hlanalysis.backtest.data.binance_perp import BinancePerpKlinesSource


def test_emits_synthetic_bbo_between_start_and_end(tmp_path: Path) -> None:
    p = tmp_path / "klines.json"
    rows = [
        {"ts_ns": 1_000_000_000_000_000_000, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1.0},
        {"ts_ns": 1_000_000_060_000_000_000, "open": 100.5, "high": 102.0, "low": 100.0, "close": 101.0, "volume": 1.0},
        {"ts_ns": 1_000_000_120_000_000_000, "open": 101.0, "high": 101.5, "low": 100.5, "close": 100.8, "volume": 1.0},
    ]
    p.write_text(json.dumps(rows))
    src = BinancePerpKlinesSource(path=p, symbol="BTC-PERP", half_spread_bps=1.0)
    events = list(
        src.book_events(
            start_ts_ns=1_000_000_060_000_000_000,
            end_ts_ns=1_000_000_120_000_000_000,
        )
    )
    assert len(events) == 1
    ev = events[0]
    assert isinstance(ev, BookSnapshot)
    assert ev.symbol == "BTC-PERP"
    assert ev.ts_ns == 1_000_000_060_000_000_000
    # close 101.0 ± 1bp half-spread = 0.0101 each side
    # bids[0] = (bid_px, bid_sz), asks[0] = (ask_px, ask_sz)
    bid_px = ev.bids[0][0]
    ask_px = ev.asks[0][0]
    assert abs(bid_px - (101.0 * (1.0 - 1e-4))) < 1e-6
    assert abs(ask_px - (101.0 * (1.0 + 1e-4))) < 1e-6
