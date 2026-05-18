"""Schema-drift sentinel against Kalshi's public REST API.

Marked ``@pytest.mark.live`` — skipped in CI; run locally via
``uv run pytest -m live tests/integration/test_kalshi_live_smoke.py -v``.

Asserts the fields the KalshiDataSource depends on are still present on
recent events: ``event_ticker``, ``expiration_time``, ``ticker``,
``floor_strike``, ``cap_strike``, ``settlement_value``, ``open_time`` for
markets; ``trades[].created_time``, ``yes_price``, ``no_price``, ``count``,
``taker_side`` for trades.
"""
from __future__ import annotations

import pytest

from hlanalysis.backtest.data._kalshi_client import (
    fetch_event_detail,
    fetch_events_page,
    fetch_market_trades,
)


@pytest.mark.live
def test_kalshi_events_page_schema():
    page, _ = fetch_events_page(series_ticker="KXBTCD", status="settled", limit=1)
    assert page, "no settled KXBTCD events returned — series_ticker may have changed"
    ev = page[0]
    for field in ("event_ticker", "expiration_time"):
        assert field in ev, f"missing field on event: {field}"


@pytest.mark.live
def test_kalshi_event_detail_and_market_schema():
    page, _ = fetch_events_page(series_ticker="KXBTCD", status="settled", limit=1)
    assert page
    et = page[0]["event_ticker"]
    event, markets = fetch_event_detail(et)
    assert event.get("event_ticker") == et
    assert markets, f"event {et} returned no markets"
    m = markets[0]
    for field in ("ticker", "floor_strike", "cap_strike", "settlement_value", "open_time"):
        assert field in m, f"missing field on market: {field}"


@pytest.mark.live
def test_kalshi_trades_schema():
    page, _ = fetch_events_page(series_ticker="KXBTCD", status="settled", limit=1)
    event, markets = fetch_event_detail(page[0]["event_ticker"])
    market_ticker = markets[0]["ticker"]
    trades, _ = fetch_market_trades(market_ticker, limit=1)
    if not trades:
        pytest.skip(f"no trades on market {market_ticker}")
    t = trades[0]
    for field in ("created_time", "yes_price", "no_price", "count", "taker_side"):
        assert field in t, f"missing field on trade: {field}"
