from __future__ import annotations

import pytest
import responses

from hlanalysis.backtest.data._kalshi_client import (
    KALSHI_BASE,
    fetch_event_detail,
    fetch_events_page,
    fetch_market_trades,
    iter_events,
    iter_market_trades,
)


@responses.activate
def test_fetch_events_page_returns_events_and_cursor():
    responses.add(
        responses.GET,
        f"{KALSHI_BASE}/events",
        json={"events": [{"event_ticker": "KXBTCD-26MAY18"}], "cursor": "abc"},
        status=200,
    )
    events, cursor = fetch_events_page(series_ticker="KXBTCD", status="settled", limit=200)
    assert events == [{"event_ticker": "KXBTCD-26MAY18"}]
    assert cursor == "abc"


@responses.activate
def test_iter_events_follows_cursor_until_empty():
    responses.add(
        responses.GET, f"{KALSHI_BASE}/events",
        json={"events": [{"event_ticker": "A"}], "cursor": "p2"}, status=200,
    )
    responses.add(
        responses.GET, f"{KALSHI_BASE}/events",
        json={"events": [{"event_ticker": "B"}], "cursor": ""}, status=200,
    )
    out = list(iter_events(series_ticker="KXBTCD", status="settled"))
    assert [e["event_ticker"] for e in out] == ["A", "B"]


@responses.activate
def test_fetch_event_detail_returns_markets():
    responses.add(
        responses.GET,
        f"{KALSHI_BASE}/events/KXBTCD-26MAY18",
        json={"event": {"event_ticker": "KXBTCD-26MAY18"},
              "markets": [{"ticker": "KXBTCD-26MAY18-B79000"}]},
        status=200,
    )
    ev, mks = fetch_event_detail("KXBTCD-26MAY18")
    assert ev["event_ticker"] == "KXBTCD-26MAY18"
    assert len(mks) == 1


@responses.activate
def test_iter_market_trades_paginates_and_stops_on_empty_cursor():
    responses.add(
        responses.GET, f"{KALSHI_BASE}/markets/trades",
        json={"trades": [{"created_time": "2026-05-18T00:00:00Z", "count": 1,
                          "yes_price": 50, "no_price": 50, "taker_side": "yes"}],
              "cursor": "c1"}, status=200,
    )
    responses.add(
        responses.GET, f"{KALSHI_BASE}/markets/trades",
        json={"trades": [], "cursor": ""}, status=200,
    )
    trades = list(iter_market_trades("KXBTCD-26MAY18-B79000"))
    assert len(trades) == 1


@responses.activate
def test_fetch_events_page_retries_on_transient_5xx(monkeypatch):
    monkeypatch.setattr(
        "hlanalysis.backtest.data._kalshi_client.time.sleep", lambda _: None
    )
    responses.add(responses.GET, f"{KALSHI_BASE}/events", status=503)
    responses.add(responses.GET, f"{KALSHI_BASE}/events", status=503)
    responses.add(
        responses.GET, f"{KALSHI_BASE}/events",
        json={"events": [], "cursor": ""}, status=200,
    )
    events, cursor = fetch_events_page(series_ticker="KXBTCD")
    assert events == []
    assert cursor == ""
