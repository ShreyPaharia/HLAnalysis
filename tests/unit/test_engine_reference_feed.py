"""The engine ingests a dedicated, minimal Binance BTCUSDT SPOT ``bbo``
(bookTicker) reference feed so a BTCUSDT_SPOT reference price/return series
exists for the PM σ source. This is the FEED side; whether a slot *reads* it
for σ is governed by ``StrategyConfig.reference_sigma_source`` (default "mark",
so adding the feed is inert for existing slots).

The perp BTCUSDT reference feed was removed in 2026-06-01 once the PM slots
moved to ``reference_symbol: BTCUSDT_SPOT`` (PM settles on Binance SPOT 1m
close — no perp/spot basis in price, σ, or strike).

These tests pin:
  - the engine's subscription set contains ONLY the spot binance bbo
    reference feed (BTCUSDT spot ``bbo``-only); no perp feed;
  - the engine's adapter routes binance subs to a BinanceAdapter;
  - a ``bbo``-only spot sub takes the pure-WS path (NO REST premium poll);
  - a BTCUSDT BboEvent reaches MarketState's ``book`` with staleness
    keyed on its recv ts;
  - the BTCUSDT book is populated but UNREAD for σ by default (mark-sourced),
    so existing slots' decisions are unchanged.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import suppress
from pathlib import Path

import pytest

from hlanalysis.adapters.binance import BinanceAdapter
from hlanalysis.adapters.composite import CompositeAdapter
from hlanalysis.config import Subscription, load_config
from hlanalysis.engine.main import (
    binance_spot_reference_subscription,
    build_engine_adapter,
    build_engine_subscriptions,
)
from hlanalysis.engine.market_state import MarketState
from hlanalysis.events import BboEvent, Mechanism, NormalizedEvent, ProductType


# ---- subscription wiring ----

def test_engine_subscribes_binance_spot_reference():
    sub = binance_spot_reference_subscription()  # default = BTCUSDT
    assert sub.venue == "binance"
    assert sub.product_type == ProductType.SPOT
    assert sub.symbol == "BTCUSDT"
    assert sub.channels == ("bbo",)


def test_build_engine_subscriptions_includes_spot_reference_only():
    from hlanalysis.config import RecorderConfig
    cfg = RecorderConfig(subscriptions=[])
    subs = build_engine_subscriptions(cfg)
    binance = [(s.product_type, s.symbol) for s in subs if s.venue == "binance"]
    assert (ProductType.SPOT, "BTCUSDT") in binance
    assert (ProductType.PERP, "BTCUSDT") not in binance   # perp feed removed


def test_build_engine_subscriptions_has_btc_and_eth_binance_bbo_subs():
    """Two binance bbo subs (BTCUSDT_SPOT + ETHUSDT_SPOT) — no perp feed."""
    sym_cfg = load_config(Path("config/symbols.yaml"))
    subs = build_engine_subscriptions(sym_cfg)

    binance_subs = [s for s in subs if s.venue == "binance"]
    assert len(binance_subs) == 2  # BTC + ETH spot reference feeds
    syms = {s.symbol for s in binance_subs}
    assert syms == {"BTCUSDT", "ETHUSDT"}
    for bsub in binance_subs:
        assert bsub.product_type == ProductType.SPOT
        assert bsub.channels == ("bbo",)


def test_build_engine_subscriptions_preserves_hl_and_pm_unchanged():
    sym_cfg = load_config(Path("config/symbols.yaml"))
    subs = build_engine_subscriptions(sym_cfg)

    # Every HL + PM sub from symbols.yaml is present verbatim; nothing dropped
    # or mutated, and no other venues leak in.
    expected_hlpm = [
        s for s in sym_cfg.subscriptions
        if s.venue in ("hyperliquid", "polymarket")
    ]
    got_hlpm = [s for s in subs if s.venue in ("hyperliquid", "polymarket")]
    assert got_hlpm == expected_hlpm
    assert {s.venue for s in subs} == {"hyperliquid", "polymarket", "binance"}


def test_engine_adapter_routes_binance_to_binance_adapter():
    adapter = build_engine_adapter()
    assert isinstance(adapter, CompositeAdapter)
    child_venues = {a.venue for a in adapter._adapters}
    assert "binance" in child_venues
    assert {"hyperliquid", "polymarket"} <= child_venues


# ---- bbo-only is pure WS (no REST premium poll) ----

@pytest.mark.asyncio
async def test_bbo_only_spot_sub_spawns_no_rest_premium_poll(monkeypatch):
    """A bbo-only spot sub must NOT spawn ``_poll_perp_premium`` — that REST
    task is only for perp mark/funding. (Positive control below proves the flag
    works.)"""
    poll_started = False
    run_one_started = False

    async def _fake_run_one(self, *_a, **_k):
        nonlocal run_one_started
        run_one_started = True
        await asyncio.sleep(3600)

    async def _fake_poll(self, *_a, **_k):
        nonlocal poll_started
        poll_started = True
        await asyncio.sleep(3600)

    monkeypatch.setattr(BinanceAdapter, "_run_one", _fake_run_one)
    monkeypatch.setattr(BinanceAdapter, "_poll_perp_premium", _fake_poll)

    await _run_stream_briefly(BinanceAdapter(), [binance_spot_reference_subscription()])

    assert run_one_started is True  # bookTicker WS is subscribed
    assert poll_started is False  # but no REST premium poll


@pytest.mark.asyncio
async def test_perp_sub_with_mark_does_spawn_rest_premium_poll(monkeypatch):
    """Positive control: a sub that DOES request ``mark`` spawns the REST poll,
    so the bbo-only assertion above isn't vacuously true."""
    poll_started = False

    async def _fake_run_one(self, *_a, **_k):
        await asyncio.sleep(3600)

    async def _fake_poll(self, *_a, **_k):
        nonlocal poll_started
        poll_started = True
        await asyncio.sleep(3600)

    monkeypatch.setattr(BinanceAdapter, "_run_one", _fake_run_one)
    monkeypatch.setattr(BinanceAdapter, "_poll_perp_premium", _fake_poll)

    mark_sub = Subscription(
        venue="binance", product_type=ProductType.PERP,
        mechanism=Mechanism.CLOB, symbol="BTCUSDT", channels=("bbo", "mark"),
    )
    await _run_stream_briefly(BinanceAdapter(), [mark_sub])

    assert poll_started is True


# ---- BBO reaches MarketState book ----

def test_binance_spot_bookticker_handle_produces_btcusdt_bbo():
    """The real adapter ``_handle`` turns a spot bookTicker frame into a
    BTCUSDT BboEvent (no stubbing of the parse path)."""
    sub = binance_spot_reference_subscription()
    sym_to_sub = {"BTCUSDT": sub}
    msg = {
        "e": "bookTicker", "s": "BTCUSDT", "u": 123, "E": 1_700_000_000,
        "b": "80000.1", "B": "2.5", "a": "80000.3", "A": "1.1",
    }
    out = BinanceAdapter()._handle(msg, recv_ns=999, sym_to_sub=sym_to_sub, label="spot")
    assert len(out) == 1
    ev = out[0]
    assert isinstance(ev, BboEvent)
    assert ev.symbol == "BTCUSDT"
    assert ev.product_type == ProductType.SPOT
    assert ev.bid_px == 80000.1
    assert ev.ask_px == 80000.3
    assert ev.local_recv_ts == 999


def test_market_state_routes_btcusdt_bbo_to_book_keyed_on_recv_ts():
    ms = MarketState()
    ev = _btcusdt_spot_bbo(bid=80000.1, ask=80000.3, recv_ns=555)
    ms.apply(ev)

    book = ms.book("BTCUSDT")
    assert book is not None
    assert book.bid_px == 80000.1
    assert book.ask_px == 80000.3
    # Staleness machinery (stale_books / stale_data_halt) keys on last_l2_ts_ns;
    # the exchange ts is used when present, recv ts otherwise.
    assert book.last_l2_ts_ns == ev.exchange_ts or book.last_l2_ts_ns == 555


@pytest.mark.asyncio
async def test_btcusdt_bbo_reaches_market_state_through_composite():
    """End-to-end of the feed side: a binance-venue adapter's BTCUSDT BboEvent
    flows through the CompositeAdapter into MarketState.book."""
    ms = MarketState()
    binance_stub = _StubAdapter("binance", [_btcusdt_spot_bbo(80000.0, 80001.0, recv_ns=7)])
    composite = CompositeAdapter([binance_stub])

    async def _drain() -> None:
        async for ev in composite.stream([binance_spot_reference_subscription()]):
            ms.apply(ev)
            return

    await asyncio.wait_for(_drain(), timeout=2.0)
    assert ms.book("BTCUSDT") is not None
    assert ms.book("BTCUSDT").bid_px == 80000.0


# ---- default behaviour unchanged: BTCUSDT book unread for σ ----

def test_btcusdt_book_populated_but_unread_for_sigma_by_default():
    """With no reference source registered (default "mark"), the BTCUSDT BBO
    populates the book ONLY — it does NOT feed the σ/OHLC reference. So
    recent_returns / last_mark stay empty, i.e. existing slots' σ inputs are
    unchanged by merely adding the feed."""
    ms = MarketState()
    for i in range(10):
        ms.apply(_btcusdt_spot_bbo(80000.0 + i, 80001.0 + i, recv_ns=1_000_000_000 * i))

    assert ms.book("BTCUSDT") is not None  # book populated
    assert ms.recent_returns("BTCUSDT", n=8) == ()  # but σ series empty
    assert ms.last_mark("BTCUSDT") is None
    assert ms.reference_source_for("BTCUSDT") == "mark"


# ---- helpers ----

def _btcusdt_spot_bbo(bid: float, ask: float, *, recv_ns: int) -> BboEvent:
    return BboEvent(
        venue="binance", product_type=ProductType.SPOT, mechanism=Mechanism.CLOB,
        symbol="BTCUSDT", exchange_ts=0, local_recv_ts=recv_ns, seq=1,
        bid_px=bid, bid_sz=1.0, ask_px=ask, ask_sz=1.0,
    )


class _StubAdapter:
    """Minimal binance-venue adapter yielding a fixed event list, then idling."""

    def __init__(self, venue: str, events: list[NormalizedEvent]) -> None:
        self.venue = venue
        self._events = events

    def supports(self, *_a, **_k) -> bool:
        return True

    async def stream(self, _subs: list[Subscription]) -> AsyncIterator[NormalizedEvent]:
        for ev in self._events:
            await asyncio.sleep(0)
            yield ev
        while True:
            await asyncio.sleep(3600)


async def _run_stream_briefly(adapter: BinanceAdapter, subs: list[Subscription]) -> None:
    """Drive ``adapter.stream(subs)`` just long enough for it to spawn its child
    tasks, then tear down. Used to observe which tasks were created without a
    live WS."""
    agen = adapter.stream(subs)
    pull = asyncio.create_task(agen.__anext__())  # runs stream() up to first yield
    await asyncio.sleep(0.05)  # let spawned child tasks run their first line
    pull.cancel()
    with suppress(asyncio.CancelledError):
        await pull
    with suppress(Exception):
        await agen.aclose()
