"""Part B: HL/PM synchronous REST must run off the shared asyncio event loop.

HL polls the info endpoint (outcomeMeta, timeout=5) and PM polls Gamma
(/events, timeout=30) with blocking `requests.*`. Invoking those directly from a
coroutine parks the WHOLE loop — stalling EVERY adapter — for the round-trip.
These tests assert the network I/O is offloaded to a worker thread (mirrors the
engine ExecutionClient offload, SHR-41), while the cache mutation stays on the
loop thread (no data races).
"""
from __future__ import annotations

import asyncio
import inspect
import threading

import pytest

from hlanalysis.adapters.hyperliquid import HyperliquidAdapter
from hlanalysis.adapters.polymarket import PolymarketAdapter
from hlanalysis.config import Subscription
from hlanalysis.events import Mechanism, ProductType, QuestionMetaEvent


class _RecordingResp:
    def __init__(self, payload, thread_box):
        self._payload = payload
        thread_box["thread"] = threading.get_ident()

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


# ---- Hyperliquid info endpoint ----

@pytest.mark.asyncio
async def test_hl_fetch_outcome_meta_runs_off_event_loop(monkeypatch):
    box: dict = {}

    def fake_post(url, json=None, timeout=None):
        return _RecordingResp({"outcomes": [], "questions": []}, box)

    monkeypatch.setattr("hlanalysis.adapters.hyperliquid.requests.post", fake_post)
    adapter = HyperliquidAdapter()
    loop_thread = threading.get_ident()

    payload = await adapter._fetch_outcome_meta()

    assert payload == {"outcomes": [], "questions": []}
    assert box["thread"] is not None, "requests.post was never called"
    assert box["thread"] != loop_thread, (
        "HL outcomeMeta requests.post ran on the event-loop thread — must be "
        "offloaded via asyncio.to_thread()."
    )


@pytest.mark.asyncio
async def test_hl_fetch_outcome_meta_returns_none_on_error(monkeypatch):
    def boom(url, json=None, timeout=None):
        raise RuntimeError("network down")

    monkeypatch.setattr("hlanalysis.adapters.hyperliquid.requests.post", boom)
    adapter = HyperliquidAdapter()
    assert await adapter._fetch_outcome_meta() is None


def test_hl_meta_fetch_helpers_are_async():
    a = HyperliquidAdapter()
    assert inspect.iscoroutinefunction(a._expand_wildcard)
    assert inspect.iscoroutinefunction(a._fetch_outcome_meta_events)
    assert inspect.iscoroutinefunction(a._fetch_question_meta_events)
    # The pure diff (no network) stays synchronous.
    assert not inspect.iscoroutinefunction(a._detect_polled_settlements)


# ---- Polymarket Gamma endpoint ----

class _RecordingGamma:
    """Records the thread `fetch_events` runs on; returns one open BTC market."""

    def __init__(self):
        self.fetch_thread: int | None = None

    def fetch_events(self, **kw):
        self.fetch_thread = threading.get_ident()
        return [{
            "markets": [{
                "conditionId": "0xfixturecond",
                "clobTokenIds": '["tok-yes","tok-no"]',
                "endDate": "2026-05-25T00:00:00Z",
                "description": (
                    "Will BTC go up or down? Resolves based on the "
                    "Binance 1 minute candle for BTC/USDT May 24 '26 "
                    "20:00 in the ET timezone..."
                ),
                "outcomePrices": '["0.5","0.5"]',
            }]
        }]

    @staticmethod
    def iter_binary_markets(events):
        for ev in events:
            yield ev["markets"][0]


class _IdleWS:
    async def send(self, _data):
        pass

    async def recv(self):
        await asyncio.sleep(3600)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


@pytest.mark.asyncio
async def test_pm_gamma_fetch_runs_off_event_loop():
    gamma = _RecordingGamma()
    adapter = PolymarketAdapter(ws_factory=lambda url: _IdleWS(), gamma_client=gamma)
    sub = Subscription(
        venue="polymarket", product_type=ProductType.PREDICTION_BINARY,
        mechanism=Mechanism.CLOB, symbol="*", channels=("book",),
        match={"series_slug": "btc-up-or-down-daily", "underlying": "BTC"},
    )
    loop_thread = threading.get_ident()
    events: list = []

    async def _drain():
        async for ev in adapter.stream([sub]):
            events.append(ev)
            if isinstance(ev, QuestionMetaEvent):
                return

    await asyncio.wait_for(_drain(), timeout=2.0)

    # Correctness preserved: the qmeta still flows through.
    assert any(isinstance(e, QuestionMetaEvent) for e in events)
    assert gamma.fetch_thread is not None, "gamma.fetch_events was never called"
    assert gamma.fetch_thread != loop_thread, (
        "PM Gamma fetch_events ran on the event-loop thread — the 30s blocking "
        "requests.get must be offloaded via asyncio.to_thread()."
    )
