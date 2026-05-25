from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import websockets
import websockets.exceptions

from hlanalysis.adapters.polymarket import PolymarketAdapter
from hlanalysis.config import Subscription
from hlanalysis.events import (
    BookSnapshotEvent,
    Mechanism,
    ProductType,
    QuestionMetaEvent,
    TradeEvent,
)


FIXTURE = Path("tests/fixtures/pm/ws_book_frames.jsonl")


class _FakeWS:
    """Async-iter shim that replays JSONL fixture lines as if from a socket.

    Empty-frames behavior simulates a real PM WS close (raises
    `ConnectionClosedOK`) so the adapter's retry-loop is exercised.
    """

    def __init__(self, frames: list[str]):
        self._frames = list(frames)
        self.sent: list[str] = []

    async def send(self, data: str) -> None:
        self.sent.append(data)

    async def recv(self) -> str:
        if not self._frames:
            await asyncio.sleep(0.05)
            raise websockets.exceptions.ConnectionClosedOK(None, None)
        return self._frames.pop(0)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class _StubGamma:
    def fetch_events(self, **kw):
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


@pytest.mark.asyncio
async def test_adapter_emits_book_and_trade_events_from_fixture():
    frames = FIXTURE.read_text().splitlines()
    fake_ws = _FakeWS(frames)
    sub = Subscription(
        venue="polymarket",
        product_type=ProductType.PREDICTION_BINARY,
        mechanism=Mechanism.CLOB,
        symbol="*",
        channels=("trades", "book"),
        match={"series_slug": "btc-up-or-down-daily", "underlying": "BTC"},
    )
    adapter = PolymarketAdapter(
        ws_factory=lambda url: fake_ws,
        gamma_client=_StubGamma(),
    )
    events: list = []

    async def _drain() -> None:
        try:
            async for ev in adapter.stream([sub]):
                events.append(ev)
        except asyncio.CancelledError:
            pass

    await asyncio.wait_for(_drain(), timeout=2.0)
    book = [e for e in events if isinstance(e, BookSnapshotEvent)]
    trade = [e for e in events if isinstance(e, TradeEvent)]
    qmeta = [e for e in events if isinstance(e, QuestionMetaEvent)]
    assert book or trade, "fixture must contain at least one book or trade frame"
    assert qmeta, "adapter must emit a QuestionMetaEvent on startup"


@pytest.mark.asyncio
async def test_adapter_reconnects_on_ws_close():
    closing_ws = _FakeWS([])
    fresh_ws = _FakeWS([
        '{"event_type":"book","asset_id":"tok-yes","timestamp":"1","bids":[],'
        '"asks":[{"price":"0.5","size":"10"}]}'
    ])

    calls = {"n": 0}

    def factory(_url: str):
        calls["n"] += 1
        return closing_ws if calls["n"] == 1 else fresh_ws

    adapter = PolymarketAdapter(ws_factory=factory, gamma_client=_StubGamma())
    sub = Subscription(
        venue="polymarket",
        product_type=ProductType.PREDICTION_BINARY,
        mechanism=Mechanism.CLOB,
        symbol="*",
        channels=("book",),
        match={"series_slug": "btc-up-or-down-daily"},
    )
    events: list = []

    async def _drain():
        try:
            async for ev in adapter.stream([sub]):
                events.append(ev)
                # Return only once we've received a BookSnapshotEvent from the
                # second (reconnected) socket — qmeta/health events alone
                # don't prove a reconnect happened.
                if isinstance(ev, BookSnapshotEvent):
                    return
        except asyncio.CancelledError:
            pass

    await asyncio.wait_for(_drain(), timeout=5.0)
    assert calls["n"] >= 2  # initial + at least one reconnect
    assert any(isinstance(e, BookSnapshotEvent) for e in events)
