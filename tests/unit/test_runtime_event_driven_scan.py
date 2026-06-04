from __future__ import annotations

import asyncio

from hlanalysis.engine.runtime import EngineRuntime, _PRICE_EVENT_TYPES
from hlanalysis.events import BboEvent, Mechanism, ProductType, QuestionMetaEvent


def test_price_event_types_cover_bbo_book_mark_trade():
    # Guard: the dirty signal must fire on the price-moving event classes.
    names = {t.__name__ for t in _PRICE_EVENT_TYPES}
    assert {"BboEvent", "BookSnapshotEvent", "BookDeltaEvent",
            "MarkEvent", "TradeEvent"} <= names


async def test_ingest_price_event_sets_market_dirty():
    rt = EngineRuntime.__new__(EngineRuntime)  # bare instance, no full wiring
    rt.market_state = __import__(
        "hlanalysis.engine.market_state", fromlist=["MarketState"]
    ).MarketState()
    rt._market_dirty = asyncio.Event()
    rt.events_ingested = 0
    ev = BboEvent(
        venue="binance", symbol="BTCUSDT_SPOT", product_type=ProductType.SPOT,
        mechanism=Mechanism.CLOB, bid_px=1.0, bid_sz=1.0, ask_px=1.1,
        ask_sz=1.0, exchange_ts=1, local_recv_ts=1,
    )
    await rt._handle_ingest_event(ev, slots=[], seen_questions=set())
    assert rt._market_dirty.is_set()


async def test_wait_helper_returns_early_on_dirty():
    rt = EngineRuntime.__new__(EngineRuntime)
    rt._market_dirty = asyncio.Event()
    rt.stop_event = asyncio.Event()

    async def _trip():
        await asyncio.sleep(0.02)
        rt._market_dirty.set()

    asyncio.create_task(_trip())
    import time
    t0 = time.perf_counter()
    # max=5s, but the dirty trip at 20ms must return well before that
    await rt._wait_for_market_or_timeout(max_interval=5.0)
    assert time.perf_counter() - t0 < 1.0
    assert not rt._market_dirty.is_set()  # helper clears it


async def test_wait_helper_returns_on_timeout_when_idle():
    rt = EngineRuntime.__new__(EngineRuntime)
    rt._market_dirty = asyncio.Event()
    rt.stop_event = asyncio.Event()
    import time
    t0 = time.perf_counter()
    await rt._wait_for_market_or_timeout(max_interval=0.05)
    assert 0.04 <= time.perf_counter() - t0 < 1.0
