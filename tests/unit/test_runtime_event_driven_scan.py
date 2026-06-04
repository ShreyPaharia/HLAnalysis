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
