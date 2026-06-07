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


async def test_question_meta_persists_coin_klass_for_hl(tmp_path, monkeypatch):
    # SHR-77: ingesting an HL question's metadata must stamp the coin("#N")→klass
    # map into the DB of each slot that can trade it, keyed by the exact "#N"
    # coins user_fills returns. Bucket: named_outcome_idxs=[16] → legs #160/#161.
    import hlanalysis.engine.runtime as runtime_mod
    from hlanalysis.engine.market_state import MarketState
    from hlanalysis.engine.state import StateDAL
    from hlanalysis.events import Mechanism, ProductType, QuestionMetaEvent

    # Force every slot to count as able to trade the question (avoids wiring a
    # full StrategyConfig); the persistence branch keys off match_question.
    monkeypatch.setattr(runtime_mod, "match_question", lambda *a, **k: object())

    dal = StateDAL(tmp_path / "state.db")
    dal.run_migrations()

    class FakeBus:
        async def publish(self, ev):  # NewQuestion alert — irrelevant here
            pass

    slot = type("Slot", (), {"dal": dal, "cfg": object(), "alias": "v31"})()

    rt = EngineRuntime.__new__(EngineRuntime)
    rt.market_state = MarketState()
    rt._market_dirty = asyncio.Event()
    rt.events_ingested = 0
    rt.bus = FakeBus()
    rt._now_ns = lambda: 1
    rt._pm_strike_locks = {}

    ev = QuestionMetaEvent(
        venue="hyperliquid", product_type=ProductType.PREDICTION_BINARY,
        mechanism=Mechanism.CLOB, symbol="qmeta", exchange_ts=1, local_recv_ts=1,
        question_idx=16, named_outcome_idxs=[16],
        keys=["class", "underlying", "period"],
        values=["priceBucket", "BTC", "1d"],
    )
    await rt._handle_ingest_event(ev, slots=[slot], seen_questions=set())

    assert dal.coin_klass_map() == {"#160": "priceBucket", "#161": "priceBucket"}


async def test_question_meta_does_not_persist_coin_klass_for_pm(tmp_path, monkeypatch):
    # PM legs are CLOB token ids (not "#N") and PM fills are binary by
    # construction, so the coin→klass map must stay HL-only.
    import hlanalysis.engine.runtime as runtime_mod
    from hlanalysis.engine.market_state import MarketState
    from hlanalysis.engine.state import StateDAL
    from hlanalysis.events import Mechanism, ProductType, QuestionMetaEvent

    monkeypatch.setattr(runtime_mod, "match_question", lambda *a, **k: object())

    dal = StateDAL(tmp_path / "state.db")
    dal.run_migrations()

    class FakeBus:
        async def publish(self, ev):
            pass

    slot = type("Slot", (), {"dal": dal, "cfg": object(), "alias": "v31_pm"})()

    rt = EngineRuntime.__new__(EngineRuntime)
    rt.market_state = MarketState()
    rt._market_dirty = asyncio.Event()
    rt.events_ingested = 0
    rt.bus = FakeBus()
    rt._now_ns = lambda: 1
    rt._pm_strike_locks = {}

    ev = QuestionMetaEvent(
        venue="polymarket", product_type=ProductType.PREDICTION_BINARY,
        mechanism=Mechanism.CLOB, symbol="yt", exchange_ts=1, local_recv_ts=1,
        question_idx=1000126, named_outcome_idxs=[0, 1],
        keys=["class", "underlying", "yes_token_id", "no_token_id", "strike"],
        values=["priceBinary", "BTC", "yt", "nt", "73500"],
    )
    await rt._handle_ingest_event(ev, slots=[slot], seen_questions=set())

    assert dal.coin_klass_map() == {}


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
