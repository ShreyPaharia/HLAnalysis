"""PM up/down strike capture: fetch the Binance SPOT 1m candle close that a
Polymarket up/down market resolves against, and persist it as the question's
strike.

Extracted from runtime.py as free functions taking the EngineRuntime (`rt`) so
the orchestrator stays thin. Behaviour is identical to the prior instance
methods — ``EngineRuntime._maybe_capture_pm_strike`` /
``_pm_strike_capture_loop`` now delegate here.
"""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from loguru import logger

from .config import match_question

if TYPE_CHECKING:  # avoid a runtime import cycle (EngineRuntime lives in runtime)
    from .runtime import AccountSlot, EngineRuntime


async def maybe_capture_pm_strike(
    rt: EngineRuntime, qv, slots: list[AccountSlot],
    fields: dict[str, str], *, now_ns: int,
) -> None:
    """Capture a PM up/down strike from the Binance SPOT 1m candle close.

    Single capture path. PM resolves against the spot 1m candle CLOSE at
    strike_ref_ts_ns, so we wait until that minute has closed
    (now >= ref_ts + 60s) and fetch the close. Fires only when: the question
    is a PM up/down market (has strike_ref_ts_ns), its strike is still unset,
    the reference minute has closed, and no slot already has a persisted
    strike (a restart reloads instead). Fetched off the event loop.
    """
    # Internal symbol the Binance SPOT reference feed is remapped to (see
    # runtime._remap_reference_symbol). Imported lazily to avoid an import cycle.
    from .runtime import _SPOT_REF_SYMBOL
    _ONE_MINUTE_NS = 60 * 1_000_000_000
    _CANDLE_SETTLE_NS = 2 * 1_000_000_000  # let Binance finalize/publish the 1m close
    if qv is None or qv.venue != "polymarket":
        return
    if qv.strike == qv.strike:  # already non-NaN
        return
    raw = dict(qv.kv).get("strike_ref_ts_ns")
    if not raw:
        return
    try:
        ref_ts_ns = int(raw)
    except (TypeError, ValueError):
        return
    if now_ns < ref_ts_ns + _ONE_MINUTE_NS + _CANDLE_SETTLE_NS:
        return  # reference 1m candle not closed+published yet — retry next tick
    qidx = qv.question_idx
    # Per-question lock: prevents a concurrent first-sight call and a periodic
    # loop tick from both entering the fetch path simultaneously. The second
    # caller acquires the lock after the first has persisted the strike and
    # bails on the get_pm_strike check inside.
    lock = rt._pm_strike_locks.setdefault(qidx, asyncio.Lock())
    async with lock:
        pm_slots = [
            s for s in slots
            if match_question(s.cfg, question_idx=qidx, fields=fields) is not None
        ]
        if any(s.dal.get_pm_strike(qidx) is not None for s in pm_slots):
            return  # a prior run captured it; scanner reloads from DB
        try:
            strike = await asyncio.to_thread(rt.klines_fetcher, ref_ts_ns)
        except Exception:
            logger.exception("pm strike capture fetch crashed qidx={}", qidx)
            return
        if strike is None:
            logger.warning(
                "pm strike capture failed qidx={} ref_ts_ns={} — market skipped",
                qidx, ref_ts_ns,
            )
            return
        rt.market_state.set_question_strike(qidx, strike)
        for s in pm_slots:
            s.dal.set_pm_strike(qidx, strike)
        logger.info(
            "pm strike captured qidx={} strike={} (binance spot 1m close)",
            qidx, strike,
        )
        _MISMATCH_TOL_BPS = 10.0
        mark = rt.market_state.last_mark(_SPOT_REF_SYMBOL)
        if mark:
            bps = abs(strike - mark) / mark * 1e4
            if bps > _MISMATCH_TOL_BPS:
                from .risk_events import PMStrikeMismatch
                await rt.bus.publish(PMStrikeMismatch(
                    ts_ns=now_ns, question_idx=qidx,
                    captured_strike=strike, reference_mark=mark,
                    divergence_bps=bps,
                ))
                logger.warning(
                    "pm strike/mark divergence qidx={} strike={} mark={} bps={:.1f} "
                    "(alert only)", qidx, strike, mark, bps,
                )


async def pm_strike_capture_loop(
    rt: EngineRuntime, slots: list[AccountSlot],
) -> None:
    """Retry PM up/down strike capture each second. First-sight capture in
    _ingest_loop fires at discovery, but PM lists markets ~24h before open,
    so the strike can only be fetched once the reference 1m candle closes.
    This loop walks unresolved PM questions and retries until captured."""
    while not rt.stop_event.is_set():
        try:
            now_ns = rt._now_ns()
            for qv in rt.market_state.all_questions():
                # skip non-PM and already-captured (strike==strike is True
                # for a real float, False for NaN = still unresolved)
                if qv.venue != "polymarket" or qv.strike == qv.strike:
                    continue
                fields = {
                    "class": qv.klass, "underlying": qv.underlying,
                    "period": qv.period, "venue": qv.venue,
                    "series_slug": dict(qv.kv).get("series_slug", ""),
                }
                if any(
                    match_question(s.cfg, question_idx=qv.question_idx, fields=fields) is not None
                    for s in slots
                ):
                    await maybe_capture_pm_strike(rt, qv, slots, fields, now_ns=now_ns)
        except Exception:
            logger.exception("pm strike capture loop tick crashed")
        await rt._sleep_or_stop(1.0)
