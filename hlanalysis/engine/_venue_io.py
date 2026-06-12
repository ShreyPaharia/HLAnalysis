"""Async venue I/O helpers extracted from ``EngineRuntime``.

Every ``ExecutionClient`` method is a synchronous, requests-backed SDK call
(some wrapped in tenacity retries). The engine runs ONE asyncio loop shared by
WS ingest, heartbeat, the stop-loss enforcer, reconcile, and every slot — so
calling these inline parks the whole loop for the network round-trip, exactly
when the venue is congested. These helpers push the blocking call onto a worker
thread. We deliberately offload only the venue calls, not the surrounding
DAL/MarketState work: StateDAL opens a fresh connection per call (thread-safe),
but MarketState is mutated by the ingest loop and must stay on the loop thread.

``EngineRuntime._venue_snapshot`` and ``EngineRuntime._realized_pnl_today``
remain as thin delegating methods so callers (tests, loop bodies) that already
call ``rt._venue_snapshot(slot)`` / ``rt._realized_pnl_today(slot, ...)``
continue to work without modification.

Do NOT import ``runtime.py`` from this module — that would create a cycle.
"""
from __future__ import annotations

import asyncio

from loguru import logger

from .exec_types import ClearinghouseState, OpenOrderRow, UserFillRow
from .scanner import Scanner


async def venue_snapshot(slot: object) -> tuple[
    list[OpenOrderRow], ClearinghouseState, list[UserFillRow]
]:
    """Fetch venue open-orders, clearinghouse state, and the full fills list
    off the event loop. Fills are fetched once and reused as the reconcile
    `fills_lookup` for every cloid — the live lambda ignores its cloid arg
    and returns all fills anyway, so this is behaviour-preserving.

    ``slot`` is typed ``object`` to avoid importing ``AccountSlot`` from
    ``runtime.py`` (import cycle); callers pass an ``AccountSlot`` instance.
    """
    open_orders = await asyncio.to_thread(slot.exec_client.open_orders)  # type: ignore[attr-defined]
    state = await asyncio.to_thread(slot.exec_client.clearinghouse_state)  # type: ignore[attr-defined]
    fills = await asyncio.to_thread(slot.exec_client.user_fills)  # type: ignore[attr-defined]
    return open_orders, state, fills


async def realized_pnl_today(
    slot: object,
    *,
    now_ns: int,
    venue_pnl_failures: dict[str, int],
    daily_loss_venue_fail_halt: int,
) -> float:
    """Realized PnL since the slot's daily window start, for both the Scanner
    daily-loss read and the continuous-checks cap.

    Venue-truth, no double-count: HL exposes HIP-4 / bucket settlement as a
    *fill* (``dir="Settlement"``, ``closedPnl`` populated), so
    ``realized_pnl_since`` ALREADY includes the settlement payout — adding the
    separately-persisted settlement PnL on top double-counts it (and
    historically mis-signed it for multi-leg buckets: a winning leg booked as a
    total loss tripped a spurious DAILY LOSS HALT). PM settles via on-chain
    redeem, which is NOT a CLOB fill, so PM's venue read misses it and the
    persisted settlement must be added back. On a venue-read failure we fall
    back to the DAL value (now also settlement-inclusive, no longer structurally
    zero) and count the failure; after ``daily_loss_venue_fail_halt`` consecutive
    failures we latch the slot halted (fail-safe) rather than keep trading
    venue-blind.

    Parameters
    ----------
    slot:
        The ``AccountSlot`` to read PnL for (typed ``object`` to avoid a
        cycle; callers pass the concrete type).
    now_ns:
        Current time in nanoseconds.
    venue_pnl_failures:
        Mutable per-alias failure-streak counter (mutated in place — same
        dict object the caller owns).
    daily_loss_venue_fail_halt:
        Number of consecutive failures after which the slot is halted.
    """
    # slot is AccountSlot at runtime; use getattr for the cycle-free typing.
    alias: str = slot.alias  # type: ignore[attr-defined]
    window_start_ns = Scanner._daily_window_start_ns(
        now_ns, hour=slot.cfg.global_.daily_window_start_hour_utc,  # type: ignore[attr-defined]
    )
    try:
        venue_pnl = await asyncio.to_thread(
            slot.exec_client.realized_pnl_since, window_start_ns,  # type: ignore[attr-defined]
            outcome_only=True,
        )
        venue_pnl_failures[alias] = 0
        if slot.is_pm:  # type: ignore[attr-defined]
            settlement_pnl = await asyncio.to_thread(
                slot.dal.settlement_pnl_since, window_start_ns,  # type: ignore[attr-defined]
            )
            return venue_pnl + settlement_pnl
        return venue_pnl  # HL: settlement already in venue fills' closedPnl
    except Exception:
        n = venue_pnl_failures.get(alias, 0) + 1
        venue_pnl_failures[alias] = n
        logger.warning(
            "realized_pnl_since failed alias={} (consecutive={}); using "
            "settlement-inclusive DAL", alias, n,
        )
        if n >= daily_loss_venue_fail_halt:
            logger.error(
                "venue PnL unreadable for {} consecutive checks; halting "
                "slot {} (fail-safe — cap can't be trusted venue-blind)",
                n, alias,
            )
            slot.halted = True  # type: ignore[attr-defined]
        # DAL realized_pnl_since already includes settlement PnL (SHR-53).
        return await asyncio.to_thread(
            slot.dal.realized_pnl_since, window_start_ns,  # type: ignore[attr-defined]
        )
