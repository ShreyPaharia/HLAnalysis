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

# Wall-clock bound on a single venue read. The SDK calls offloaded below are
# requests-backed and tenacity-wrapped, but tenacity's ``stop_after_delay`` only
# fires *between* attempts — it cannot interrupt one socket read that stalls with
# no timeout. Without this bound a hung venue connection (PM data-api flap) parks
# the worker thread forever; the reconcile loop's ``await`` never returns, so
# ``slot.last_reconcile_ns`` freezes and the per-slot ``stale_reconcile`` gate
# vetoes every entry indefinitely with NO exception and NO log (incident
# 2026-06-14: v31_pm stuck 2.5h). On timeout ``asyncio.wait_for`` raises
# ``TimeoutError`` — already classified transient by ``_is_transient_venue_error``
# — so the reconcile loop logs concisely and retries next cycle. The bound is
# generous (the three reads are sequential and each carries its own ~8s tenacity
# budget, plus the data-api positions fetch) so a legitimately-retrying snapshot
# is never false-aborted; only a true wedge trips it.
#
# NOTE: ``wait_for`` cancels the *await*, not the underlying worker thread — a
# truly-wedged socket leaks one thread until it unblocks. That is an acceptable,
# bounded cost: the loop recovers and keeps reconciling, which is the property
# that matters (a permanently-dead reconcile loop also kills exit/settlement
# detection for the slot).
_VENUE_READ_TIMEOUT_S = 30.0

# Visibility flag for the venue-PnL-unreadable fail-safe halt. Unlike the
# operator/daily-loss kill-switch flag (which is persistent and operator-cleared),
# this one is written when the fail-safe latches and REMOVED when the venue read
# recovers, so `engine-diag` / `engine-status` reflect the live state. Sits next
# to the slot's kill-switch flag.
VENUE_PNL_HALT_FLAG = "venue_pnl_halt"


def _venue_pnl_halt_flag_path(slot: object):
    return slot.kill_switch_path.parent / VENUE_PNL_HALT_FLAG  # type: ignore[attr-defined]


def _set_venue_pnl_halt_flag(slot: object, *, present: bool) -> None:
    """Write (present=True) or remove (present=False) the venue-PnL-halt flag
    file. Best-effort: a filesystem hiccup must not crash the PnL read."""
    path = _venue_pnl_halt_flag_path(slot)
    try:
        if present:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch(exist_ok=True)
        else:
            path.unlink(missing_ok=True)
    except OSError as e:
        logger.warning("venue_pnl_halt flag {} update failed: {}", path, e)


def _recover_venue_pnl_halt(slot: object) -> None:
    """A successful venue read auto-clears a prior venue-fail halt. Scoped to the
    venue-fail path ONLY (``venue_pnl_halted``): an operator/daily-loss kill-switch
    (flag file present) stays latched for the operator. The transient-outage halt
    must not survive once the venue is readable again (incident 2026-06-24)."""
    if not getattr(slot, "venue_pnl_halted", False):
        return
    slot.venue_pnl_halted = False  # type: ignore[attr-defined]
    _set_venue_pnl_halt_flag(slot, present=False)
    # Never override a persistent operator/daily-loss kill (its own flag file).
    if not slot.kill_switch_path.exists():  # type: ignore[attr-defined]
        slot.halted = False  # type: ignore[attr-defined]
        logger.info("venue PnL readable again; cleared venue-fail halt on slot {}", slot.alias)  # type: ignore[attr-defined]


async def venue_snapshot(
    slot: object,
    *,
    timeout_s: float = _VENUE_READ_TIMEOUT_S,
) -> tuple[list[OpenOrderRow], ClearinghouseState, list[UserFillRow]]:
    """Fetch venue open-orders, clearinghouse state, and the full fills list
    off the event loop. Fills are fetched once and reused as the reconcile
    `fills_lookup` for every cloid — the live lambda ignores its cloid arg
    and returns all fills anyway, so this is behaviour-preserving.

    Each read is wall-clock bounded (``timeout_s``) so a wedged venue socket
    cannot freeze the caller's loop; on timeout ``TimeoutError`` propagates.

    ``slot`` is typed ``object`` to avoid importing ``AccountSlot`` from
    ``runtime.py`` (import cycle); callers pass an ``AccountSlot`` instance.
    """

    async def _bounded(call, what: str):
        try:
            return await asyncio.wait_for(asyncio.to_thread(call), timeout=timeout_s)
        except TimeoutError:
            logger.warning(
                "venue read '{}' exceeded {}s wall-clock — aborting (retried next cycle)",
                what,
                timeout_s,
            )
            raise

    open_orders = await _bounded(slot.exec_client.open_orders, "open_orders")  # type: ignore[attr-defined]
    state = await _bounded(slot.exec_client.clearinghouse_state, "clearinghouse_state")  # type: ignore[attr-defined]
    fills = await _bounded(slot.exec_client.user_fills, "user_fills")  # type: ignore[attr-defined]
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
        now_ns,
        hour=slot.cfg.global_.daily_window_start_hour_utc,  # type: ignore[attr-defined]
    )
    try:
        venue_pnl = await asyncio.to_thread(
            slot.exec_client.realized_pnl_since,
            window_start_ns,  # type: ignore[attr-defined]
            outcome_only=True,
        )
        venue_pnl_failures[alias] = 0
        _recover_venue_pnl_halt(slot)
        if slot.is_pm:  # type: ignore[attr-defined]
            settlement_pnl = await asyncio.to_thread(
                slot.dal.settlement_pnl_since,
                window_start_ns,  # type: ignore[attr-defined]
            )
            return venue_pnl + settlement_pnl
        return venue_pnl  # HL: settlement already in venue fills' closedPnl
    except Exception:
        n = venue_pnl_failures.get(alias, 0) + 1
        venue_pnl_failures[alias] = n
        logger.warning(
            "realized_pnl_since failed alias={} (consecutive={}); using settlement-inclusive DAL",
            alias,
            n,
        )
        if n >= daily_loss_venue_fail_halt:
            logger.error(
                "venue PnL unreadable for {} consecutive checks; halting "
                "slot {} (fail-safe — cap can't be trusted venue-blind)",
                n,
                alias,
            )
            slot.halted = True  # type: ignore[attr-defined]
            slot.venue_pnl_halted = True  # type: ignore[attr-defined]
            _set_venue_pnl_halt_flag(slot, present=True)
        # DAL realized_pnl_since already includes settlement PnL (SHR-53).
        return await asyncio.to_thread(
            slot.dal.realized_pnl_since,
            window_start_ns,  # type: ignore[attr-defined]
        )
