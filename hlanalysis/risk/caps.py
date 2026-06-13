"""Entry-only risk cap predicates shared by the live engine and the backtester.

These are pure functions with no IO and no imports from ``engine/`` or
``backtest/`` — following the same precedent as ``hlanalysis/marketdata/
position_math.py``. Both the live ``engine.risk.RiskGate`` and the sim
``backtest.halt_replay.entry_veto`` import from here so the comparison
operators, semantics, and ``daily_window_start_ns`` boundary logic stay in
a single canonical location.

Comparison operators (all match the live engine):
- inventory:   held_plus_orders_notional + intent_notional  >  cap    (strict >)
- concurrent:  n_held  >=  cap  AND NOT top-up              (>=, top-up exempt)
- daily loss:  realized_pnl_window  <  -cap                 (strict <, cap is positive)
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta


def inventory_cap_exceeded(
    held_plus_orders_notional: float,
    intent_notional: float,
    cap: float | None,
) -> bool:
    """Return ``True`` when adding *intent_notional* would breach *cap*.

    Args:
        held_plus_orders_notional: Current total exposure — the sum of
            ``live_orders_total_notional`` (pending/open orders) and the
            mark-to-cost of all held positions (``sum(|qty| * avg_entry)``).
            The caller is responsible for combining these two components before
            calling; the engine does so inline, the sim bundles them into
            ``EntryGateInputs.held_inventory_usd``.
        intent_notional: ``size * limit_price`` of the proposed entry.
        cap: The ``max_total_inventory_usd`` threshold.  ``None`` disables the
            check (returns ``False``).
    """
    if cap is None:
        return False
    return held_plus_orders_notional + intent_notional > cap


def concurrent_cap_exceeded(
    n_held: int,
    is_topup: bool,
    cap: int | None,
) -> bool:
    """Return ``True`` when adding a new position would breach *cap*.

    A top-up to an already-held position is exempt (``is_topup=True`` always
    returns ``False``), matching the live engine's behaviour where only a truly
    new slot increments the concurrent count.

    Args:
        n_held: Number of currently held positions.
        is_topup: ``True`` if the entry targets an already-open position.
        cap: The ``max_concurrent_positions`` threshold.  ``None`` disables the
            check.
    """
    if cap is None or is_topup:
        return False
    return n_held >= cap


def daily_loss_exceeded(
    realized_pnl_window: float,
    cap: float | None,
) -> bool:
    """Return ``True`` when the daily realized loss has breached *cap*.

    Args:
        realized_pnl_window: Realized PnL accumulated since the start of the
            current daily window (negative when losing).
        cap: The ``daily_loss_cap_usd`` threshold (a positive dollar amount).
            ``None`` disables the check.
    """
    if cap is None:
        return False
    return realized_pnl_window < -cap


def daily_window_start_ns(now_ns: int, *, hour: int) -> int:
    """Most-recent ``HH:00:00 UTC`` boundary at-or-before ``now_ns``.

    Canonical implementation shared by:
    - ``engine.scanner.Scanner._daily_window_start_ns`` (delegates here)
    - ``backtest.halt_replay.daily_window_start_ns`` (re-exports this)

    If *now_ns* is already past today's HH:00 UTC, returns today's HH:00.
    Otherwise rolls back to yesterday's HH:00.  This gives a stable daily
    window boundary so the same fill never appears in two consecutive windows.
    """
    dt = datetime.fromtimestamp(now_ns / 1e9, tz=UTC)
    boundary = dt.replace(hour=hour, minute=0, second=0, microsecond=0)
    if dt < boundary:
        boundary = boundary - timedelta(days=1)
    return int(boundary.timestamp() * 1_000_000_000)


__all__ = [
    "inventory_cap_exceeded",
    "concurrent_cap_exceeded",
    "daily_loss_exceeded",
    "daily_window_start_ns",
]
