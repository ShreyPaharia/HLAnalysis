"""Canonical signed-position fill math — the ONE implementation shared by the
live router (``engine.router._book_fill``) and the backtest runner
(``backtest.runner.hftbt_runner``).

Both used to reimplement the same accounting in parallel and stay identical by
manual discipline; drift produced the position-fidelity bugs in the project
history. This module is the single source of truth, so the two paths are
bit-identical by construction.

``apply_fill`` is a PURE function over a minimal :class:`PositionState`
(qty / avg_entry / realized_pnl) — deliberately decoupled from both
``engine.state.Position`` and the runner's ``Position`` so one function serves
both callers. Stop-price stamping is intentionally NOT folded in here: the two
callers stamp stops from different sources (the router from the matched
allowlist entry, the runner from the strategy's ``stop_loss_pct``), so each
caller stamps its own stop after calling ``apply_fill``. The shared stop FORMULA
lives in ``stop_price`` and the shared disabled-stop sentinel in
``STOP_DISABLED_SENTINEL``.
"""
from __future__ import annotations

from dataclasses import dataclass

# A negative price the risk gate's stop-loss check (``px <= stop`` for longs)
# can never trigger on, since real bid prices are >= 0. Defining it once here
# (it used to be copy-pasted into router.py, reconcile.py and hftbt_runner.py)
# means a change can never desync the three.
STOP_DISABLED_SENTINEL = -1.0

# Sub-precision dust floor for Polymarket positions. PM market sells round the
# share amount DOWN to 2dp (``pm_client._round_down_2``), so closing a non-round
# buy strands the floored-off fraction (a 58.1279-share exit sells 58.12 and
# leaves 0.0079). That residual is BOTH un-sellable (PM floors it back to 0.00 →
# `invalid maker amount`) and ~$0.01 of notional, so a position at or below this
# size is "closed for all practical purposes". The router treats a reduce that
# lands here as a full close (``reduce_close_atol``); the reconciler clears a
# stranded dust row instead of re-firing DRIFT forever. The floor sits above the
# max 2dp sell residual (≤5e-3) with margin and ~100x below a 1-share min order,
# so it can never swallow a real lagging position (which is whole shares).
DUST_QTY_ABS_TOL = 1e-2


@dataclass(frozen=True, slots=True)
class PositionState:
    """The signed-position fields the fill math reads and writes. A thin value
    object so ``apply_fill`` need not know about either caller's richer
    ``Position`` row (symbol, timestamps, stop price, …)."""

    qty: float
    avg_entry: float
    realized_pnl: float = 0.0


def stop_price(entry: float, pct: float | None) -> float:
    """Stop-loss trigger price for an entry at ``entry`` with a ``pct`` percent
    drawdown stop. ``None`` disables the stop (returns the sentinel). Matches
    the formula both callers used: ``max(0, entry * (1 - pct/100))``."""
    if pct is None:
        return STOP_DISABLED_SENTINEL
    return max(0.0, entry * (1.0 - pct / 100.0))


def apply_fill(
    pos: PositionState | None,
    side: str,
    size: float,
    price: float,
    *,
    close_atol: float = 1e-9,
) -> tuple[PositionState | None, float]:
    """Apply one fill to ``pos`` and return ``(new_pos, realized_this_fill)``.

    Semantics (preserved exactly from ``router._book_fill``):
      * ``side`` ``"buy"`` adds ``+size``, ``"sell"`` adds ``-size`` (signed qty).
      * Opening from flat (``pos is None``): the new position is ``(signed,
        price, 0.0)`` and nothing is realized.
      * Add-on (same-direction fill): cost basis becomes the qty-weighted
        average of the prior position and this fill; nothing is realized.
      * Reduce (opposite-direction fill): realizes PnL on the closed lot
        (``closed_qty = min(size, |qty|)``) — ``(price - avg) * closed_qty`` for
        a long, ``(avg - price) * closed_qty`` for a short — and leaves the cost
        basis UNCHANGED (the closed lot's PnL is already captured, so recomputing
        avg would inflate it on every subsequent reduce).
      * Close: when ``|new_qty| < close_atol`` the position is fully closed and
        ``new_pos`` is ``None``. ``close_atol`` defaults to ``1e-9`` (the live
        router's threshold); the backtest runner passes its lot size so a
        sub-lot residual is treated as closed.

    ``realized_this_fill`` is 0.0 on opens and add-ons, and the realized PnL on
    the closed lot otherwise. On a full close it equals ``(price - avg) * qty``
    for both long and short, so a caller can publish ``realized_this_fill +
    prior_realized`` as the trade's realized PnL.
    """
    signed = size if side == "buy" else -size

    realized_this_fill = 0.0
    if pos is not None and (
        (pos.qty > 0 and side == "sell") or (pos.qty < 0 and side == "buy")
    ):
        closed_qty = min(size, abs(pos.qty))
        if pos.qty > 0:
            realized_this_fill = (price - pos.avg_entry) * closed_qty
        else:
            realized_this_fill = (pos.avg_entry - price) * closed_qty

    if pos is None:
        return PositionState(qty=signed, avg_entry=price, realized_pnl=0.0), 0.0

    new_qty = pos.qty + signed
    if abs(new_qty) < close_atol:
        return None, realized_this_fill

    is_addon = (pos.qty > 0 and side == "buy") or (pos.qty < 0 and side == "sell")
    if is_addon:
        avg = (pos.qty * pos.avg_entry + signed * price) / new_qty
    else:
        avg = pos.avg_entry
    return (
        PositionState(
            qty=new_qty,
            avg_entry=avg,
            realized_pnl=pos.realized_pnl + realized_this_fill,
        ),
        realized_this_fill,
    )


__all__ = [
    "STOP_DISABLED_SENTINEL", "DUST_QTY_ABS_TOL",
    "PositionState", "stop_price", "apply_fill",
]
