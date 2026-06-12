"""Fill-price heuristics, IOC marketability checks, hedge math, and settlement
payoff helpers for the hftbacktest runner.

Extracted verbatim from ``hftbt_runner.py`` — no logic changes.
"""
from __future__ import annotations

import numpy as np

from hlanalysis.marketdata.position_math import settlement_payoff_price
from hlanalysis.strategy.types import BookState, Position

from .result import Fill


def _slipped_buy_price(book: BookState, limit_price: float, slippage_bps: float) -> float:
    if book.ask_px is None:
        return 0.0
    slipped = book.ask_px * (1.0 + slippage_bps / 1e4)
    px = min(slipped, limit_price) if limit_price > 0 else slipped
    return max(0.0, min(1.0, px))


def _slipped_sell_price(book: BookState, limit_price: float, slippage_bps: float) -> float:
    if book.bid_px is None:
        return 0.0
    slipped = book.bid_px * (1.0 - slippage_bps / 1e4)
    px = max(slipped, limit_price) if limit_price > 0 else slipped
    return max(0.0, min(1.0, px))


def _classify_reject(book: BookState, side: str, submit_px: float) -> bool:
    """SHR-89: was a no-fill IOC a *reject* (limit unmarketable at fill time)?

    An order is a reject when it was submitted as a crossing/marketable order
    against the decision-time book (buy at/through the ask; sell at/through the
    bid) yet recorded no fill — the book moved away (or the queue wasn't swept)
    during the latency δ before the order reached the exchange. The strategy
    then re-fires on the next scan, reproducing live's churn.

    A non-fill on a *non-crossing* order (limit below the ask / above the bid),
    or when there was no opposing liquidity to cross, is a plain no-op — not a
    reject."""
    if side == "buy":
        return book.ask_px is not None and submit_px >= book.ask_px
    return book.bid_px is not None and submit_px <= book.bid_px


def _inter_order_blocked(
    now_ns: int, last_order_ns: int | None, min_inter_order_ns: int
) -> bool:
    """SHR-79/SHR-89: would dispatching an order on this leg now violate the
    minimum inter-order re-fire floor?

    Returns ``True`` when a prior order was dispatched on this leg less than
    ``min_inter_order_ns`` ago — the caller must suppress this submission and
    let the strategy re-fire on a later scan (reproducing live's one-order-in-
    flight serialization). Returns ``False`` when:
    - the floor is disabled (``min_inter_order_ns <= 0``); or
    - this is the first order on the leg (``last_order_ns is None``); or
    - at least ``min_inter_order_ns`` has elapsed since the last dispatch.
    """
    if min_inter_order_ns <= 0 or last_order_ns is None:
        return False
    return (now_ns - last_order_ns) < min_inter_order_ns


def _is_fleeting_ask(
    sym: str,
    fill_price: float,
    now_ns: int,
    latency_ns: int,
    snap_best_ask_per_leg: dict[str, np.ndarray],
    book_ts_per_leg: dict[str, np.ndarray],
    book_idx: dict[str, int],
    persistence_ns: int = 0,
) -> bool:
    """SHR-94 + SHR-89b: was the ask at ``fill_price`` a fleeting level?

    Two complementary checks:

    **SHR-94 (latency-window check):** looks at the *next* snapshot within the
    order-latency window ``[now_ns, now_ns + latency_ns]``. If it shows the
    ask has moved ABOVE ``fill_price``, the level is fleeting. Effective when
    HL snapshots are more frequent than the latency (~50ms).

    **SHR-89b (wall-clock persistence check, ``persistence_ns > 0``):** checks
    the *current* snapshot's prior snapshot. If the ask at ``fill_price`` just
    appeared (present in the current snapshot but ABSENT in the previous one),
    the level has been live for fewer nanoseconds than ``persistence_ns`` →
    fleeting. This catches HL's burst-sampled book where the next snapshot is
    seconds away (making the SHR-94 window never fire).

    Either check may veto the IOC independently. Returns ``False`` when:
    - No snapshot arrays for this leg.
    - persistence_ns=0 AND no next snapshot within the latency window.
    """
    snap_ask = snap_best_ask_per_leg.get(sym)
    ts_arr = book_ts_per_leg.get(sym)
    if snap_ask is None or ts_arr is None or len(snap_ask) == 0 or len(ts_arr) == 0:
        return False
    # book_idx[sym] is the first snapshot index NOT YET consumed at now_ns
    # (the scan loop advances it past all ts <= now_ns).
    idx = book_idx.get(sym, 0)

    # SHR-89b: wall-clock persistence check on the CURRENT (most-recently-
    # consumed) snapshot. current_idx = idx - 1 (last consumed snapshot).
    if persistence_ns > 0:
        current_idx = idx - 1
        if current_idx >= 0:
            current_ask = float(snap_ask[current_idx])
            if not np.isnan(current_ask) and current_ask <= fill_price:
                # The ask is marketable at the current snapshot. Check if it
                # appeared here for the first time (no prior snapshot) or if
                # the previous snapshot had a DIFFERENT ask.
                if current_idx == 0:
                    # First-ever snapshot: the level just appeared → fleeting.
                    return True
                prev_ask = float(snap_ask[current_idx - 1])
                if np.isnan(prev_ask) or prev_ask > fill_price:
                    # Previous snapshot had no marketable ask at fill_price:
                    # the level is new. Check persistence duration.
                    current_ts = int(ts_arr[current_idx])
                    prev_ts = int(ts_arr[current_idx - 1])
                    if current_ts - prev_ts < persistence_ns:
                        return True  # level just appeared; within persistence window

    # SHR-94: forward-looking latency-window check.
    if idx >= len(ts_arr):
        return False  # no next snapshot
    next_ts = int(ts_arr[idx])
    # Only a snapshot that falls STRICTLY WITHIN the latency window can reveal
    # a fleeting level. A snapshot AFTER arrival (next_ts > now_ns + latency_ns)
    # means the book was stable through the latency window — fill is valid.
    arrival_ns = now_ns + latency_ns
    if next_ts > arrival_ns:
        return False
    # The next snapshot is within [now_ns, arrival_ns]. Check if the ask at
    # that snapshot is worse (higher) than fill_price — fleeting!
    next_ask = float(snap_ask[idx])
    if np.isnan(next_ask):
        return False  # no ask recorded at that snapshot — can't tell; allow fill
    return next_ask > fill_price


def _is_fleeting_bid(
    sym: str,
    fill_price: float,
    now_ns: int,
    latency_ns: int,
    snap_best_bid_per_leg: dict[str, np.ndarray],
    book_ts_per_leg: dict[str, np.ndarray],
    book_idx: dict[str, int],
    persistence_ns: int = 0,
) -> bool:
    """SHR-94 + SHR-89b: was the bid at ``fill_price`` a fleeting level?

    Symmetric to ``_is_fleeting_ask`` for the sell side.

    **SHR-94:** next snapshot within latency window shows bid BELOW fill_price.
    **SHR-89b:** bid just appeared in the current snapshot (not in the prior
    one) and has been present for fewer than ``persistence_ns`` nanoseconds.
    """
    snap_bid = snap_best_bid_per_leg.get(sym)
    ts_arr = book_ts_per_leg.get(sym)
    if snap_bid is None or ts_arr is None or len(snap_bid) == 0 or len(ts_arr) == 0:
        return False
    idx = book_idx.get(sym, 0)

    # SHR-89b: wall-clock persistence check (bid side).
    if persistence_ns > 0:
        current_idx = idx - 1
        if current_idx >= 0:
            current_bid = float(snap_bid[current_idx])
            if not np.isnan(current_bid) and current_bid >= fill_price:
                if current_idx == 0:
                    return True
                prev_bid = float(snap_bid[current_idx - 1])
                if np.isnan(prev_bid) or prev_bid < fill_price:
                    current_ts = int(ts_arr[current_idx])
                    prev_ts = int(ts_arr[current_idx - 1])
                    if current_ts - prev_ts < persistence_ns:
                        return True

    # SHR-94: forward-looking latency-window check.
    if idx >= len(ts_arr):
        return False
    next_ts = int(ts_arr[idx])
    arrival_ns = now_ns + latency_ns
    if next_ts > arrival_ns:
        return False
    next_bid = float(snap_bid[idx])
    if np.isnan(next_bid):
        return False
    return next_bid < fill_price


def _hedge_avg_entry(
    prev_qty: float, prev_avg: float, fill_side: str, fill_size: float, fill_price: float,
) -> tuple[float, float]:
    """Return (new_qty, new_avg_entry) after applying a hedge fill (SHR-55).

    Qty-weighted average when adding in the same direction, basis preserved when
    reducing, and reset to the fill price when the position flips through zero.
    Replaces the bug where avg_entry was overwritten with the latest fill price
    on every top-up.
    """
    add = fill_size if fill_side == "buy" else -fill_size
    new_qty = prev_qty + add
    if prev_qty == 0.0:
        return new_qty, fill_price
    if (prev_qty > 0) == (add > 0):  # growing in the same direction
        new_avg = (prev_avg * abs(prev_qty) + fill_price * fill_size) / (
            abs(prev_qty) + fill_size
        )
        return new_qty, new_avg
    if abs(add) < abs(prev_qty):  # partial reduction — basis unchanged
        return new_qty, prev_avg
    if new_qty == 0.0:  # fully closed
        return 0.0, 0.0
    return new_qty, fill_price  # flipped past zero — new basis is the fill price


def _hedge_mtm_fill(symbol: str, qty: float, mark_px: float) -> Fill:
    """Closing Fill that marks an open hedge residual to ``mark_px`` (the last
    observed hedge mid) at end-of-data, mirroring binary settlement (SHR-55).

    No slippage/fee — this is a mark-to-market of the residual at fair value,
    not a modelled liquidation, so realized PnL reflects the hedge's true
    economic value rather than only its opening leg.
    """
    side = "sell" if qty > 0 else "buy"
    return Fill(
        cloid=f"hedge_settle:{symbol}",
        symbol=symbol,
        side=side,
        price=mark_px,
        size=abs(qty),
        fee=0.0,
        partial=False,
        is_hedge=True,
    )


def _settle_px_for_outcome(pos: Position, q: object, outcome: str) -> float:
    """Binary-leg settlement payoff, via the shared
    :func:`position_math.settlement_payoff_price` (SHR-88).

    The held leg's index (leg[0] = YES, leg[1] = NO) is the position side; the
    venue/recorded resolved ``outcome`` supplies the winning (settled) side
    index — ``yes`` -> 0, ``no`` -> 1. Routing through the shared function makes
    the sim book payoffs identically to the live engine and, critically, derives
    the winner from the resolved outcome rather than re-deriving a YES winner.
    An unrecognised outcome, or a position whose leg isn't one of the question's
    legs, can never match the settled side and settles worthless (0.0) — exactly
    as before.
    """
    legs = q.leg_symbols  # type: ignore[attr-defined]
    yes_leg = legs[0] if legs else ""
    no_leg = legs[1] if len(legs) > 1 else ""
    if pos.symbol == yes_leg:
        position_side_idx = 0
    elif pos.symbol == no_leg:
        position_side_idx = 1
    else:
        return 0.0  # held leg isn't part of this question — can't win
    if outcome == "yes":
        settled_side_idx = 0
    elif outcome == "no":
        settled_side_idx = 1
    else:
        return 0.0  # unresolved / unknown outcome settles worthless
    return settlement_payoff_price(position_side_idx, settled_side_idx)
