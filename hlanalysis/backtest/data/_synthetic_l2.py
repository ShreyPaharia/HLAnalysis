"""Trade → single-level synthetic L2 snapshot.

Moved from `hlanalysis/sim/synthetic_l2.py`. Behaviour unchanged: a trade
becomes a flat (bid_px = price - half_spread, ask_px = price + half_spread)
one-level book at fixed depth. The `bid_px` is clamped to ≥ 0 and `ask_px`
to ≤ 1 (PM CLOB prices are in [0, 1]).

This module deliberately stays free of `hlanalysis.sim.*` imports — `sim/`
is scheduled for deletion in Task E.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class L2Snapshot:
    ts_ns: int
    token_id: str
    bid_px: float
    bid_sz: float
    ask_px: float
    ask_sz: float


def trade_to_l2(
    *,
    ts_ns: int,
    token_id: str,
    price: float,
    half_spread: float,
    depth: float,
) -> L2Snapshot:
    bid_px = max(0.0, price - half_spread)
    ask_px = min(1.0, price + half_spread)
    return L2Snapshot(
        ts_ns=ts_ns,
        token_id=token_id,
        bid_px=bid_px,
        bid_sz=depth,
        ask_px=ask_px,
        ask_sz=depth,
    )


__all__ = ["L2Snapshot", "trade_to_l2"]
