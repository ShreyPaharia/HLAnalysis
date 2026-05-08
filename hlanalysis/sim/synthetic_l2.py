# hlanalysis/sim/synthetic_l2.py
from __future__ import annotations

from dataclasses import dataclass

from .data.schemas import PMTrade


@dataclass(frozen=True, slots=True)
class L2Snapshot:
    ts_ns: int
    token_id: str
    bid_px: float
    bid_sz: float
    ask_px: float
    ask_sz: float


def trade_to_l2(trade: PMTrade, *, half_spread: float, depth: float) -> L2Snapshot:
    bid_px = max(0.0, trade.price - half_spread)
    ask_px = min(1.0, trade.price + half_spread)
    return L2Snapshot(
        ts_ns=trade.ts_ns,
        token_id=trade.token_id,
        bid_px=bid_px,
        bid_sz=depth,
        ask_px=ask_px,
        ask_sz=depth,
    )
