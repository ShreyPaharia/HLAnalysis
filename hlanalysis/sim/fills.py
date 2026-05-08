# hlanalysis/sim/fills.py
from __future__ import annotations

from dataclasses import dataclass

from hlanalysis.strategy.types import BookState, OrderIntent


@dataclass(frozen=True, slots=True)
class FillModelConfig:
    slippage_bps: float       # added to ask on buy, subtracted from bid on sell
    fee_taker: float          # fraction (e.g. 0.02 for 2%)
    book_depth_assumption: float


@dataclass(frozen=True, slots=True)
class Fill:
    cloid: str
    symbol: str
    side: str
    price: float
    size: float
    fee: float
    partial: bool


def simulate_fill(intent: OrderIntent, book: BookState, cfg: FillModelConfig) -> Fill:
    if intent.side == "buy":
        if book.ask_px is None:
            return Fill(intent.cloid, intent.symbol, "buy", 0.0, 0.0, 0.0, partial=True)
        px = book.ask_px * (1.0 + cfg.slippage_bps / 1e4)
    else:
        if book.bid_px is None:
            return Fill(intent.cloid, intent.symbol, "sell", 0.0, 0.0, 0.0, partial=True)
        px = book.bid_px * (1.0 - cfg.slippage_bps / 1e4)
    size = min(intent.size, cfg.book_depth_assumption)
    fee = px * size * cfg.fee_taker
    return Fill(intent.cloid, intent.symbol, intent.side, px, size, fee, partial=size < intent.size)
