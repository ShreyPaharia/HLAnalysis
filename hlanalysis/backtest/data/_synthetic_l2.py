"""Trade → single-level synthetic L2 snapshot.

Moved from `hlanalysis/sim/synthetic_l2.py`. Behaviour unchanged: a trade
becomes a flat (bid_px = price - half_spread, ask_px = price + half_spread)
one-level book at fixed depth. The `bid_px` is clamped to ≥ 0 and `ask_px`
to ≤ 1 (PM CLOB prices are in [0, 1]).

This module deliberately stays free of `hlanalysis.sim.*` imports — `sim/`
is scheduled for deletion in Task E.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class L2Snapshot:
    ts_ns: int
    token_id: str
    bid_px: float
    bid_sz: float
    ask_px: float
    ask_sz: float


@dataclass(frozen=True)
class LiquidityProfile:
    """Per-price-bucket liquidity calibration for the synthetic book builder.

    ``half_spread`` and ``depth`` are per-bucket lists indexed by
    ``int(clamp(p, 0, 1) / bucket_width)``.  A ``None`` entry means "no data
    for this bucket" and falls back to ``global_half_spread`` /
    ``global_depth``.

    Pass an instance as ``profile=`` to :func:`trade_to_l2` to use calibrated
    values instead of the flat legacy constants.
    """

    bucket_width: float
    half_spread: Sequence[float | None]
    depth: Sequence[float | None]
    global_half_spread: float
    global_depth: float

    def _bucket(self, p: float) -> int:
        clamped = max(0.0, min(1.0, p))
        idx = int(clamped / self.bucket_width)
        return min(idx, len(self.half_spread) - 1)

    def half_spread_at(self, p: float) -> float:
        v = self.half_spread[self._bucket(p)]
        return self.global_half_spread if v is None else v

    def depth_at(self, p: float) -> float:
        v = self.depth[self._bucket(p)]
        return self.global_depth if v is None else v


def trade_to_l2(
    *,
    ts_ns: int,
    token_id: str,
    price: float,
    half_spread: float,
    depth: float,
    profile: LiquidityProfile | None = None,
) -> L2Snapshot:
    if profile is not None:
        half_spread = profile.half_spread_at(price)
        depth = profile.depth_at(price)
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


__all__ = ["L2Snapshot", "LiquidityProfile", "trade_to_l2"]
