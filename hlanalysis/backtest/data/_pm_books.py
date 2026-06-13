"""Polymarket book assembly helpers.

Covers:
- _RawTrade dataclass + _parse_clob_trade (trade row parsing)
- _normalize_levels (recorded book level normalisation)
- _book_from_l2 (L2Snapshot → BookSnapshot conversion)
- Module-level constants: _P_CLIP_LO, _P_CLIP_HI, _HALF_SPREAD_DEFAULT,
  _DEPTH_DEFAULT, _PM_BOOK_DATA_SUBPATH, _PM_SETTLEMENT_DATA_SUBPATH

Extracted verbatim from polymarket.py — no logic changes.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.events import BookSnapshot
from ._synthetic_l2 import L2Snapshot

_HALF_SPREAD_DEFAULT = 0.005
_DEPTH_DEFAULT = 10_000.0
_P_CLIP_LO = 1e-6
_P_CLIP_HI = 1.0 - 1e-6

# Recorded PM L2 book partitions (native recorder; coverage starts 2026-05-27).
# Symbol partition is the PM token_id, joining to manifest yes/no token ids.
_PM_BOOK_DATA_SUBPATH = "venue=polymarket/product_type=prediction_binary/mechanism=clob/event=book_snapshot"
# Recorded PM settlement (on-chain redemption): the winning leg token redeems
# at settle_price≈1.0. Authoritative for the resolved outcome when present.
_PM_SETTLEMENT_DATA_SUBPATH = "venue=polymarket/product_type=prediction_binary/mechanism=clob/event=settlement"


# ---- Trade row parsing -------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _RawTrade:
    ts_ns: int
    token_id: str
    side: str  # "buy" | "sell"
    price: float
    size: float


def _parse_clob_trade(row: dict) -> _RawTrade | None:
    try:
        ts = float(row.get("timestamp", 0))
        return _RawTrade(
            ts_ns=int(ts * 1e9),
            token_id=str(row["asset"]),
            side="buy" if str(row.get("side", "")).upper() == "BUY" else "sell",
            price=float(row["price"]),
            size=float(row["size"]),
        )
    except (KeyError, ValueError, TypeError):
        return None


# ---- Book level helpers ------------------------------------------------------


def _normalize_levels(
    px: list[float] | None,
    sz: list[float] | None,
    *,
    descending: bool,
) -> tuple[tuple[float, float], ...]:
    """Pair recorded (px, sz) levels and sort by price.

    ``descending=True`` for bids (best = max), ``False`` for asks (best = min).
    Tolerates ``None``/empty/ragged arrays (one side may be empty in a recorded
    snapshot). Sizes travel with their price level.
    """
    if not px:
        return ()
    sz = sz or []
    levels = [(float(px[i]), float(sz[i]) if i < len(sz) else 0.0) for i in range(len(px))]
    levels.sort(key=lambda lv: lv[0], reverse=descending)
    return tuple(levels)


def _book_from_l2(s: L2Snapshot) -> BookSnapshot:
    return BookSnapshot(
        ts_ns=s.ts_ns,
        symbol=s.token_id,
        bids=((s.bid_px, s.bid_sz),),
        asks=((s.ask_px, s.ask_sz),),
    )
