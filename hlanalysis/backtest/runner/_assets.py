"""hftbacktest asset-building helpers and book-state reader for the runner.

Extracted verbatim from ``hftbt_runner.py`` — no logic changes.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import hftbacktest as hb
import numpy as np
from hftbacktest.types import (
    BUY_EVENT,
    DEPTH_CLEAR_EVENT,
    EXCH_EVENT,
    LOCAL_EVENT,
    SELL_EVENT,
    event_dtype,
)

from hlanalysis.strategy.types import BookState

if TYPE_CHECKING:
    pass


def _sentinel_event_array(ts_ns: int) -> np.ndarray:
    """Single no-op DEPTH_CLEAR event at ``ts_ns``.

    Used to extend hftbacktest's data timeline to the question's end_ts_ns in
    event-driven scan mode (SHR-95). Without at least one event at/near end_ts,
    hbt.elapse() returns rc=1 (end-of-data) as soon as the last real event is
    consumed — preventing the max-interval ceiling from firing over a quiet
    window. This sentinel keeps the timeline alive without affecting book state.
    """
    arr = np.zeros(1, dtype=event_dtype)
    flag = EXCH_EVENT | LOCAL_EVENT
    arr[0]["ev"] = DEPTH_CLEAR_EVENT | flag | BUY_EVENT
    arr[0]["exch_ts"] = ts_ns
    arr[0]["local_ts"] = ts_ns
    return arr


def _initial_clear_array(start_ts_ns: int) -> np.ndarray:
    """Two depth-clear events at the start so the engine begins with an empty book.

    Without this an asset's first elapse can leave book sides in an undefined
    initial state. Cheap (2 events) and clearer than relying on the first
    snapshot to overwrite ghost levels.
    """
    arr = np.zeros(2, dtype=event_dtype)
    flag = EXCH_EVENT | LOCAL_EVENT
    arr[0]["ev"] = DEPTH_CLEAR_EVENT | flag | BUY_EVENT
    arr[0]["exch_ts"] = start_ts_ns
    arr[0]["local_ts"] = start_ts_ns
    arr[1]["ev"] = DEPTH_CLEAR_EVENT | flag | SELL_EVENT
    arr[1]["exch_ts"] = start_ts_ns
    arr[1]["local_ts"] = start_ts_ns
    return arr


def _latency_span(
    arr: np.ndarray, start_ts_ns: int | None, end_ts_ns: int | None
) -> tuple[int, int]:
    """Resolve the [start, end] ts window the latency model spans. Callers in the
    runner pass the question's start/end; low-level tests may omit them, in which
    case the window is derived from the event array's exch_ts range."""
    if start_ts_ns is not None and end_ts_ns is not None:
        return int(start_ts_ns), int(end_ts_ns)
    ts = arr["exch_ts"]
    derived_start = int(ts.min()) if len(ts) else 0
    derived_end = int(ts.max()) if len(ts) else derived_start
    return (
        int(start_ts_ns) if start_ts_ns is not None else derived_start,
        int(end_ts_ns) if end_ts_ns is not None else derived_end,
    )


def _build_asset(
    arr: np.ndarray, cfg: Any, *,
    start_ts_ns: int | None = None, end_ts_ns: int | None = None,
) -> Any:
    # SHR-79: partial_fill_exchange replaces no_partial_fill_exchange so IOC
    # orders fill only real recorded book depth and walk levels naturally.
    # SHR-89: the pluggable latency model (constant by default; a sampled δ
    # distribution when configured) wires the order RTT into the sim.
    start, end = _latency_span(arr, start_ts_ns, end_ts_ns)
    asset = hb.BacktestAsset().data(arr).linear_asset(1.0)
    asset = cfg.effective_latency_model().apply(asset, start_ts_ns=start, end_ts_ns=end)
    return (
        asset
        .risk_adverse_queue_model()
        .partial_fill_exchange()
        .trading_value_fee_model(0.0, 0.0)
        .tick_size(cfg.tick_size)
        .lot_size(cfg.lot_size)
        .last_trades_capacity(cfg.last_trades_capacity)
    )


def _build_hedge_asset(
    arr: np.ndarray, cfg: Any, *,
    start_ts_ns: int | None = None, end_ts_ns: int | None = None,
) -> Any:
    """Like _build_asset but uses hedge-specific tick/lot sizes."""
    start, end = _latency_span(arr, start_ts_ns, end_ts_ns)
    asset = hb.BacktestAsset().data(arr).linear_asset(1.0)
    asset = cfg.effective_latency_model().apply(asset, start_ts_ns=start, end_ts_ns=end)
    return (
        asset
        .risk_adverse_queue_model()
        .partial_fill_exchange()
        .trading_value_fee_model(0.0, 0.0)
        .tick_size(cfg.hedge_tick_size)
        .lot_size(cfg.hedge_lot_size)
        .last_trades_capacity(cfg.last_trades_capacity)
    )


def _book_state(
    hbt: Any, asset_no: int, symbol: str, *, last_l2_ts_ns: int = 0
) -> BookState | None:
    depth = hbt.depth(asset_no)
    bid = float(depth.best_bid)
    ask = float(depth.best_ask)
    bid_sz = float(depth.best_bid_qty)
    ask_sz = float(depth.best_ask_qty)
    has_bid = bid > 0.0
    has_ask = ask > 0.0
    if not (has_bid or has_ask):
        return None
    return BookState(
        symbol=symbol,
        bid_px=bid if has_bid else None,
        bid_sz=bid_sz if has_bid else None,
        ask_px=ask if has_ask else None,
        ask_sz=ask_sz if has_ask else None,
        last_trade_ts_ns=0,
        last_l2_ts_ns=last_l2_ts_ns,
    )
