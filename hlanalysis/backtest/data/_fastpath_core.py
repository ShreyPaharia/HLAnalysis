"""Source-agnostic hftbacktest event-array assembly.

Shared by the HL HIP-4 (`_hl_hip4_fastpath`) and Polymarket (`_pm_fastpath`)
fast paths. Contains everything that does NOT care which venue the bytes came
from: the flat-column → `event_dtype` assembler, the numba stale-level diff,
the per-leg containers, and the reference OHLC resampler.

BUILD_VERSION is bumped whenever the assembly logic changes; it is part of the
event-array cache key so a logic change invalidates every cached entry.
"""
from __future__ import annotations

# Bump on ANY change to assembly semantics below (cache-invalidation tag).
# v2: cache key now folds in config_sig (reference resample dt + source mode).
# Bumped to orphan any pre-existing on-disk cache keyed WITHOUT config_sig —
# those could serve a bundle built at the wrong dt (a dt=60 bundle for a dt=5
# request), the sigma-inflation footgun.
BUILD_VERSION = 2

import logging
from dataclasses import dataclass

import numpy as np
from numba import njit

from hftbacktest.types import (
    BUY_EVENT,
    DEPTH_EVENT,
    EXCH_EVENT,
    LOCAL_EVENT,
    SELL_EVENT,
    TRADE_EVENT,
    event_dtype,
)

from ..core.events import ReferenceEvent, SettlementEvent

log = logging.getLogger(__name__)


@njit(cache=True)
def _diff_clears(
    ts: np.ndarray,
    px_flat: np.ndarray,
    offsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-snapshot, return (out_ts, out_px) of stale-level clear events.

    For each snapshot i, finds prices in ``px_flat[offsets[i-1]:offsets[i]]``
    that are absent from ``px_flat[offsets[i]:offsets[i+1]]`` (linear search
    — book depth is ≤20 levels). Emits one clear per stale price at the
    new snapshot's timestamp ``ts[i]``.

    Inputs are the flat numpy column arrays from Arrow. JIT-compiled because
    the Python equivalent (set diffs on 160k snapshots) was 4 s / leg.
    """
    n_snaps = len(ts)
    # Pre-allocate up to the previous-snapshot levels per step. Slightly
    # over-allocated (worst case = total levels), but tractable.
    capacity = int(offsets[-1])  # total levels across all snapshots
    out_ts = np.empty(capacity, dtype=np.int64)
    out_px = np.empty(capacity, dtype=np.float64)
    n_out = 0
    prev_start = 0
    prev_end = 0
    for i in range(n_snaps):
        new_start = int(offsets[i])
        new_end = int(offsets[i + 1])
        snap_ts = ts[i]
        for k in range(prev_start, prev_end):
            p = px_flat[k]
            found = False
            for m in range(new_start, new_end):
                if px_flat[m] == p:
                    found = True
                    break
            if not found:
                out_ts[n_out] = snap_ts
                out_px[n_out] = p
                n_out += 1
        prev_start = new_start
        prev_end = new_end
    return out_ts[:n_out], out_px[:n_out]


_DEFAULT_REFERENCE_RESAMPLE_NS = 60 * 1_000_000_000


def _resample_reference_rows(
    events: list[ReferenceEvent],
    *,
    resample_ns: int,
) -> list[ReferenceEvent]:
    """List-input twin of ``hl_hip4._resample_reference``. Aggregates
    consecutive ReferenceEvents into OHLC bars of width ``resample_ns``.
    See that function for rationale.
    """
    out: list[ReferenceEvent] = []
    if not events:
        return out
    cur_bucket: int | None = None
    h = l = c = 0.0
    last_ts = 0
    sym = "BTC"
    for ev in events:
        bucket = ev.ts_ns // resample_ns
        if cur_bucket is None:
            cur_bucket = bucket
            h, l, c = ev.high, ev.low, ev.close
            last_ts = ev.ts_ns
            sym = ev.symbol
        elif bucket != cur_bucket:
            out.append(ReferenceEvent(last_ts, sym, h, l, c))
            cur_bucket = bucket
            h, l, c = ev.high, ev.low, ev.close
            last_ts = ev.ts_ns
            sym = ev.symbol
        else:
            if ev.high > h:
                h = ev.high
            if ev.low < l:
                l = ev.low
            c = ev.close
            last_ts = ev.ts_ns
    out.append(ReferenceEvent(last_ts, sym, h, l, c))
    return out


@dataclass(frozen=True, slots=True)
class LegArrays:
    """Per-leg event array + the last book ts cursor needed by the runner."""

    events: np.ndarray  # event_dtype array, sorted by exch_ts
    book_ts: np.ndarray  # int64 array of snapshot timestamps (for stale-book gate)


@dataclass(frozen=True, slots=True)
class FastPathBundle:
    """Output of ``HLHip4DataSource.events_arrays(q)``: everything the runner
    needs to skip the dataclass-based ``events()`` iteration entirely.
    """

    leg_arrays: dict[str, LegArrays]
    reference_events: list[ReferenceEvent]
    settlement_events: list[SettlementEvent]


def build_leg_event_array_from_columns(
    book_cols: dict[str, np.ndarray] | None,
    trade_cols: dict[str, np.ndarray] | None,
) -> np.ndarray:
    """Build the hftbacktest ``event_dtype`` array from flat column inputs.

    Preserves the legacy ``_build_leg_event_array`` semantics exactly:
    inter-snapshot stale levels are emitted as per-level ``qty=0`` events
    (not via ``DEPTH_CLEAR_EVENT``), and current levels are emitted as
    per-level ``DEPTH_EVENT`` sets. We tried using ``DEPTH_CLEAR_EVENT`` to
    skip the diff entirely — depth-state sampling at 100 timestamps matched
    100/100, but real backtest fills diverged ($0.004/$0.07 per-share price
    shifts on the v1 fixture), so the diff stays. The Python set-diff loop
    runs once per snapshot; with arrow-backed flat numpy column inputs
    (instead of ``BookSnapshot`` dataclasses) it is fast enough.

    Output layout vs the legacy builder:
    - Per snapshot, in order: stale-bid clears (qty=0), stale-ask clears
      (qty=0), new-bid sets, new-ask sets. Identical to legacy.
    - Trades follow as ``TRADE_EVENT`` with BUY/SELL side flag.
    - Final ``np.argsort`` is stable by ``exch_ts``.
    """
    flag = EXCH_EVENT | LOCAL_EVENT
    bid_set_ev = DEPTH_EVENT | flag | BUY_EVENT
    ask_set_ev = DEPTH_EVENT | flag | SELL_EVENT

    ts_chunks: list[np.ndarray] = []
    ev_chunks: list[np.ndarray] = []
    px_chunks: list[np.ndarray] = []
    qty_chunks: list[np.ndarray] = []

    if book_cols is not None and len(book_cols["ts"]) > 0:
        ts = book_cols["ts"]
        bid_px = book_cols["bid_px"]
        bid_sz = book_cols["bid_sz"]
        bid_off = book_cols["bid_offsets"]
        ask_px = book_cols["ask_px"]
        ask_sz = book_cols["ask_sz"]
        ask_off = book_cols["ask_offsets"]

        # --- Stale-level clear events: numba-JIT'd per-snapshot diff.
        # Operates on flat numpy column arrays; linear search across ≤20-level
        # books is fast under nopython.
        bid_clear_ts, bid_clear_px = _diff_clears(ts, bid_px, bid_off)
        ask_clear_ts, ask_clear_px = _diff_clears(ts, ask_px, ask_off)
        if len(bid_clear_ts) > 0:
            ts_chunks.append(bid_clear_ts)
            ev_chunks.append(np.full(len(bid_clear_ts), bid_set_ev, dtype=np.uint64))
            px_chunks.append(bid_clear_px)
            qty_chunks.append(np.zeros(len(bid_clear_ts), dtype=np.float64))
        if len(ask_clear_ts) > 0:
            ts_chunks.append(ask_clear_ts)
            ev_chunks.append(np.full(len(ask_clear_ts), ask_set_ev, dtype=np.uint64))
            px_chunks.append(ask_clear_px)
            qty_chunks.append(np.zeros(len(ask_clear_ts), dtype=np.float64))

        # --- Per-level SET events: fully vectorised via np.repeat.
        bid_lens = np.diff(bid_off)
        ask_lens = np.diff(ask_off)
        ts_chunks.append(np.repeat(ts, bid_lens))
        ev_chunks.append(np.full(len(bid_px), bid_set_ev, dtype=np.uint64))
        px_chunks.append(bid_px)
        qty_chunks.append(bid_sz)
        ts_chunks.append(np.repeat(ts, ask_lens))
        ev_chunks.append(np.full(len(ask_px), ask_set_ev, dtype=np.uint64))
        px_chunks.append(ask_px)
        qty_chunks.append(ask_sz)

    if trade_cols is not None and len(trade_cols["ts"]) > 0:
        trade_ts = trade_cols["ts"]
        trade_px = trade_cols["px"]
        trade_sz = trade_cols["sz"]
        trade_side = trade_cols["side"]
        trade_ev = np.where(
            trade_side == "sell",
            np.uint64(TRADE_EVENT | flag | SELL_EVENT),
            np.uint64(TRADE_EVENT | flag | BUY_EVENT),
        ).astype(np.uint64)
        ts_chunks.append(trade_ts)
        ev_chunks.append(trade_ev)
        px_chunks.append(trade_px)
        qty_chunks.append(trade_sz)

    if not ts_chunks:
        return np.zeros(0, dtype=event_dtype)

    out_ts = np.concatenate(ts_chunks)
    out_ev = np.concatenate(ev_chunks)
    out_px = np.concatenate(px_chunks)
    out_qty = np.concatenate(qty_chunks)

    arr = np.zeros(len(out_ts), dtype=event_dtype)
    arr["exch_ts"] = out_ts
    arr["local_ts"] = out_ts
    arr["ev"] = out_ev
    arr["px"] = out_px
    arr["qty"] = out_qty
    arr = arr[np.argsort(arr["exch_ts"], kind="stable")]
    return arr


__all__ = [
    "BUILD_VERSION", "LegArrays", "FastPathBundle",
    "build_leg_event_array_from_columns", "_diff_clears",
    "_resample_reference_rows", "event_dtype",
]
