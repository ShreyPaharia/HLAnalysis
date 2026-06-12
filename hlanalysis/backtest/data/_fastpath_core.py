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
# Also bump on a change to the on-disk serialization format, since old entries
# can no longer be read by the new loader; the version-prefixed filename makes
# stale-format entries get evicted on the next run.
# v2: cache key now folds in config_sig (reference resample dt + source mode).
# v3: on-disk format is column-split + delta-encoded timestamps (was the raw
#     64-byte interleaved struct); deflate compresses homogeneous columns far
#     better. Old v2 .npz are a different layout → orphaned + evicted.
# v4: assembler now orders events by (exch_ts, snapshot_index) instead of
#     exch_ts alone, fixing the per-timestamp final depth when a leg has
#     duplicate exchange timestamps (a price set then cleared within one ts).
#     Bit-identical for unique-ts feeds; bumped so any dup-ts bundle cached
#     under the old order is rebuilt.
# v5: on-disk format now persists per-leg trade events (tr_*__{ts,px,sz,side})
#     so a cache HIT restores the recent_volume_usd inputs (SHR-78). Pre-v5 npz
#     carried no trades → the volume gate silently read 0 on every cached run
#     (0 trades for any strategy with min_recent_volume_usd > 0). Bumped so
#     those trade-less bundles are evicted and rebuilt with trades.
# v6: on-disk format now persists per-snapshot best-ask/bid (sba_{i}, sbb_{i})
#     for the SHR-94 IOC marketability re-check. Pre-v6 npz lacked these arrays
#     → the fleeting-level veto would silently be skipped for cached bundles
#     (safe fall-back but less protective). Bumped so stale bundles are evicted
#     and rebuilt with the snap_best arrays.
# v7: on-disk format now persists ``reference_events_are_raw_ticks`` as the
#     ``__ref_raw_ticks__`` scalar. Pre-v7 npz always loaded the flag as False,
#     so a bundle built with ``reference_ticks="raw"`` (raw/event path) read back
#     as False on a cache hit, causing the runner to call ``apply_reference``
#     (bar path) instead of ``apply_reference_tick`` (raw-tick path). That
#     inflated σ on every cache-hit run → far fewer qualifying trades
#     (74 → 3 on v31 binary 2026-06-10).
BUILD_VERSION = 7

import logging
from dataclasses import dataclass, field

import numpy as np
from hftbacktest.types import (
    BUY_EVENT,
    DEPTH_EVENT,
    EXCH_EVENT,
    LOCAL_EVENT,
    SELL_EVENT,
    TRADE_EVENT,
    event_dtype,
)
from numba import njit

from hlanalysis.marketdata.ohlc import resample_ohlc

from ..core.events import BookSnapshot, ReferenceEvent, SettlementEvent, TradeEvent

log = logging.getLogger(__name__)


@njit(cache=True)
def _diff_clears(
    ts: np.ndarray,
    px_flat: np.ndarray,
    offsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-snapshot, return (out_ts, out_px, out_snap) of stale-level clears.

    For each snapshot i, finds prices in ``px_flat[offsets[i-1]:offsets[i]]``
    that are absent from ``px_flat[offsets[i]:offsets[i+1]]`` (linear search
    — book depth is ≤20 levels). Emits one clear per stale price at the
    new snapshot's timestamp ``ts[i]``; ``out_snap`` carries the snapshot
    index ``i`` so the assembler can order a clear AFTER same-timestamp sets
    from earlier snapshots (the legacy per-snapshot ordering).

    Inputs are the flat numpy column arrays from Arrow. JIT-compiled because
    the Python equivalent (set diffs on 160k snapshots) was 4 s / leg.
    """
    n_snaps = len(ts)
    # Pre-allocate up to the previous-snapshot levels per step. Slightly
    # over-allocated (worst case = total levels), but tractable.
    capacity = int(offsets[-1])  # total levels across all snapshots
    out_ts = np.empty(capacity, dtype=np.int64)
    out_px = np.empty(capacity, dtype=np.float64)
    out_snap = np.empty(capacity, dtype=np.int64)
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
                out_snap[n_out] = i
                n_out += 1
        prev_start = new_start
        prev_end = new_end
    return out_ts[:n_out], out_px[:n_out], out_snap[:n_out]


_DEFAULT_REFERENCE_RESAMPLE_NS = 60 * 1_000_000_000


def _resample_reference_rows(
    events: list[ReferenceEvent],
    *,
    resample_ns: int,
) -> list[ReferenceEvent]:
    """List-input twin of ``hl_hip4._resample_reference``. Aggregates
    consecutive ReferenceEvents into OHLC bars of width ``resample_ns``.

    Thin adapter over the canonical ``marketdata.ohlc.resample_ohlc`` (shared
    with the live engine and the generator path) so the three formerly-separate
    bucketers can no longer drift. The reference stream carries one symbol
    throughout (always ``"BTC"``), captured from the first event.
    """
    if not events:
        return []
    sym = events[0].symbol
    bars = resample_ohlc(
        ((ev.ts_ns, ev.high, ev.low, ev.close) for ev in events),
        bucket_ns=resample_ns,
    )
    return [ReferenceEvent(last_ts, sym, h, l, c) for last_ts, h, l, c in bars]


@dataclass(frozen=True, slots=True)
class LegArrays:
    """Per-leg event array + the last book ts cursor needed by the runner.

    ``snap_best_ask`` and ``snap_best_bid`` are parallel to ``book_ts``: the
    best ask / best bid price at each recorded snapshot.  The runner uses them
    for the SHR-94 IOC marketability re-check: before submitting an IOC, it
    looks at the *next* snapshot that falls within ``[now, now + latency_ns]``
    and vetoes fills on levels that are no longer marketable by the time the
    order would reach the exchange.  ``nan`` signals "no data" (the
    per-snapshot ask/bid wasn't recorded; skip the check for that snapshot).
    """

    events: np.ndarray  # event_dtype array, sorted by exch_ts
    book_ts: np.ndarray  # int64 array of snapshot timestamps (for stale-book gate)
    # Per-snapshot top-of-book for the SHR-94 fleeting-level check.
    # Parallel to book_ts; nan when the snapshot had no book data.
    snap_best_ask: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))  # float64, best ask price at each snapshot
    snap_best_bid: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))  # float64, best bid price at each snapshot


@dataclass(frozen=True, slots=True)
class FastPathBundle:
    """Output of ``HLHip4DataSource.events_arrays(q)``: everything the runner
    needs to skip the dataclass-based ``events()`` iteration entirely.

    ``trade_events_per_leg`` carries the per-symbol ``TradeEvent`` lists so the
    runner can drain them into ``MarketState`` for the ``recent_volume_usd``
    gate (SHR-78). They are NOT fed to hftbacktest (the depth engine already
    ingests them via the ``event_dtype`` array in ``leg_arrays``); they are
    only needed for the rolling-notional accumulator.
    """

    leg_arrays: dict[str, LegArrays]
    reference_events: list[ReferenceEvent]
    settlement_events: list[SettlementEvent]
    # Per-leg trade events for MarketState volume accounting (SHR-78).
    # Empty dict for sources that don't record trades (e.g. PM synthetic).
    trade_events_per_leg: dict[str, list[TradeEvent]] = None  # type: ignore[assignment]
    # SHR-93: when True, ``reference_events`` contains raw per-tick events
    # (H=L=C=mid, one per recorded tick) rather than pre-bucketed OHLC bars.
    # The runner routes them through ``MarketState.apply_reference_tick`` so
    # last_mark is the instantaneous raw price. Default False = bars (legacy).
    reference_events_are_raw_ticks: bool = False

    def __post_init__(self) -> None:
        # ``frozen=True`` + ``slots=True`` prevents direct assignment; use
        # object.__setattr__ to provide a safe empty-dict default.
        if self.trade_events_per_leg is None:
            object.__setattr__(self, "trade_events_per_leg", {})


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
    - Final ordering is by ``(exch_ts, snapshot_index)``: events from an earlier
      snapshot precede those of a later one at the SAME timestamp, so a price
      set by one snapshot and cleared by a later same-ts snapshot ends cleared
      (the legacy per-snapshot order). A plain ``argsort(exch_ts)`` would batch
      all clears ahead of all sets and corrupt the per-timestamp final depth
      when timestamps repeat (the in-memory sources emit sub-tick duplicates).
      For unique timestamps the ``snapshot_index`` is monotone with ``exch_ts``,
      so this is bit-identical to the old layout (HL/PM fast paths unchanged).
    """
    flag = EXCH_EVENT | LOCAL_EVENT
    bid_set_ev = DEPTH_EVENT | flag | BUY_EVENT
    ask_set_ev = DEPTH_EVENT | flag | SELL_EVENT

    ts_chunks: list[np.ndarray] = []
    ev_chunks: list[np.ndarray] = []
    px_chunks: list[np.ndarray] = []
    qty_chunks: list[np.ndarray] = []
    # Secondary sort key: the index of the snapshot that produced each event
    # (trades get a sentinel past the last snapshot so they sort after all book
    # events at the same timestamp, matching legacy's append-trades-last order).
    order_chunks: list[np.ndarray] = []
    n_snaps = 0

    if book_cols is not None and len(book_cols["ts"]) > 0:
        ts = book_cols["ts"]
        bid_px = book_cols["bid_px"]
        bid_sz = book_cols["bid_sz"]
        bid_off = book_cols["bid_offsets"]
        ask_px = book_cols["ask_px"]
        ask_sz = book_cols["ask_sz"]
        ask_off = book_cols["ask_offsets"]
        n_snaps = len(ts)

        # --- Stale-level clear events: numba-JIT'd per-snapshot diff.
        # Operates on flat numpy column arrays; linear search across ≤20-level
        # books is fast under nopython.
        bid_clear_ts, bid_clear_px, bid_clear_snap = _diff_clears(ts, bid_px, bid_off)
        ask_clear_ts, ask_clear_px, ask_clear_snap = _diff_clears(ts, ask_px, ask_off)
        if len(bid_clear_ts) > 0:
            ts_chunks.append(bid_clear_ts)
            ev_chunks.append(np.full(len(bid_clear_ts), bid_set_ev, dtype=np.uint64))
            px_chunks.append(bid_clear_px)
            qty_chunks.append(np.zeros(len(bid_clear_ts), dtype=np.float64))
            order_chunks.append(bid_clear_snap)
        if len(ask_clear_ts) > 0:
            ts_chunks.append(ask_clear_ts)
            ev_chunks.append(np.full(len(ask_clear_ts), ask_set_ev, dtype=np.uint64))
            px_chunks.append(ask_clear_px)
            qty_chunks.append(np.zeros(len(ask_clear_ts), dtype=np.float64))
            order_chunks.append(ask_clear_snap)

        # --- Per-level SET events: fully vectorised via np.repeat.
        bid_lens = np.diff(bid_off)
        ask_lens = np.diff(ask_off)
        snap_idx = np.arange(n_snaps, dtype=np.int64)
        ts_chunks.append(np.repeat(ts, bid_lens))
        ev_chunks.append(np.full(len(bid_px), bid_set_ev, dtype=np.uint64))
        px_chunks.append(bid_px)
        qty_chunks.append(bid_sz)
        order_chunks.append(np.repeat(snap_idx, bid_lens))
        ts_chunks.append(np.repeat(ts, ask_lens))
        ev_chunks.append(np.full(len(ask_px), ask_set_ev, dtype=np.uint64))
        px_chunks.append(ask_px)
        qty_chunks.append(ask_sz)
        order_chunks.append(np.repeat(snap_idx, ask_lens))

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
        # Sentinel order past the last snapshot → trades sort after all
        # same-timestamp book events (legacy appended trades last).
        order_chunks.append(np.full(len(trade_ts), n_snaps, dtype=np.int64))

    if not ts_chunks:
        return np.zeros(0, dtype=event_dtype)

    out_ts = np.concatenate(ts_chunks)
    out_ev = np.concatenate(ev_chunks)
    out_px = np.concatenate(px_chunks)
    out_qty = np.concatenate(qty_chunks)
    out_order = np.concatenate(order_chunks)

    arr = np.zeros(len(out_ts), dtype=event_dtype)
    arr["exch_ts"] = out_ts
    arr["local_ts"] = out_ts
    arr["ev"] = out_ev
    arr["px"] = out_px
    arr["qty"] = out_qty
    # lexsort is stable: primary key exch_ts, secondary key snapshot order.
    arr = arr[np.lexsort((out_order, out_ts))]
    return arr


def snap_best_from_columns(
    book_cols: dict[str, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract per-snapshot best-ask and best-bid from flat book columns.

    Returns ``(snap_best_ask, snap_best_bid)`` as float64 arrays of length
    ``len(book_cols['ts'])``.  Entries are ``nan`` when the snapshot carried
    no levels on that side (rare but possible on thin HIP-4 books).

    These arrays are parallel to ``book_ts`` and stored in ``LegArrays`` for
    the SHR-94 IOC marketability re-check in the runner.

    HL HIP-4 parquet stores asks sorted best-first (ascending: lowest ask
    first), so ``ask_px[ask_offsets[i]]`` is the best ask at snapshot ``i``.
    Bids are best-first descending, so ``bid_px[bid_offsets[i]]`` is the best
    bid.  Both conventions are enforced by the recorder's normaliser; in-memory
    sources (synthetic, pm_nba) also write best-first via ``_snapshots_to_columns``.
    """
    if book_cols is None or len(book_cols["ts"]) == 0:
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty
    n = len(book_cols["ts"])
    ask_px = book_cols["ask_px"]
    ask_off = book_cols["ask_offsets"]
    bid_px = book_cols["bid_px"]
    bid_off = book_cols["bid_offsets"]
    best_ask = np.full(n, np.nan, dtype=np.float64)
    best_bid = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        a_start = int(ask_off[i])
        a_end = int(ask_off[i + 1])
        if a_end > a_start:
            best_ask[i] = ask_px[a_start]
        b_start = int(bid_off[i])
        b_end = int(bid_off[i + 1])
        if b_end > b_start:
            best_bid[i] = bid_px[b_start]
    return best_ask, best_bid


def _snapshots_to_columns(snapshots: list[BookSnapshot]) -> dict[str, np.ndarray]:
    """Flatten in-memory ``BookSnapshot`` dataclasses into the variable-length
    flat-column layout ``build_leg_event_array_from_columns`` consumes.

    Level order is preserved exactly as the snapshots carry it (no re-sort): the
    in-memory sources already emit levels in the canonical order (synthetic =
    top-of-book; pm_nba / binance_perp = best-first), so funnelling through the
    shared assembler reproduces the legacy per-cell builder's output.
    """
    ts: list[int] = []
    bid_px: list[float] = []
    bid_sz: list[float] = []
    bid_off: list[int] = [0]
    ask_px: list[float] = []
    ask_sz: list[float] = []
    ask_off: list[int] = [0]
    for snap in snapshots:
        ts.append(snap.ts_ns)
        for px, sz in snap.bids:
            bid_px.append(px)
            bid_sz.append(sz)
        bid_off.append(len(bid_px))
        for px, sz in snap.asks:
            ask_px.append(px)
            ask_sz.append(sz)
        ask_off.append(len(ask_px))
    return {
        "ts": np.asarray(ts, dtype=np.int64),
        "bid_px": np.asarray(bid_px, dtype=np.float64),
        "bid_sz": np.asarray(bid_sz, dtype=np.float64),
        "bid_offsets": np.asarray(bid_off, dtype=np.int64),
        "ask_px": np.asarray(ask_px, dtype=np.float64),
        "ask_sz": np.asarray(ask_sz, dtype=np.float64),
        "ask_offsets": np.asarray(ask_off, dtype=np.int64),
    }


def _trades_to_columns(trades: list[TradeEvent]) -> dict[str, np.ndarray]:
    """Flatten in-memory ``TradeEvent`` dataclasses into trade columns."""
    return {
        "ts": np.asarray([t.ts_ns for t in trades], dtype=np.int64),
        "px": np.asarray([t.price for t in trades], dtype=np.float64),
        "sz": np.asarray([t.size for t in trades], dtype=np.float64),
        "side": np.asarray([t.side for t in trades], dtype=object),
    }


def build_leg_event_array_from_snapshots(
    snapshots: list[BookSnapshot], trades: list[TradeEvent]
) -> np.ndarray:
    """Assemble an hftbacktest ``event_dtype`` array from in-memory
    ``BookSnapshot`` / ``TradeEvent`` lists.

    The single entry point for the in-memory sources (synthetic, pm_nba,
    binance_perp) and the hedge leg: it adapts the dataclass lists into flat
    columns and funnels them through :func:`build_leg_event_array_from_columns`
    — the same vectorised assembler the HL/PM fast paths use. Replaces the
    formerly-duplicated per-cell ``_build_leg_event_array`` in the runner.

    Output is byte-for-byte identical to the legacy builder for single-level
    books and multiset-equivalent (fill-identical) in general — see
    ``build_leg_event_array_from_columns`` for the clear-ordering note.
    """
    book_cols = _snapshots_to_columns(snapshots) if snapshots else None
    trade_cols = _trades_to_columns(trades) if trades else None
    return build_leg_event_array_from_columns(book_cols, trade_cols)


__all__ = [
    "BUILD_VERSION", "LegArrays", "FastPathBundle",
    "build_leg_event_array_from_columns", "build_leg_event_array_from_snapshots",
    "snap_best_from_columns",
    "_diff_clears", "_resample_reference_rows", "event_dtype",
]
