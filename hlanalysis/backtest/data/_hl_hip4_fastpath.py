"""Arrow-backed fast path for HL HIP-4 → hftbacktest event arrays.

The default ``HLHip4DataSource.events()`` path yields per-event ``BookSnapshot``
/ ``TradeEvent`` dataclasses; the runner then unpacks them in
``_build_leg_event_array``. For HL HIP-4 with 20-level books and ~160k
snapshots per leg, the dataclass round-trip dominates wall time even after the
per-cell ``_build_leg_event_array`` was vectorised.

This module gives the runner a shortcut: it reads each leg's
``book_snapshot`` / ``trade`` parquet via DuckDB → Arrow → flat numpy column
arrays, then builds the hftbacktest ``event_dtype`` array with the same
semantics as the legacy builder (per-snapshot ``qty=0`` clears for stale
levels + per-level ``DEPTH_EVENT`` sets for current levels). The set events
are emitted fully vectorised via ``np.repeat`` over the snapshot ts column;
only the inter-snapshot stale-level set diff runs in Python, and it operates
on flat numpy slices (no ``BookSnapshot`` allocation).

We also tried emitting ``DEPTH_CLEAR_EVENT`` per side per snapshot — that
fully vectorises the build and a Python depth-replay confirmed 100/100
sampled snapshot timestamps had identical post-state. But the actual
hftbacktest fills diverged by a few price ticks on the v1 fixture, so this
optimisation was abandoned. The diff-based path here is bit-identical to the
legacy builder.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import duckdb
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

from ..core.data_source import QuestionDescriptor
from ..core.events import ReferenceEvent, SettlementEvent

log = logging.getLogger(__name__)


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


# ---------------------------------------------------------------------------
# Internal column readers
# ---------------------------------------------------------------------------


def _read_book_columns(
    con: "duckdb.DuckDBPyConnection",
    glob: str,
    date_list: list[str],
    start_ns: int,
    end_ns: int,
) -> dict[str, np.ndarray] | None:
    tbl = con.sql(
        f"""
        SELECT exchange_ts, bid_px, bid_sz, ask_px, ask_sz
        FROM read_parquet('{glob}', hive_partitioning=1)
        WHERE date IN ({','.join(repr(d) for d in date_list)})
          AND exchange_ts >= {start_ns} AND exchange_ts < {end_ns}
        ORDER BY exchange_ts
        """
    ).to_arrow_table()
    if tbl.num_rows == 0:
        return None
    bid_px = tbl["bid_px"].combine_chunks()
    bid_sz = tbl["bid_sz"].combine_chunks()
    ask_px = tbl["ask_px"].combine_chunks()
    ask_sz = tbl["ask_sz"].combine_chunks()
    return {
        "ts": tbl["exchange_ts"].to_numpy(),
        "bid_px": bid_px.flatten().to_numpy(),
        "bid_sz": bid_sz.flatten().to_numpy(),
        "bid_offsets": bid_px.offsets.to_numpy().astype(np.int64, copy=False),
        "ask_px": ask_px.flatten().to_numpy(),
        "ask_sz": ask_sz.flatten().to_numpy(),
        "ask_offsets": ask_px.offsets.to_numpy().astype(np.int64, copy=False),
    }


def _read_trade_columns(
    con: "duckdb.DuckDBPyConnection",
    glob: str,
    date_list: list[str],
    start_ns: int,
    end_ns: int,
) -> dict[str, np.ndarray] | None:
    tbl = con.sql(
        f"""
        SELECT exchange_ts, price, size, side
        FROM read_parquet('{glob}', hive_partitioning=1)
        WHERE date IN ({','.join(repr(d) for d in date_list)})
          AND exchange_ts >= {start_ns} AND exchange_ts < {end_ns}
        ORDER BY exchange_ts
        """
    ).to_arrow_table()
    if tbl.num_rows == 0:
        return None
    return {
        "ts": tbl["exchange_ts"].to_numpy(),
        "px": tbl["price"].to_numpy(),
        "sz": tbl["size"].to_numpy(),
        # `side` column may be Utf8 with nulls; to_pylist normalises both.
        "side": np.asarray(tbl["side"].to_pylist(), dtype=object),
    }


def _read_settlement_columns(
    con: "duckdb.DuckDBPyConnection",
    glob: str,
    date_list: list[str],
    start_ns: int,
    end_ns: int,
) -> list[tuple[int, int]] | None:
    """Return list of (effective_ts, side_idx) tuples, or None if no files / error."""
    try:
        rs = con.sql(
            f"""
            SELECT exchange_ts, settle_ts, settled_side_idx
            FROM read_parquet('{glob}', hive_partitioning=1)
            WHERE date IN ({','.join(repr(d) for d in date_list)})
              AND exchange_ts >= {start_ns} AND exchange_ts <= {end_ns}
            ORDER BY exchange_ts
            """
        ).fetchall()
    except duckdb.Error as e:
        log.warning("settlement read failed: %s", e)
        return None
    if not rs:
        return None
    return [(int(settle_ts or ts), int(side_idx)) for ts, settle_ts, side_idx in rs]


# ---------------------------------------------------------------------------
# Vectorised builders
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Top-level fast path
# ---------------------------------------------------------------------------


def build_fast_path_bundle(
    *,
    con: "duckdb.DuckDBPyConnection",
    q: QuestionDescriptor,
    date_list: list[str],
    book_glob_for: Any,
    trade_glob_for: Any,
    settlement_glob_for: Any,
    reference_rows: list[tuple],
    ref_event_kind: Literal["bbo", "mark"],
) -> FastPathBundle:
    """Assemble the per-leg event arrays + reference + settlement events.

    The ``*_glob_for`` callables map a symbol to its parquet glob; they're
    passed in so we don't depend on the data source's private helpers.
    """
    leg_arrays: dict[str, LegArrays] = {}
    for leg in q.leg_symbols:
        book_glob = book_glob_for(leg)
        trade_glob = trade_glob_for(leg)
        book_cols = _read_book_columns(con, book_glob, date_list, q.start_ts_ns, q.end_ts_ns) \
            if _glob_has_files(book_glob) else None
        trade_cols = _read_trade_columns(con, trade_glob, date_list, q.start_ts_ns, q.end_ts_ns) \
            if _glob_has_files(trade_glob) else None
        arr = build_leg_event_array_from_columns(book_cols, trade_cols)
        book_ts = book_cols["ts"] if book_cols is not None else np.zeros(0, dtype=np.int64)
        leg_arrays[leg] = LegArrays(events=arr, book_ts=book_ts)

    # Reference events: already fetched as flat rows in legacy path. Convert here.
    ref_events: list[ReferenceEvent] = []
    if ref_event_kind == "bbo":
        for ts, bid, ask in reference_rows:
            mid = (float(bid) + float(ask)) / 2.0
            ref_events.append(ReferenceEvent(int(ts), "BTC", mid, mid, mid))
    else:
        for ts, px in reference_rows:
            p = float(px)
            ref_events.append(ReferenceEvent(int(ts), "BTC", p, p, p))

    # Settlement events: per-leg.
    settle_events: list[SettlementEvent] = []
    for leg in q.leg_symbols:
        glob = settlement_glob_for(leg)
        if not _glob_has_files(glob):
            continue
        rows = _read_settlement_columns(con, glob, date_list, q.start_ts_ns, q.end_ts_ns)
        if rows is None:
            continue
        for t, side_idx in rows:
            outcome: Literal["yes", "no", "unknown"] = "yes" if side_idx == 0 else "no"
            settle_events.append(SettlementEvent(int(t), q.question_idx, outcome))
    settle_events.sort(key=lambda e: e.ts_ns)

    return FastPathBundle(
        leg_arrays=leg_arrays,
        reference_events=ref_events,
        settlement_events=settle_events,
    )


def _glob_has_files(glob: str) -> bool:
    from glob import glob as _glob
    return bool(_glob(glob, recursive=True))


__all__ = [
    "LegArrays",
    "FastPathBundle",
    "build_leg_event_array_from_columns",
    "build_fast_path_bundle",
]
