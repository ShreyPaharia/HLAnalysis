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
from typing import Any, Literal

import duckdb
import numpy as np

from ..core.data_source import QuestionDescriptor
from ..core.events import ReferenceEvent, SettlementEvent, TradeEvent
from ._fastpath_core import (
    FastPathBundle,
    LegArrays,
    _diff_clears,  # noqa: F401  (kept importable for tests)
    _resample_reference_rows,
    build_leg_event_array_from_columns,
    snap_best_from_columns,
)

log = logging.getLogger(__name__)


_DEFAULT_REFERENCE_RESAMPLE_NS = 60 * 1_000_000_000


# ---------------------------------------------------------------------------
# Internal column readers
# ---------------------------------------------------------------------------


def _read_book_columns(
    con: duckdb.DuckDBPyConnection,
    glob: str,
    date_list: list[str],
    start_ns: int,
    end_ns: int,
) -> dict[str, np.ndarray] | None:
    tbl = con.sql(
        f"""
        SELECT exchange_ts, bid_px, bid_sz, ask_px, ask_sz
        FROM read_parquet('{glob}', hive_partitioning=1, union_by_name=true)
        WHERE date IN ({",".join(repr(d) for d in date_list)})
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
    con: duckdb.DuckDBPyConnection,
    glob: str,
    date_list: list[str],
    start_ns: int,
    end_ns: int,
) -> dict[str, np.ndarray] | None:
    tbl = con.sql(
        f"""
        SELECT exchange_ts, price, size, side
        FROM read_parquet('{glob}', hive_partitioning=1, union_by_name=true)
        WHERE date IN ({",".join(repr(d) for d in date_list)})
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
    con: duckdb.DuckDBPyConnection,
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
            WHERE date IN ({",".join(repr(d) for d in date_list)})
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
# Top-level fast path
# ---------------------------------------------------------------------------


def _leg_best_bid_ask_maxima(
    con: duckdb.DuckDBPyConnection,
    glob: str,
    date_list: list[str],
    start_ns: int,
    end_ns: int,
) -> tuple[float, float] | None:
    """Return ``(max_best_bid, max_best_ask)`` over the question window, or None
    when the leg has no in-window snapshots.

    ``max_best_bid`` = the highest the best (top-of-book) bid ever reached;
    ``max_best_ask`` = the highest the best (lowest) ask ever reached. These are
    the only book quantities the favorite gate can depend on (see
    ``_is_prunable_bucket_leg``). The aggregate is a metadata-light scan — no
    Arrow materialisation of the 20-level book — so it is far cheaper than the
    full ``_read_book_columns`` + event-array build it lets us skip.

    ``union_by_name=true`` tolerates empty-book leg files whose ``bid_px`` /
    ``ask_px`` arrays are NULL-typed (DuckDB infers the union schema from the
    first file otherwise and fails to cast a real ``DOUBLE[]`` to ``NULL[]``).
    """
    df = ",".join(repr(d) for d in date_list)
    sql = f"""
        SELECT max(list_max(bid_px)) AS max_best_bid,
               max(list_min(ask_px)) AS max_best_ask
        FROM read_parquet('{glob}', hive_partitioning=1, union_by_name=true)
        WHERE date IN ({df})
          AND exchange_ts >= {start_ns} AND exchange_ts < {end_ns}
        """
    row = con.sql(sql).fetchone()
    if row is None:
        return None
    mbid, mask = row
    if mbid is None and mask is None:
        return None
    return (
        float(mbid) if mbid is not None else float("-inf"),
        float(mask) if mask is not None else float("-inf"),
    )


def _is_prunable_bucket_leg(
    con: duckdb.DuckDBPyConnection,
    leg: str,
    idx: int,
    q: QuestionDescriptor,
    date_list: list[str],
    threshold: float,
    book_glob_for: Any,
) -> bool:
    """True iff this bucket leg can NEVER be entered/held/exited, so loading its
    book is wasted work and emitting an empty (no-quote) leg is decision-identical.

    Two provably-safe cases:

    1. **Odd index = NO leg.** ``theta_harvester._evaluate_entry`` restricts
       bucket candidate legs to even (YES) indices, so a NO leg is never entered
       → never held → never exited. Its book is only ever read by ``books.get``
       for a held leg or a candidate leg, neither of which a NO leg can be. Prune
       unconditionally. (The bucket settlement winner is derived from BTC +
       priceThresholds read straight from parquet in ``leg_payoff`` — independent
       of the bundle — so dropping a NO leg's events changes no payoff either.)

    2. **YES leg whose best bid AND best ask both stay below ``threshold``.** The
       favorite gate is ``_mid(book) >= favorite_threshold`` and in every branch
       ``_mid <= max(best_bid, best_ask)``. So if, over the whole window, the best
       bid and best ask never reach ``threshold``, the gate never fires → the leg
       is never entered. A leg with no book files / no in-window rows is likewise
       untradeable.
    """
    if idx % 2 == 1:
        return True
    glob = book_glob_for(leg)
    if not _glob_has_files(glob):
        return True
    maxima = _leg_best_bid_ask_maxima(con, glob, date_list, q.start_ts_ns, q.end_ts_ns)
    if maxima is None:
        return True
    max_best_bid, max_best_ask = maxima
    return max_best_bid < threshold and max_best_ask < threshold


def build_fast_path_bundle(
    *,
    con: duckdb.DuckDBPyConnection,
    q: QuestionDescriptor,
    date_list: list[str],
    book_glob_for: Any,
    trade_glob_for: Any,
    settlement_glob_for: Any,
    reference_rows: list[tuple],
    ref_event_kind: Literal["bbo", "mark"],
    reference_resample_ns: int = _DEFAULT_REFERENCE_RESAMPLE_NS,
    reference_ticks: Literal["bars", "raw"] = "bars",
    leg_prune_favorite_threshold: float | None = None,
) -> FastPathBundle:
    """Assemble the per-leg event arrays + reference + settlement events.

    The ``*_glob_for`` callables map a symbol to its parquet glob; they're
    passed in so we don't depend on the data source's private helpers.

    ``leg_prune_favorite_threshold`` (priceBucket only): when set, legs that can
    never be entered at this favorite threshold are emitted as empty (no-quote)
    legs WITHOUT decoding/replaying their book — a behavior-preserving speed-up
    for many-leg bucket questions (most middle/tail buckets never become
    favorites). See ``_is_prunable_bucket_leg`` for the safety argument. ``None``
    (default) loads every leg, exactly as before.
    """
    do_prune = leg_prune_favorite_threshold is not None and q.klass == "priceBucket"
    leg_arrays: dict[str, LegArrays] = {}
    # Trade events per leg: built alongside the event arrays so the runner can
    # drain them into MarketState for the recent_volume_usd gate (SHR-78).
    trade_events_per_leg: dict[str, list[TradeEvent]] = {}
    for leg_idx, leg in enumerate(q.leg_symbols):
        if do_prune and _is_prunable_bucket_leg(
            con, leg, leg_idx, q, date_list, leg_prune_favorite_threshold, book_glob_for
        ):
            # Untradeable leg → emit exactly what a no-book leg produces (empty
            # event array, empty book_ts / snap_best). Skips the expensive
            # DuckDB read + diff_clears + event assembly for this leg.
            leg_arrays[leg] = LegArrays(
                events=build_leg_event_array_from_columns(None, None),
                book_ts=np.zeros(0, dtype=np.int64),
                snap_best_ask=snap_best_from_columns(None)[0],
                snap_best_bid=snap_best_from_columns(None)[1],
            )
            trade_events_per_leg[leg] = []
            continue
        book_glob = book_glob_for(leg)
        trade_glob = trade_glob_for(leg)
        book_cols = (
            _read_book_columns(con, book_glob, date_list, q.start_ts_ns, q.end_ts_ns)
            if _glob_has_files(book_glob)
            else None
        )
        trade_cols = (
            _read_trade_columns(con, trade_glob, date_list, q.start_ts_ns, q.end_ts_ns)
            if _glob_has_files(trade_glob)
            else None
        )
        arr = build_leg_event_array_from_columns(book_cols, trade_cols)
        book_ts = book_cols["ts"] if book_cols is not None else np.zeros(0, dtype=np.int64)
        snap_best_ask, snap_best_bid = snap_best_from_columns(book_cols)
        leg_arrays[leg] = LegArrays(
            events=arr,
            book_ts=book_ts,
            snap_best_ask=snap_best_ask,
            snap_best_bid=snap_best_bid,
        )
        # Convert trade columns → TradeEvent dataclasses for volume accounting.
        # The event array already consumed these for depth simulation; here we
        # just need (ts, px, sz) — no extra parquet I/O.
        if trade_cols is not None:
            ts_arr = trade_cols["ts"]
            px_arr = trade_cols["px"]
            sz_arr = trade_cols["sz"]
            side_arr = trade_cols["side"]
            trade_events_per_leg[leg] = [
                TradeEvent(
                    ts_ns=int(ts_arr[i]),
                    symbol=leg,
                    side="buy" if side_arr[i] != "sell" else "sell",
                    price=float(px_arr[i]),
                    size=float(sz_arr[i]),
                )
                for i in range(len(ts_arr))
            ]
        else:
            trade_events_per_leg[leg] = []

    # Reference events: already fetched as flat rows in legacy path. Convert
    # here and — in "bars" mode (default) — resample to OHLC bars of width
    # ``reference_resample_ns`` so σ-annualization is honest. In "raw" mode
    # (SHR-93), skip the resampling and emit one ReferenceEvent per tick
    # (H=L=C=mid); the runner will bucket them via apply_reference_tick so
    # last_mark is the instantaneous raw price, matching the live engine path.
    ref_events_raw: list[ReferenceEvent] = []
    if ref_event_kind == "bbo":
        for ts, bid, ask in reference_rows:
            mid = (float(bid) + float(ask)) / 2.0
            ref_events_raw.append(ReferenceEvent(int(ts), "BTC", mid, mid, mid))
    else:
        for ts, px in reference_rows:
            p = float(px)
            ref_events_raw.append(ReferenceEvent(int(ts), "BTC", p, p, p))
    if reference_ticks == "raw":
        # Raw mode: emit ticks as-is (H=L=C already set above); no bucketing.
        ref_events: list[ReferenceEvent] = ref_events_raw
    else:
        # Bars mode (default): resample to OHLC bars. See hl_hip4.py::_resample_reference
        # for the generator-path twin and detailed rationale.
        ref_events = _resample_reference_rows(ref_events_raw, resample_ns=reference_resample_ns)

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
        trade_events_per_leg=trade_events_per_leg,
        reference_events_are_raw_ticks=(reference_ticks == "raw"),
    )


def _glob_has_files(glob: str) -> bool:
    from glob import glob as _glob

    return bool(_glob(glob, recursive=True))


__all__ = [
    "LegArrays",
    "FastPathBundle",
    "build_leg_event_array_from_columns",
    "build_fast_path_bundle",
    "_is_prunable_bucket_leg",
    "_leg_best_bid_ask_maxima",
]
