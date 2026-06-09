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
from pathlib import Path
from typing import Any, Literal

import duckdb
import numpy as np

from ._fastpath_core import (
    BUILD_VERSION,
    FastPathBundle,
    LegArrays,
    _diff_clears,  # noqa: F401  (kept importable for tests)
    _resample_reference_rows,
    build_leg_event_array_from_columns,
    snap_best_from_columns,
    event_dtype,
)

from ..core.data_source import QuestionDescriptor
from ..core.events import ReferenceEvent, SettlementEvent, TradeEvent

log = logging.getLogger(__name__)


_DEFAULT_REFERENCE_RESAMPLE_NS = 60 * 1_000_000_000


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
    reference_resample_ns: int = _DEFAULT_REFERENCE_RESAMPLE_NS,
) -> FastPathBundle:
    """Assemble the per-leg event arrays + reference + settlement events.

    The ``*_glob_for`` callables map a symbol to its parquet glob; they're
    passed in so we don't depend on the data source's private helpers.
    """
    leg_arrays: dict[str, LegArrays] = {}
    # Trade events per leg: built alongside the event arrays so the runner can
    # drain them into MarketState for the recent_volume_usd gate (SHR-78).
    trade_events_per_leg: dict[str, list[TradeEvent]] = {}
    for leg in q.leg_symbols:
        book_glob = book_glob_for(leg)
        trade_glob = trade_glob_for(leg)
        book_cols = _read_book_columns(con, book_glob, date_list, q.start_ts_ns, q.end_ts_ns) \
            if _glob_has_files(book_glob) else None
        trade_cols = _read_trade_columns(con, trade_glob, date_list, q.start_ts_ns, q.end_ts_ns) \
            if _glob_has_files(trade_glob) else None
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
    # here and resample to OHLC bars of width ``reference_resample_ns`` so
    # σ-annualization in the strategy (which assumes vol_sampling_dt_seconds
    # inter-sample spacing) is honest. Raw HL BBO/mark feeds tick ~1-6/s;
    # without bucketing the strategy's last-N returns span ~5-30s of price
    # action while the annualization treats them as `dt` apart. See
    # hl_hip4.py::_resample_reference for the legacy-path twin and rationale.
    ref_events_raw: list[ReferenceEvent] = []
    if ref_event_kind == "bbo":
        for ts, bid, ask in reference_rows:
            mid = (float(bid) + float(ask)) / 2.0
            ref_events_raw.append(ReferenceEvent(int(ts), "BTC", mid, mid, mid))
    else:
        for ts, px in reference_rows:
            p = float(px)
            ref_events_raw.append(ReferenceEvent(int(ts), "BTC", p, p, p))
    ref_events: list[ReferenceEvent] = _resample_reference_rows(
        ref_events_raw, resample_ns=reference_resample_ns
    )

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
