"""Polymarket recorded-book → flat numpy columns for the shared assembler.

Only recorded mode. Produces the same {ts, bid_px, bid_sz, bid_offsets,
ask_px, ask_sz, ask_offsets} / {ts, px, sz, side} dicts that
``_fastpath_core.build_leg_event_array_from_columns`` consumes, with
within-snapshot level ordering normalised to MATCH the legacy ``events()``
path:

  bids sorted px DESC (best = max), asks sorted px ASC (best = min)

so fills stay bit-equivalent to the legacy dataclass path.
"""
from __future__ import annotations

import numpy as np

from ._fastpath_core import (
    FastPathBundle,
    LegArrays,
    build_leg_event_array_from_columns,
)


def read_pm_book_columns(
    book_glob: str, start_ns: int, end_ns: int
) -> dict[str, np.ndarray] | None:
    """Read the recorded PM book parquet for one token leg and return flat
    column arrays suitable for ``build_leg_event_array_from_columns``.

    Level ordering is normalised to match ``_load_recorded_book`` /
    ``_normalize_levels``: bids px DESC, asks px ASC. Each level's size is
    carried alongside its price.

    Returns ``None`` when there is no parquet coverage matching the glob or
    no rows fall in the requested window.
    """
    import duckdb
    from glob import glob as _glob

    if not _glob(book_glob, recursive=True):
        return None

    con = duckdb.connect()
    try:
        rows = con.sql(
            f"""
            SELECT exchange_ts, bid_px, bid_sz, ask_px, ask_sz
            FROM read_parquet('{book_glob}', hive_partitioning=1)
            WHERE exchange_ts >= {start_ns} AND exchange_ts < {end_ns}
            ORDER BY exchange_ts
            """
        ).fetchall()
    finally:
        con.close()

    if not rows:
        return None

    ts: list[int] = []
    bpx: list[float] = []
    bsz: list[float] = []
    boff: list[int] = [0]
    apx: list[float] = []
    asz: list[float] = []
    aoff: list[int] = [0]

    for exch_ts, bid_px, bid_sz, ask_px, ask_sz in rows:
        # Normalise level ordering to match _normalize_levels:
        #   bids DESC by px (best bid = highest price)
        #   asks ASC by px  (best ask = lowest price)
        # Sizes travel with their price level.
        b_levels = sorted(
            zip(bid_px or [], bid_sz or []),
            key=lambda x: -float(x[0]),
        )
        a_levels = sorted(
            zip(ask_px or [], ask_sz or []),
            key=lambda x: float(x[0]),
        )
        ts.append(int(exch_ts))
        bpx.extend(float(p) for p, _ in b_levels)
        bsz.extend(float(s) for _, s in b_levels)
        boff.append(len(bpx))
        apx.extend(float(p) for p, _ in a_levels)
        asz.extend(float(s) for _, s in a_levels)
        aoff.append(len(apx))

    return {
        "ts": np.asarray(ts, dtype=np.int64),
        "bid_px": np.asarray(bpx, dtype=np.float64),
        "bid_sz": np.asarray(bsz, dtype=np.float64),
        "bid_offsets": np.asarray(boff, dtype=np.int64),
        "ask_px": np.asarray(apx, dtype=np.float64),
        "ask_sz": np.asarray(asz, dtype=np.float64),
        "ask_offsets": np.asarray(aoff, dtype=np.int64),
    }


def read_pm_trade_columns(
    trades: list,  # list[_RawTrade]
    token_id: str,
) -> dict[str, np.ndarray] | None:
    """Build trade columns for one leg from already-loaded ``_RawTrade`` rows.

    Filters to ``token_id`` and sorts by ``ts_ns``.  Returns ``None`` when no
    matching trades exist.

    ``_RawTrade`` attrs used: ``.ts_ns``, ``.price``, ``.size``, ``.side``
    (string ``"buy"`` | ``"sell"``).  The assembler's side branch is
    ``side == "sell"`` → ``SELL_EVENT``; everything else → ``BUY_EVENT``.
    """
    rows = [t for t in trades if t.token_id == token_id]
    rows.sort(key=lambda t: t.ts_ns)
    if not rows:
        return None
    return {
        "ts": np.asarray([t.ts_ns for t in rows], dtype=np.int64),
        "px": np.asarray([t.price for t in rows], dtype=np.float64),
        "sz": np.asarray([t.size for t in rows], dtype=np.float64),
        "side": np.asarray([t.side for t in rows], dtype=object),
    }


def build_pm_fast_path_bundle(
    *,
    q,
    book_glob_for,
    trades: list,
    reference_events: list,
    settlement_events: list,
) -> FastPathBundle:
    """Assemble a :class:`FastPathBundle` for a PM recorded-mode question.

    Parameters
    ----------
    q:
        ``QuestionDescriptor`` whose ``leg_symbols`` and timestamp window are
        used.
    book_glob_for:
        Callable ``token_id → glob_pattern`` pointing at the recorded
        ``book_snapshot`` parquet partitions for that token.
    trades:
        All ``_RawTrade`` rows for the question (pre-loaded by the caller).
    reference_events:
        ``ReferenceEvent`` list (klines or BBO, already built by the caller).
    settlement_events:
        ``SettlementEvent`` list (one per leg, already built by the caller).
    """
    leg_arrays: dict[str, LegArrays] = {}
    for leg in q.leg_symbols:
        book_cols = read_pm_book_columns(
            book_glob_for(leg), q.start_ts_ns, q.end_ts_ns
        )
        trade_cols = read_pm_trade_columns(trades, leg)
        arr = build_leg_event_array_from_columns(book_cols, trade_cols)
        bts = (
            book_cols["ts"]
            if book_cols is not None
            else np.zeros(0, dtype=np.int64)
        )
        leg_arrays[leg] = LegArrays(events=arr, book_ts=bts)
    return FastPathBundle(
        leg_arrays=leg_arrays,
        reference_events=reference_events,
        settlement_events=settlement_events,
    )


__all__ = [
    "read_pm_book_columns",
    "read_pm_trade_columns",
    "build_pm_fast_path_bundle",
]
