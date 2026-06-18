"""Unit tests for HL HIP-4 bucket leg-pruning + empty-book schema tolerance.

These exercise the two perf/robustness changes to the fast path:

1. ``union_by_name=true`` lets the column reader tolerate empty-book leg files
   whose ``bid_px``/``ask_px`` arrays are NULL-typed (``NULL[]``) — DuckDB
   otherwise infers the union schema from the first file and fails to cast a
   real ``DOUBLE[]`` to ``NULL[]`` (the 2026-06-15 multicoin-expansion crash).

2. Bucket leg-pruning skips decoding legs that can never be entered (NO/odd
   legs structurally; YES legs whose best bid/ask never reach the favorite
   threshold), emitting them as empty (no-quote) legs. This is decision-identical
   to a full load — proven here by asserting the KEPT leg's event array is
   byte-identical with pruning on vs off.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from hlanalysis.backtest.core.data_source import QuestionDescriptor
from hlanalysis.backtest.data._hl_hip4_fastpath import (
    _is_prunable_bucket_leg,
    _leg_best_bid_ask_maxima,
    _read_book_columns,
    build_fast_path_bundle,
)

DAY = "2026-05-10"
START = int(datetime(2026, 5, 10, 4, tzinfo=UTC).timestamp() * 1e9)
END = int(datetime(2026, 5, 10, 6, tzinfo=UTC).timestamp() * 1e9)
_PRED = "venue=hyperliquid/product_type=prediction_binary/mechanism=clob"


def _book_glob_for(root: Path):
    return lambda leg: str(root / _PRED / "event=book_snapshot" / f"symbol={leg}" / "**" / "*.parquet")


def _missing_glob_for(root: Path, event: str):
    return lambda leg: str(root / _PRED / f"event={event}" / f"symbol={leg}" / "**" / "*.parquet")


def _write_book(
    leg: str, root: Path, snaps: list[tuple], *, fname: str = "f.parquet", null_typed: bool = False
) -> None:
    """Write one book_snapshot parquet for ``leg``. snaps: (ts, bids, bsz, asks, asz)."""
    d = root / _PRED / "event=book_snapshot" / f"symbol={leg}" / f"date={DAY}"
    d.mkdir(parents=True, exist_ok=True)
    lt = pa.list_(pa.null()) if null_typed else pa.list_(pa.float64())
    tbl = pa.table(
        {
            "exchange_ts": pa.array([s[0] for s in snaps], pa.int64()),
            "bid_px": pa.array([s[1] for s in snaps], lt),
            "bid_sz": pa.array([s[2] for s in snaps], lt),
            "ask_px": pa.array([s[3] for s in snaps], lt),
            "ask_sz": pa.array([s[4] for s in snaps], lt),
            "date": pa.array([DAY] * len(snaps), pa.string()),
        }
    )
    pq.write_table(tbl, d / fname)


def _ts(i: int) -> int:
    return START + i * 1_000_000_000


# ---------------------------------------------------------------------------
# 1. Empty-book NULL[] schema tolerance
# ---------------------------------------------------------------------------


def test_read_book_columns_tolerates_null_typed_empty_file(tmp_path: Path) -> None:
    leg = "#100"
    # File the recorder wrote when the book was momentarily empty → NULL[] arrays.
    _write_book(leg=leg, root=tmp_path, fname="0_empty.parquet", null_typed=True, snaps=[(_ts(0), [], [], [], [])])
    # A normal DOUBLE[] file later in the window.
    _write_book(
        leg=leg,
        root=tmp_path,
        fname="1_real.parquet",
        snaps=[(_ts(1), [0.88], [10.0], [0.90], [10.0]), (_ts(2), [0.89], [10.0], [0.91], [10.0])],
    )
    con = duckdb.connect()
    glob = _book_glob_for(tmp_path)(leg)
    # Without union_by_name this raises ConversionException (DOUBLE -> NULL cast).
    cols = _read_book_columns(con, glob, [DAY], START, END)
    assert cols is not None
    # Empty NULL[] row contributes no levels; the two real snapshots survive.
    assert len(cols["ts"]) == 3
    assert cols["ask_px"].tolist() == [0.90, 0.91]


# ---------------------------------------------------------------------------
# 2. Pruning criterion (pure-ish, via DuckDB aggregate)
# ---------------------------------------------------------------------------


def _q(legs: tuple[str, ...]) -> QuestionDescriptor:
    return QuestionDescriptor(
        question_id="Qtest",
        question_idx=1,
        start_ts_ns=START,
        end_ts_ns=END,
        leg_symbols=legs,
        klass="priceBucket",
        underlying="BTC",
    )


def test_best_bid_ask_maxima_and_prunable(tmp_path: Path) -> None:
    con = duckdb.connect()
    bg = _book_glob_for(tmp_path)
    q = _q(("#100", "#101", "#110", "#111"))
    # YES favorite bucket: mid climbs to 0.89.
    _write_book("#100", tmp_path, [(_ts(0), [0.5], [9], [0.6], [9]), (_ts(1), [0.88], [9], [0.90], [9])])
    # YES tail bucket: never above 0.12.
    _write_book("#110", tmp_path, [(_ts(0), [0.08], [9], [0.10], [9]), (_ts(1), [0.10], [9], [0.12], [9])])
    # NO legs (odd indices) — books present but structurally never entered.
    _write_book("#101", tmp_path, [(_ts(0), [0.10], [9], [0.12], [9])])
    _write_book("#111", tmp_path, [(_ts(0), [0.88], [9], [0.90], [9])])

    mb = _leg_best_bid_ask_maxima(con, bg("#100"), [DAY], START, END)
    # max best-bid over snaps = 0.88; max best-ask = max(list_min(asks)) = max(0.60, 0.90) = 0.90.
    assert mb == (0.88, 0.90)
    mt = _leg_best_bid_ask_maxima(con, bg("#110"), [DAY], START, END)
    assert mt == (0.10, 0.12)

    # YES favorite kept; YES tail pruned on price; both NO legs pruned structurally.
    assert _is_prunable_bucket_leg(con, "#100", 0, q, [DAY], 0.85, bg) is False
    assert _is_prunable_bucket_leg(con, "#110", 2, q, [DAY], 0.85, bg) is True
    assert _is_prunable_bucket_leg(con, "#101", 1, q, [DAY], 0.85, bg) is True  # NO leg
    assert _is_prunable_bucket_leg(con, "#111", 3, q, [DAY], 0.85, bg) is True  # NO leg even if priced high


def test_no_book_leg_is_prunable(tmp_path: Path) -> None:
    con = duckdb.connect()
    bg = _book_glob_for(tmp_path)
    q = _q(("#100", "#101"))
    # #100 has no book files at all → untradeable.
    assert _is_prunable_bucket_leg(con, "#100", 0, q, [DAY], 0.85, bg) is True


# ---------------------------------------------------------------------------
# 3. build_fast_path_bundle: pruned legs empty, kept leg bit-identical
# ---------------------------------------------------------------------------


def test_bundle_prune_is_decision_identical(tmp_path: Path) -> None:
    bg = _book_glob_for(tmp_path)
    tg = _missing_glob_for(tmp_path, "trade")
    sg = _missing_glob_for(tmp_path, "settlement")
    legs = ("#100", "#101", "#110", "#111")
    q = _q(legs)
    _write_book("#100", tmp_path, [(_ts(i), [0.86 + 0.001 * i], [9], [0.90], [9]) for i in range(5)])
    _write_book("#110", tmp_path, [(_ts(i), [0.10], [9], [0.12], [9]) for i in range(5)])
    _write_book("#101", tmp_path, [(_ts(i), [0.10], [9], [0.12], [9]) for i in range(5)])
    _write_book("#111", tmp_path, [(_ts(i), [0.88], [9], [0.90], [9]) for i in range(5)])

    def build(threshold):
        con = duckdb.connect()
        try:
            return build_fast_path_bundle(
                con=con,
                q=q,
                date_list=[DAY],
                book_glob_for=bg,
                trade_glob_for=tg,
                settlement_glob_for=sg,
                reference_rows=[],
                ref_event_kind="mark",
                reference_resample_ns=2_000_000_000,
                leg_prune_favorite_threshold=threshold,
            )
        finally:
            con.close()

    full = build(None)
    pruned = build(0.85)

    # Full load: every leg with a book has events.
    assert all(len(full.leg_arrays[s].events) > 0 for s in legs)
    # Pruned: only the YES favorite (#100) survives; the rest are empty no-quote legs.
    assert len(pruned.leg_arrays["#100"].events) > 0
    for s in ("#101", "#110", "#111"):
        assert len(pruned.leg_arrays[s].events) == 0
        assert len(pruned.leg_arrays[s].book_ts) == 0
    # The KEPT leg's event array + snap arrays are byte-identical to the full load.
    np.testing.assert_array_equal(pruned.leg_arrays["#100"].events, full.leg_arrays["#100"].events)
    np.testing.assert_array_equal(pruned.leg_arrays["#100"].snap_best_ask, full.leg_arrays["#100"].snap_best_ask)


def test_prune_disabled_for_binary(tmp_path: Path) -> None:
    """A non-bucket question must never prune, even with a threshold set."""
    bg = _book_glob_for(tmp_path)
    tg = _missing_glob_for(tmp_path, "trade")
    sg = _missing_glob_for(tmp_path, "settlement")
    q = QuestionDescriptor(
        question_id="Qbin",
        question_idx=2,
        start_ts_ns=START,
        end_ts_ns=END,
        leg_symbols=("#100", "#101"),
        klass="priceBinary",
        underlying="BTC",
    )
    # Both legs cheap (would be pruned if this were a bucket) — must still load.
    _write_book("#100", tmp_path, [(_ts(0), [0.10], [9], [0.12], [9])])
    _write_book("#101", tmp_path, [(_ts(0), [0.10], [9], [0.12], [9])])
    con = duckdb.connect()
    bundle = build_fast_path_bundle(
        con=con,
        q=q,
        date_list=[DAY],
        book_glob_for=bg,
        trade_glob_for=tg,
        settlement_glob_for=sg,
        reference_rows=[],
        ref_event_kind="mark",
        leg_prune_favorite_threshold=0.85,
    )
    con.close()
    assert all(len(bundle.leg_arrays[s].events) > 0 for s in ("#100", "#101"))
