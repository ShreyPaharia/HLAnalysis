"""PM bucket leg-pruning — decode/replay only the legs theta could ever enter.

theta_harvester only ENTERS the favorite YES leg of a bucket (even index, mid ≥
favorite_threshold), so the vast majority of a PM multistrike bucket's 22–30 legs
are never tradeable yet the recorded-book fast path decoded/replayed every one.
``build_pm_fast_path_bundle(leg_prune_favorite_threshold=…)`` emits those legs as
empty (no-quote) legs without reading their book parquet.

This mirrors the HL HIP-4 prune (same shared strategy logic). Decision-identical
is proven here by asserting the KEPT leg's event array is byte-identical with
pruning on vs off, and that prunable legs become empty — pruned legs are exactly
the ones the strategy's even-index + favorite-gate filter would have dropped, so
the favorite SELECTION (and thus every fill/edge/decision) is unchanged.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from hlanalysis.backtest.core.data_source import QuestionDescriptor
from hlanalysis.backtest.data._pm_books import _PM_BOOK_DATA_SUBPATH
from hlanalysis.backtest.data._pm_fastpath import (
    _is_prunable_pm_bucket_leg,
    _pm_leg_best_bid_ask_maxima,
    build_pm_fast_path_bundle,
)

DAY = "2026-06-06"
START = int(datetime(2026, 6, 6, 4, tzinfo=UTC).timestamp() * 1e9)
END = int(datetime(2026, 6, 6, 6, tzinfo=UTC).timestamp() * 1e9)


def _book_glob_for(root: Path):
    return lambda token: str(root / _PM_BOOK_DATA_SUBPATH / f"symbol={token}" / "**" / "*.parquet")


def _write_pm_book(token: str, root: Path, snaps: list[tuple]) -> None:
    """Write one recorded book_snapshot parquet. snaps: (ts, bids, bsz, asks, asz)."""
    d = root / _PM_BOOK_DATA_SUBPATH / f"symbol={token}" / f"date={DAY}"
    d.mkdir(parents=True, exist_ok=True)
    lt = pa.list_(pa.float64())
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
    pq.write_table(tbl, d / "f.parquet")


def _ts(i: int) -> int:
    return START + i * 1_000_000_000


def _q(legs: tuple[str, ...], *, klass: str = "priceBucket") -> QuestionDescriptor:
    return QuestionDescriptor(
        question_id="Qpm",
        question_idx=1,
        start_ts_ns=START,
        end_ts_ns=END,
        leg_symbols=legs,
        klass=klass,
        underlying="BTC",
    )


def test_maxima_and_prunable_predicate(tmp_path: Path) -> None:
    bg = _book_glob_for(tmp_path)
    q = _q(("y0", "n0", "y1", "n1"))
    # YES favorite (even idx 0): mid climbs to 0.89.
    _write_pm_book("y0", tmp_path, [(_ts(0), [0.5], [9], [0.6], [9]), (_ts(1), [0.88], [9], [0.90], [9])])
    # YES tail (even idx 2): never above 0.12.
    _write_pm_book("y1", tmp_path, [(_ts(0), [0.08], [9], [0.10], [9]), (_ts(1), [0.10], [9], [0.12], [9])])
    # NO legs (odd) — books present but structurally never entered.
    _write_pm_book("n0", tmp_path, [(_ts(0), [0.10], [9], [0.12], [9])])
    _write_pm_book("n1", tmp_path, [(_ts(0), [0.88], [9], [0.90], [9])])

    # max best-bid = 0.88; max best-ask = max(list_min(asks)) = max(0.60, 0.90) = 0.90.
    assert _pm_leg_best_bid_ask_maxima(bg("y0"), START, END) == (0.88, 0.90)
    assert _pm_leg_best_bid_ask_maxima(bg("y1"), START, END) == (0.10, 0.12)

    assert _is_prunable_pm_bucket_leg("y0", 0, q, 0.85, bg) is False  # YES favorite kept
    assert _is_prunable_pm_bucket_leg("y1", 2, q, 0.85, bg) is True  # YES tail pruned on price
    assert _is_prunable_pm_bucket_leg("n0", 1, q, 0.85, bg) is True  # NO leg (odd)
    assert _is_prunable_pm_bucket_leg("n1", 3, q, 0.85, bg) is True  # NO leg even if priced high


def test_no_book_leg_is_prunable(tmp_path: Path) -> None:
    bg = _book_glob_for(tmp_path)
    q = _q(("y0", "n0"))
    # y0 has no book files at all → untradeable.
    assert _is_prunable_pm_bucket_leg("y0", 0, q, 0.85, bg) is True


def test_bundle_prune_is_decision_identical(tmp_path: Path) -> None:
    bg = _book_glob_for(tmp_path)
    legs = ("y0", "n0", "y1", "n1")
    q = _q(legs)
    _write_pm_book("y0", tmp_path, [(_ts(i), [0.86 + 0.001 * i], [9], [0.90], [9]) for i in range(5)])
    _write_pm_book("y1", tmp_path, [(_ts(i), [0.10], [9], [0.12], [9]) for i in range(5)])
    _write_pm_book("n0", tmp_path, [(_ts(i), [0.10], [9], [0.12], [9]) for i in range(5)])
    _write_pm_book("n1", tmp_path, [(_ts(i), [0.88], [9], [0.90], [9]) for i in range(5)])

    def build(threshold):
        return build_pm_fast_path_bundle(
            q=q,
            book_glob_for=bg,
            trades=[],
            reference_events=[],
            settlement_events=[],
            leg_prune_favorite_threshold=threshold,
        )

    full = build(None)
    pruned = build(0.85)

    # Full load: every leg with a book has events.
    assert all(len(full.leg_arrays[s].events) > 0 for s in legs)
    # Pruned: only the YES favorite (y0) survives; the rest are empty no-quote legs.
    assert len(pruned.leg_arrays["y0"].events) > 0
    for s in ("n0", "y1", "n1"):
        assert len(pruned.leg_arrays[s].events) == 0
        assert len(pruned.leg_arrays[s].book_ts) == 0
    # The KEPT leg's event array + book_ts are byte-identical to the full load.
    np.testing.assert_array_equal(pruned.leg_arrays["y0"].events, full.leg_arrays["y0"].events)
    np.testing.assert_array_equal(pruned.leg_arrays["y0"].book_ts, full.leg_arrays["y0"].book_ts)


def test_prune_disabled_for_binary(tmp_path: Path) -> None:
    """A non-bucket question must never prune, even with a threshold set."""
    bg = _book_glob_for(tmp_path)
    q = _q(("y0", "n0"), klass="priceBinary")
    # Both legs cheap (would be pruned if this were a bucket) — must still load.
    _write_pm_book("y0", tmp_path, [(_ts(0), [0.10], [9], [0.12], [9])])
    _write_pm_book("n0", tmp_path, [(_ts(0), [0.10], [9], [0.12], [9])])
    bundle = build_pm_fast_path_bundle(
        q=q,
        book_glob_for=bg,
        trades=[],
        reference_events=[],
        settlement_events=[],
        leg_prune_favorite_threshold=0.85,
    )
    assert all(len(bundle.leg_arrays[s].events) > 0 for s in ("y0", "n0"))
