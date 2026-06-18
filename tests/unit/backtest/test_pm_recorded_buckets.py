"""Recorder-native PM bucket discovery — build priceBucket questions from the
recorded ``question_meta`` so their leg tokens match the recorded books.

Regression context: the Gamma cache manifest's bucket questions reference token
IDs that don't overlap the recorded book_snapshot tokens (15 of 131 for
btc-multi-strikes-weekly), so manifest buckets backtest to 0 decisions. These
tests pin the builder that reconstructs buckets from the recorder's per-leg
``question_meta``, including the load-bearing pre-expiry window guard (the
recorder also captures post-settlement snapshots, which must not invert the
question window).
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from hlanalysis.backtest.data._pm_recorded_buckets import (
    _parse_threshold,
    build_recorded_bucket_entries,
)

_QM_SUB = "venue=polymarket/product_type=prediction_binary/mechanism=clob/event=question_meta"
_BK_SUB = "venue=polymarket/product_type=prediction_binary/mechanism=clob/event=book_snapshot"
EXPIRY_NS = 1780761600000000000  # 2026-06-06 16:00 UTC


def _qm_glob(root: Path) -> str:
    return str(root / _QM_SUB / "**" / "*.parquet")


def _book_glob_for(root: Path):
    return lambda tok: str(root / _BK_SUB / f"symbol={tok}" / "**" / "*.parquet")


def _write_qm(root: Path, legs: list[dict], *, slug: str, expiry: str, expiry_ns: int) -> None:
    """Write one question_meta parquet row per leg (recorder schema: keys/values lists)."""
    keys_col, vals_col, sym_col = [], [], []
    for leg in legs:
        kv = {
            "series_slug": slug,
            "expiry": expiry,
            "expiry_ns": str(expiry_ns),
            "yes_token_id": leg["yes"],
            "no_token_id": leg["no"],
            "condition_id": leg["cond"],
            "question_name": leg["qn"],
        }
        keys_col.append(list(kv.keys()))
        vals_col.append([str(v) for v in kv.values()])
        sym_col.append(leg["yes"])
    d = root / _QM_SUB / "date=2026-06-06"
    d.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "symbol": pa.array(sym_col, pa.string()),
                "keys": pa.array(keys_col, pa.list_(pa.string())),
                "values": pa.array(vals_col, pa.list_(pa.string())),
            }
        ),
        d / "qm.parquet",
    )


def _write_book(root: Path, tok: str, ts_list: list[int]) -> None:
    d = root / _BK_SUB / f"symbol={tok}" / "date=2026-06-06"
    d.mkdir(parents=True, exist_ok=True)
    lt = pa.list_(pa.float64())
    pq.write_table(
        pa.table(
            {
                "exchange_ts": pa.array(ts_list, pa.int64()),
                "bid_px": pa.array([[0.5]] * len(ts_list), lt),
                "bid_sz": pa.array([[9.0]] * len(ts_list), lt),
                "ask_px": pa.array([[0.55]] * len(ts_list), lt),
                "ask_sz": pa.array([[9.0]] * len(ts_list), lt),
            }
        ),
        d / "b.parquet",
    )


def test_parse_threshold() -> None:
    assert _parse_threshold("Will the price of Bitcoin be above $56,000 on June 6?") == 56000.0
    assert _parse_threshold("...above $1.60 on June 17?") == 1.60
    assert _parse_threshold("above 84000") == 84000.0
    assert _parse_threshold("no strike here") is None
    assert _parse_threshold(None) is None


def _legs(n: int) -> list[dict]:
    out = []
    for i in range(n):
        strike = 56000 + i * 2000
        out.append(
            {
                "yes": f"y{i}",
                "no": f"n{i}",
                "cond": f"c{i}",
                "qn": f"Will the price of Bitcoin be above ${strike:,} on June 6?",
            }
        )
    return out


def test_build_groups_sorts_and_oracle_resolves(tmp_path: Path) -> None:
    legs = _legs(3)  # strikes 56000, 58000, 60000
    _write_qm(tmp_path, legs, slug="btc-multi-strikes-weekly", expiry="20260606-1600", expiry_ns=EXPIRY_NS)
    # all YES legs get a pre-expiry book snapshot
    for leg in legs:
        _write_book(tmp_path, leg["yes"], [EXPIRY_NS - 3600 * 10**9, EXPIRY_NS - 60 * 10**9])

    entries = build_recorded_bucket_entries(
        question_meta_glob=_qm_glob(tmp_path),
        series_slug="btc-multi-strikes-weekly",
        book_glob_for=_book_glob_for(tmp_path),
        oracle_close_at=lambda _ns: 59000.0,  # BTC closes at $59k
    )
    assert list(entries) == ["btc-multi-strikes-weekly:20260606-1600"]
    b = entries["btc-multi-strikes-weekly:20260606-1600"]["bucket"]
    assert b["thresholds"] == [56000.0, 58000.0, 60000.0]
    # even index = YES token, ordered by ascending strike
    assert b["leg_tokens"] == [["y0", "n0"], ["y1", "n1"], ["y2", "n2"]]
    # oracle: 59k > 56k yes, > 58k yes, > 60k no
    assert b["leg_resolutions"] == ["yes", "yes", "no"]
    # window is valid (start < end) and start is the min PRE-expiry book ts
    assert b["start_ts_ns"] == EXPIRY_NS - 3600 * 10**9
    assert b["start_ts_ns"] < b["end_ts_ns"] == EXPIRY_NS


def test_skips_bucket_with_only_post_expiry_books(tmp_path: Path) -> None:
    """The load-bearing guard: a bucket whose only book snapshots are AFTER expiry
    is untradeable and must be skipped (else start_ts_ns > end_ts_ns inverts the
    window and the scan loop breaks before its first tick → 0 decisions)."""
    legs = _legs(3)
    _write_qm(tmp_path, legs, slug="btc-multi-strikes-weekly", expiry="20260606-1600", expiry_ns=EXPIRY_NS)
    for leg in legs:
        _write_book(tmp_path, leg["yes"], [EXPIRY_NS + 60 * 10**9, EXPIRY_NS + 3600 * 10**9])  # post-expiry only

    entries = build_recorded_bucket_entries(
        question_meta_glob=_qm_glob(tmp_path),
        series_slug="btc-multi-strikes-weekly",
        book_glob_for=_book_glob_for(tmp_path),
        oracle_close_at=lambda _ns: 59000.0,
    )
    assert entries == {}


def test_skips_when_oracle_unavailable(tmp_path: Path) -> None:
    legs = _legs(2)
    _write_qm(tmp_path, legs, slug="btc-multi-strikes-weekly", expiry="20260606-1600", expiry_ns=EXPIRY_NS)
    for leg in legs:
        _write_book(tmp_path, leg["yes"], [EXPIRY_NS - 60 * 10**9])
    entries = build_recorded_bucket_entries(
        question_meta_glob=_qm_glob(tmp_path),
        series_slug="btc-multi-strikes-weekly",
        book_glob_for=_book_glob_for(tmp_path),
        oracle_close_at=lambda _ns: None,  # no reference price → cannot resolve
    )
    assert entries == {}


def test_other_series_ignored(tmp_path: Path) -> None:
    legs = _legs(3)
    _write_qm(tmp_path, legs, slug="ethereum-multi-strikes-weekly", expiry="20260606-1600", expiry_ns=EXPIRY_NS)
    for leg in legs:
        _write_book(tmp_path, leg["yes"], [EXPIRY_NS - 60 * 10**9])
    entries = build_recorded_bucket_entries(
        question_meta_glob=_qm_glob(tmp_path),
        series_slug="btc-multi-strikes-weekly",  # different series
        book_glob_for=_book_glob_for(tmp_path),
        oracle_close_at=lambda _ns: 59000.0,
    )
    assert entries == {}
