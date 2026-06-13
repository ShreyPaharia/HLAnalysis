"""Unit tests for tools/recorder_completeness.py (SHR-84).

The tool is a read-only completeness / seq-gap reconciliation checker over the
recorder's Hive-partitioned parquet. Tests cover the pure analysis primitives
plus an end-to-end pass over synthetic parquet with injected gaps.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from tools.recorder_completeness import (
    QuestionSpec,
    analyze_table,
    build_completeness_report,
    format_summary,
    longest_time_gap,
    question_leg_coverage,
    seq_gap_stats,
)

NS = 1_000_000_000


# ----------------------------- seq_gap_stats --------------------------------


def test_seq_gap_stats_contiguous_is_complete() -> None:
    s = seq_gap_stats([1, 2, 3, 4, 5])
    assert s.n_with_seq == 5
    assert s.n_gaps == 0
    assert s.n_missing == 0
    assert s.largest_gap == 0
    assert s.complete is True


def test_seq_gap_stats_detects_single_gap_and_size() -> None:
    # 4 and 5 missing between 3 and 6 -> one gap of 2.
    s = seq_gap_stats([1, 2, 3, 6, 7])
    assert s.n_gaps == 1
    assert s.n_missing == 2
    assert s.largest_gap == 2
    assert s.complete is False


def test_seq_gap_stats_tracks_largest_of_multiple_gaps() -> None:
    # gaps: 5..9 missing (4), then 11 missing (1).
    s = seq_gap_stats([4, 10, 12])
    assert s.n_gaps == 2
    # 10-4-1 = 5 missing; 12-10-1 = 1 missing => total 6
    assert s.n_missing == 6
    assert s.largest_gap == 5


def test_seq_gap_stats_counts_duplicates_separately() -> None:
    s = seq_gap_stats([1, 2, 2, 3])
    assert s.n_dup == 1
    assert s.n_missing == 0
    assert s.complete is True  # duplicates are not a coverage hole


def test_seq_gap_stats_ignores_none_and_counts_them() -> None:
    s = seq_gap_stats([1, None, 2, None, 3])
    assert s.n_total == 5
    assert s.n_with_seq == 3
    assert s.n_no_seq == 2
    assert s.complete is True


def test_seq_gap_stats_empty() -> None:
    s = seq_gap_stats([])
    assert s.n_total == 0
    assert s.complete is True


def test_seq_gap_stats_order_independent() -> None:
    # Out-of-order arrival should not be mistaken for a gap.
    a = seq_gap_stats([3, 1, 2, 5, 4])
    assert a.n_missing == 0
    assert a.complete is True


# ----------------------------- longest_time_gap -----------------------------


def test_longest_time_gap_picks_max_interval() -> None:
    ts = [0, 1 * NS, 2 * NS, 12 * NS]  # last gap is 10s
    g = longest_time_gap(ts)
    assert g.longest_gap_s == 10.0
    assert g.gap_start_ns == 2 * NS
    assert g.gap_end_ns == 12 * NS


def test_longest_time_gap_unsorted_input() -> None:
    ts = [12 * NS, 0, 2 * NS, 1 * NS]
    g = longest_time_gap(ts)
    assert g.longest_gap_s == 10.0


def test_longest_time_gap_single_event_is_zero() -> None:
    assert longest_time_gap([5 * NS]).longest_gap_s == 0.0
    assert longest_time_gap([]).longest_gap_s == 0.0


# --------------------------- question_leg_coverage --------------------------


def test_question_leg_coverage_all_legs_covered() -> None:
    q = QuestionSpec(
        question_id="Q1",
        start_ts_ns=100 * NS,
        end_ts_ns=200 * NS,
        leg_symbols=("@30", "@31"),
    )
    spans = {"@30": (90 * NS, 210 * NS), "@31": (100 * NS, 200 * NS)}
    reps = question_leg_coverage([q], spans)
    assert len(reps) == 2
    assert all(r.covered for r in reps)


def test_question_leg_coverage_missing_leg() -> None:
    q = QuestionSpec(
        question_id="Q1",
        start_ts_ns=100 * NS,
        end_ts_ns=200 * NS,
        leg_symbols=("@30", "@31"),
    )
    spans = {"@30": (90 * NS, 210 * NS)}  # @31 absent entirely
    reps = question_leg_coverage([q], spans)
    by_leg = {r.leg: r for r in reps}
    assert by_leg["@30"].covered is True
    assert by_leg["@31"].covered is False
    assert by_leg["@31"].present is False


def test_question_leg_coverage_partial_window() -> None:
    q = QuestionSpec(
        question_id="Q1",
        start_ts_ns=100 * NS,
        end_ts_ns=200 * NS,
        leg_symbols=("@30",),
    )
    # Data starts late (110s) and ends early (180s): not covering [100,200].
    spans = {"@30": (110 * NS, 180 * NS)}
    reps = question_leg_coverage([q], spans, edge_tolerance_s=1.0)
    r = reps[0]
    assert r.present is True
    assert r.covered is False
    assert r.gap_after_start_s == 10.0
    assert r.gap_before_end_s == 20.0


# ------------------------------ analyze_table -------------------------------


def _trade_table(rows: list[dict]) -> pa.Table:
    return pa.Table.from_pylist(rows)


def test_analyze_table_clean_trade_stream() -> None:
    rows = [
        {
            "symbol": "BTC",
            "exchange_ts": i * NS,
            "local_recv_ts": i * NS,
            "seq": i,
            "event_type": "trade",
            "price": 100.0,
            "size": 1.0,
        }
        for i in range(1, 6)
    ]
    seq_reps, time_reps, vol_reps = analyze_table(_trade_table(rows), event="trade", min_events=1, min_notional=0.0)
    assert len(seq_reps) == 1
    assert seq_reps[0].symbol == "BTC"
    assert seq_reps[0].complete is True
    # notional = sum(price*size) = 5 * 100
    assert vol_reps[0].notional == 500.0
    assert all(not v.quiet for v in vol_reps)


def test_analyze_table_flags_seq_gap_and_quiet_window() -> None:
    rows = [
        {
            "symbol": "BTC",
            "exchange_ts": NS,
            "local_recv_ts": NS,
            "seq": 1,
            "event_type": "trade",
            "price": 100.0,
            "size": 1.0,
        },
        {
            "symbol": "BTC",
            "exchange_ts": 2 * NS,
            "local_recv_ts": 2 * NS,
            "seq": 5,
            "event_type": "trade",
            "price": 100.0,
            "size": 1.0,
        },
    ]
    seq_reps, _time_reps, vol_reps = analyze_table(_trade_table(rows), event="trade", min_events=10, min_notional=0.0)
    assert seq_reps[0].complete is False
    assert seq_reps[0].n_missing == 3
    # Only 2 events in the hour, below min_events=10 -> quiet.
    assert vol_reps[0].quiet is True


def test_analyze_table_groups_by_symbol() -> None:
    rows = [
        {
            "symbol": "BTC",
            "exchange_ts": NS,
            "local_recv_ts": NS,
            "seq": 1,
            "event_type": "trade",
            "price": 100.0,
            "size": 1.0,
        },
        {
            "symbol": "ETH",
            "exchange_ts": NS,
            "local_recv_ts": NS,
            "seq": 1,
            "event_type": "trade",
            "price": 50.0,
            "size": 2.0,
        },
    ]
    seq_reps, _, vol_reps = analyze_table(_trade_table(rows), event="trade", min_events=1, min_notional=0.0)
    assert {r.symbol for r in seq_reps} == {"BTC", "ETH"}


# --------------------------- end-to-end over parquet ------------------------


def _write_partition(
    root: Path,
    event: str,
    symbol: str,
    rows: list[dict],
    *,
    date: str = "2026-06-08",
    hour: str = "00",
) -> None:
    part = (
        root
        / "venue=hyperliquid"
        / "product_type=prediction_binary"
        / "mechanism=clob"
        / f"event={event}"
        / f"symbol={symbol}"
        / f"date={date}"
        / f"hour={hour}"
    )
    part.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), part / "data.parquet")


def test_build_completeness_report_clean_corpus(tmp_path: Path) -> None:
    rows = [
        {
            "venue": "hyperliquid",
            "symbol": "@30",
            "exchange_ts": i * NS,
            "local_recv_ts": i * NS,
            "seq": i,
            "event_type": "trade",
            "price": 100.0,
            "size": 1.0,
        }
        for i in range(1, 21)
    ]
    _write_partition(tmp_path, "trade", "@30", rows)
    report = build_completeness_report(tmp_path, events=("trade",), min_events=5, min_notional=0.0)
    assert report["summary"]["complete"] is True
    assert report["summary"]["n_seq_incomplete"] == 0
    assert report["summary"]["n_quiet_windows"] == 0
    text = format_summary(report)
    assert "complete" in text.lower()


def test_build_completeness_report_detects_injected_gaps(tmp_path: Path) -> None:
    # Inject a seq gap (skip 5..9) and a 30s time gap.
    seqs = [1, 2, 3, 4, 10, 11]
    times = [1, 2, 3, 4, 35, 36]  # 31s gap between event 4 and 5
    rows = [
        {
            "venue": "hyperliquid",
            "symbol": "@30",
            "exchange_ts": t * NS,
            "local_recv_ts": t * NS,
            "seq": s,
            "event_type": "trade",
            "price": 100.0,
            "size": 1.0,
        }
        for s, t in zip(seqs, times)
    ]
    _write_partition(tmp_path, "trade", "@30", rows)
    report = build_completeness_report(tmp_path, events=("trade",), min_events=1, min_notional=0.0)
    assert report["summary"]["complete"] is False
    assert report["summary"]["n_seq_incomplete"] == 1
    # Largest time gap surfaced.
    max_gap = max(g["longest_gap_s"] for g in report["time_gaps"])
    assert max_gap >= 31.0


def test_build_completeness_report_empty_root(tmp_path: Path) -> None:
    report = build_completeness_report(tmp_path, events=("trade",))
    assert report["summary"]["complete"] is True
    assert report["summary"]["n_symbols"] == 0
