"""Unit tests for tools/hl_clip_cap_calibration.py (SHR-105).

Covers the pure analysis primitives: CSV parsing, regime-table enrichment,
cap derivation, and own-fills reconciliation.  All tests are pure / in-memory
— no disk I/O, no DuckDB.
"""

from __future__ import annotations

import pytest

from tools.hl_clip_cap_calibration import (
    ClipCapSpec,
    OwnClipRow,
    RegimeSummary,
    _coverage_pct,
    aggregate_by_kind,
    derive_caps,
    parse_own_clips_csv,
    parse_summary_csv,
    per_leg_regime_table,
    per_leg_rows,
    reconcile_caps_with_own_fills,
)

# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

SAMPLE_SUMMARY_CSV = """\
group,kind,n,p10,p25,p50,p75,p90,p99,max,mean
#1591,binary,1000,15.0,20.0,35.0,80.0,150.0,500.0,1000.0,60.0
#1640,binary,2000,12.0,18.0,30.0,75.0,140.0,400.0,800.0,55.0
#1610,bucket,500,10.0,15.0,25.0,55.0,120.0,400.0,600.0,40.0
#1670,bucket,100,8.0,14.0,20.0,45.0,100.0,300.0,350.0,35.0
ALL_binary,binary,3000,13.0,19.0,32.0,77.0,145.0,450.0,1000.0,57.0
ALL_bucket,bucket,600,9.0,14.0,23.0,50.0,110.0,350.0,600.0,38.0
ALL,,3600,12.0,18.0,30.0,72.0,140.0,440.0,1000.0,53.0
"""

SAMPLE_OWN_FILLS_CSV = """\
slot,leg,kind,side,ts_ns,n_prints,filled,limit_px,decision_ts_ns,best_bid,best_ask,width,width_bucket,disp_top,disp_at_limit,ratio_top,ratio_at_limit
v31,#1591,binary,buy,1000000000,2,150.0,0.85,999000000,0.83,0.85,0.02,tight,200.0,500.0,0.75,0.3
v31,#1591,binary,sell,2000000000,1,220.0,0.84,1999000000,0.84,0.86,0.02,tight,150.0,300.0,1.47,0.73
v31,#1640,binary,buy,3000000000,3,180.0,0.90,2999000000,0.88,0.90,0.02,tight,180.0,600.0,1.0,0.3
v31,#1610,bucket,sell,4000000000,1,30.0,0.85,3999000000,0.81,0.95,0.14,wide,100.0,100.0,0.30,0.30
v31,#1670,bucket,buy,5000000000,2,60.0,0.70,4999000000,0.60,0.80,0.20,wide,50.0,200.0,1.20,0.30
"""


# ---------------------------------------------------------------------------
# parse_summary_csv
# ---------------------------------------------------------------------------


def test_parse_summary_csv_basic() -> None:
    rows = parse_summary_csv(SAMPLE_SUMMARY_CSV)
    assert len(rows) == 7
    r = rows[0]
    assert r.group == "#1591"
    assert r.kind == "binary"
    assert r.n == 1000
    assert r.p50 == 35.0
    assert r.p90 == 150.0
    assert r.max == 1000.0


def test_parse_summary_csv_all_row() -> None:
    rows = parse_summary_csv(SAMPLE_SUMMARY_CSV)
    all_row = next(r for r in rows if r.group == "ALL")
    assert all_row.kind == ""
    assert all_row.n == 3600


def test_parse_summary_csv_aggregate_rows() -> None:
    rows = parse_summary_csv(SAMPLE_SUMMARY_CSV)
    agg = aggregate_by_kind(rows)
    assert "binary" in agg
    assert "bucket" in agg
    assert "ALL" in agg
    assert agg["binary"].n == 3000
    assert agg["bucket"].n == 600


# ---------------------------------------------------------------------------
# aggregate_by_kind
# ---------------------------------------------------------------------------


def test_aggregate_by_kind_returns_only_agg_rows() -> None:
    rows = parse_summary_csv(SAMPLE_SUMMARY_CSV)
    agg = aggregate_by_kind(rows)
    # Should not contain per-leg rows
    assert all(k in ("binary", "bucket", "ALL") for k in agg)


def test_aggregate_by_kind_correct_stats() -> None:
    rows = parse_summary_csv(SAMPLE_SUMMARY_CSV)
    agg = aggregate_by_kind(rows)
    assert agg["binary"].p90 == 145.0
    assert agg["bucket"].p90 == 110.0


# ---------------------------------------------------------------------------
# per_leg_rows
# ---------------------------------------------------------------------------


def test_per_leg_rows_excludes_aggregate() -> None:
    rows = parse_summary_csv(SAMPLE_SUMMARY_CSV)
    legs = per_leg_rows(rows)
    assert all(r.group.startswith("#") for r in legs)
    assert len(legs) == 4  # #1591, #1640, #1610, #1670


def test_per_leg_rows_includes_both_kinds() -> None:
    rows = parse_summary_csv(SAMPLE_SUMMARY_CSV)
    legs = per_leg_rows(rows)
    kinds = {r.kind for r in legs}
    assert kinds == {"binary", "bucket"}


# ---------------------------------------------------------------------------
# per_leg_regime_table
# ---------------------------------------------------------------------------


def test_per_leg_regime_table_known_legs() -> None:
    rows = parse_summary_csv(SAMPLE_SUMMARY_CSV)
    legs = per_leg_rows(rows)
    table = per_leg_regime_table(legs)
    reg = {r.group: r.width_regime for r in table}
    assert reg["#1591"] == "tight"
    assert reg["#1640"] == "tight"
    assert reg["#1610"] == "wide"
    assert reg["#1670"] == "wide"


def test_per_leg_regime_table_unknown_leg() -> None:
    """An unknown leg (not in _LEG_WIDTH_REGIME) gets 'unknown', not an error."""
    unknown_csv = (
        "group,kind,n,p10,p25,p50,p75,p90,p99,max,mean\n#9999,binary,100,10.0,15.0,25.0,50.0,80.0,200.0,500.0,30.0\n"
    )
    rows = parse_summary_csv(unknown_csv)
    legs = per_leg_rows(rows)
    table = per_leg_regime_table(legs)
    assert table[0].width_regime == "unknown"


# ---------------------------------------------------------------------------
# derive_caps
# ---------------------------------------------------------------------------


def test_derive_caps_p90() -> None:
    rows = parse_summary_csv(SAMPLE_SUMMARY_CSV)
    agg = aggregate_by_kind(rows)
    legs = per_leg_rows(rows)
    specs = derive_caps(agg, legs, cap_percentile="p90")
    cap_by_kind = {s.kind: s.recommended_cap for s in specs}
    assert cap_by_kind["binary"] == 145.0  # ALL_binary p90
    assert cap_by_kind["bucket"] == 110.0  # ALL_bucket p90


def test_derive_caps_p99() -> None:
    rows = parse_summary_csv(SAMPLE_SUMMARY_CSV)
    agg = aggregate_by_kind(rows)
    legs = per_leg_rows(rows)
    specs = derive_caps(agg, legs, cap_percentile="p99")
    cap_by_kind = {s.kind: s.recommended_cap for s in specs}
    assert cap_by_kind["binary"] == 450.0
    assert cap_by_kind["bucket"] == 350.0


def test_derive_caps_returns_two_kinds() -> None:
    rows = parse_summary_csv(SAMPLE_SUMMARY_CSV)
    agg = aggregate_by_kind(rows)
    legs = per_leg_rows(rows)
    specs = derive_caps(agg, legs)
    assert {s.kind for s in specs} == {"binary", "bucket"}


def test_derive_caps_rationale_not_empty() -> None:
    rows = parse_summary_csv(SAMPLE_SUMMARY_CSV)
    agg = aggregate_by_kind(rows)
    legs = per_leg_rows(rows)
    for spec in derive_caps(agg, legs):
        assert spec.rationale, f"Empty rationale for {spec.kind}"


# ---------------------------------------------------------------------------
# parse_own_clips_csv
# ---------------------------------------------------------------------------


def test_parse_own_clips_csv_basic() -> None:
    rows = parse_own_clips_csv(SAMPLE_OWN_FILLS_CSV)
    assert len(rows) == 5
    assert rows[0].kind == "binary"
    assert rows[0].filled == 150.0
    assert rows[0].leg == "#1591"


def test_parse_own_clips_csv_both_kinds() -> None:
    rows = parse_own_clips_csv(SAMPLE_OWN_FILLS_CSV)
    kinds = {r.kind for r in rows}
    assert kinds == {"binary", "bucket"}


def test_parse_own_clips_csv_legacy_column_names() -> None:
    """Should also accept 'coin' and 'filled_qty' column names (older format)."""
    legacy_csv = (
        "slot,coin,kind,side,ts_ns,n_prints,filled_qty,ratio_top,width_bucket\n"
        "v31,#1591,binary,buy,1000000000,1,100.0,1.0,tight\n"
    )
    rows = parse_own_clips_csv(legacy_csv)
    assert len(rows) == 1
    assert rows[0].leg == "#1591"
    assert rows[0].filled == 100.0


def test_parse_own_clips_csv_ratio_top() -> None:
    rows = parse_own_clips_csv(SAMPLE_OWN_FILLS_CSV)
    ratios = [r.ratio_top for r in rows]
    # First row: ratio_top = 0.75
    assert abs(ratios[0] - 0.75) < 1e-6


# ---------------------------------------------------------------------------
# reconcile_caps_with_own_fills
# ---------------------------------------------------------------------------


def _make_spec(kind: str, cap: float) -> ClipCapSpec:
    return ClipCapSpec(
        kind=kind,
        width_regime="tight" if kind == "binary" else "wide",
        n=1000,
        p50=30.0,
        p90=cap,
        p99=300.0,
        max=500.0,
        recommended_cap=cap,
        rationale="test",
    )


def test_reconcile_cap_below_doom_loop_is_ok() -> None:
    """Cap well below the sim's 516-sh doom-loop dump is acceptable."""
    specs = [_make_spec("binary", 166.0)]
    own_rows = [
        OwnClipRow("v31", "#1591", "binary", 100.0, 1.0, "tight"),
        OwnClipRow("v31", "#1591", "binary", 150.0, 1.2, "tight"),
        OwnClipRow("v31", "#1591", "binary", 200.0, 1.5, "tight"),
    ]
    notes = reconcile_caps_with_own_fills(specs, own_rows)
    assert len(notes) == 1
    assert "OK" in notes[0]
    assert "doom" in notes[0].lower()


def test_reconcile_cap_at_or_above_doom_loop_is_warning() -> None:
    """Cap >= 516 sh does NOT reshape the sim's single dump and must warn."""
    specs = [_make_spec("binary", 600.0)]  # above the 516-sh sim dump
    own_rows = [
        OwnClipRow("v31", "#1591", "binary", 100.0, 1.0, "tight"),
    ]
    notes = reconcile_caps_with_own_fills(specs, own_rows)
    assert len(notes) == 1
    assert "WARNING" in notes[0]


def test_reconcile_no_own_rows_for_kind() -> None:
    specs = [_make_spec("binary", 166.0)]
    own_rows: list[OwnClipRow] = []
    notes = reconcile_caps_with_own_fills(specs, own_rows)
    assert len(notes) == 1
    assert "no own-fill rows" in notes[0]


def test_reconcile_bucket_ok() -> None:
    specs = [_make_spec("bucket", 142.0)]
    own_rows = [
        OwnClipRow("v31", "#1610", "bucket", 30.0, 0.5, "wide"),
        OwnClipRow("v31", "#1670", "bucket", 45.0, 0.9, "wide"),
        OwnClipRow("v31", "#1670", "bucket", 60.0, 1.0, "wide"),
    ]
    notes = reconcile_caps_with_own_fills(specs, own_rows)
    assert "OK" in notes[0]


def test_reconcile_note_includes_per_clip_sizes() -> None:
    """Reconciliation note should mention the SHR-104 per-clip sizes."""
    specs = [_make_spec("binary", 166.0)]
    own_rows = [OwnClipRow("v31", "#1591", "binary", 200.0, 1.0, "tight")]
    notes = reconcile_caps_with_own_fills(specs, own_rows)
    # Should reference per-clip distinction
    assert "per-clip" in notes[0].lower() or "per_clip" in notes[0].lower() or "per-ORDER" in notes[0]


# ---------------------------------------------------------------------------
# _coverage_pct
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pct_str,expected",
    [
        ("p10", 10.0),
        ("p25", 25.0),
        ("p50", 50.0),
        ("p75", 75.0),
        ("p90", 90.0),
        ("p99", 99.0),
    ],
)
def test_coverage_pct_known_values(pct_str: str, expected: float) -> None:
    spec = _make_spec("binary", 100.0)
    assert _coverage_pct(spec, pct_str) == expected


def test_coverage_pct_unknown_defaults_90() -> None:
    spec = _make_spec("binary", 100.0)
    assert _coverage_pct(spec, "p77") == 90.0


# ---------------------------------------------------------------------------
# Integration: build_report produces meaningful output
# ---------------------------------------------------------------------------


def test_build_report_contains_key_sections() -> None:
    from tools.hl_clip_cap_calibration import build_report

    rows = parse_summary_csv(SAMPLE_SUMMARY_CSV)
    report = build_report(rows, None)
    assert "Per-leg clip-size distribution" in report
    assert "Kind-aggregate summary" in report
    assert "Derived clip-cap model" in report
    assert "Reconciliation" in report


def test_build_report_with_own_fills() -> None:
    from tools.hl_clip_cap_calibration import build_report

    rows = parse_summary_csv(SAMPLE_SUMMARY_CSV)
    report = build_report(rows, SAMPLE_OWN_FILLS_CSV)
    assert "binary" in report
    assert "bucket" in report
    # reconcile section should not be empty
    assert "cap" in report.lower()


def test_build_report_p99_percentile() -> None:
    from tools.hl_clip_cap_calibration import build_report

    rows = parse_summary_csv(SAMPLE_SUMMARY_CSV)
    report = build_report(rows, cap_percentile="p99")
    # p99 cap for binary is 450
    assert "450" in report
