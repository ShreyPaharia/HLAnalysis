"""Unit tests for tools/hl_own_fills_displayed_vs_filled.py (SHR-104).

Covers the pure analysis primitives: CSV parsing (addresses / leg-kind / fills),
taker classification (aggressor side), order clustering (re-fire-floor gap),
strictly-before book join, and the displayed-depth metrics that the whole
displayed-vs-filled ratio rests on. The duckdb I/O layer is a thin wrapper over
these and is exercised by the real-corpus run in the research note.
"""
from __future__ import annotations

from tools.hl_own_fills_displayed_vs_filled import (
    BookSnap,
    Print,
    _clip_summary_row,
    book_before_index,
    cluster_orders,
    displayed_metrics,
    is_our_taker,
    normalize_kind,
    order_metric,
    parse_fills_csv,
    summarize,
    width_bucket,
)

NS = 1_000_000_000


# ------------------------------- CSV parsing --------------------------------


SAMPLE_CSV = """\
DIAG,v1,0xAAA9bF0B573EB7Fe8506B323e20Ce070b881bBe4,2412,
DIAG,v31,0xcfA66a727a42a4F287AaB24235f9DaA6D3367fbc,664,
FILL,v31,1780690291984000000,#1591,Buy,buy,0.850900,150.0000,0.000000,0.000000,0xac0893f5
FILL,v31,1780690296164000000,#1591,Sell,sell,0.850000,363.0000,0.000000,-1.222310,0x3e21c968
FILL,v1,1780725608521000000,#1610,Settlement,sell,1.000000,305.0000,0.000000,5.343600,
KLASS,v1,#1591,priceBinary
KLASS,v31,#1670,priceBucket
"""


def test_parse_fills_csv_addresses_lowercased() -> None:
    meta = parse_fills_csv(SAMPLE_CSV)
    assert meta.addr_to_slot == {
        "0xaaa9bf0b573eb7fe8506b323e20ce070b881bbe4": "v1",
        "0xcfa66a727a42a4f287aab24235f9daa6d3367fbc": "v31",
    }
    assert meta.slot_addr()["v31"] == "0xcfa66a727a42a4f287aab24235f9daa6d3367fbc"


def test_parse_fills_csv_kind_and_fills() -> None:
    meta = parse_fills_csv(SAMPLE_CSV)
    assert meta.leg_to_kind == {"#1591": "binary", "#1670": "bucket"}
    assert len(meta.fills) == 3
    assert meta.fills[0].coin == "#1591" and meta.fills[0].sz == 150.0


def test_traded_legs_excludes_settlement() -> None:
    meta = parse_fills_csv(SAMPLE_CSV)
    # The #1610 row is a Settlement and must NOT count as a traded leg.
    assert meta.traded_legs() == {("v31", "#1591")}


def test_parse_skips_err_diag_rows() -> None:
    meta = parse_fills_csv("DIAG,v1,ERR,0,RuntimeError: boom\n")
    assert meta.addr_to_slot == {}


def test_normalize_kind() -> None:
    assert normalize_kind("priceBinary") == "binary"
    assert normalize_kind("priceBucket") == "bucket"
    assert normalize_kind("other") == "other"


# --------------------------- taker classification ---------------------------


def test_is_our_taker_buy_aggressor() -> None:
    # side='buy' => aggressor is the buyer.
    assert is_our_taker("buy", buyer="0xUS", seller="0xOTHER", addr="0xus")
    assert not is_our_taker("buy", buyer="0xOTHER", seller="0xUS", addr="0xus")


def test_is_our_taker_sell_aggressor() -> None:
    assert is_our_taker("sell", buyer="0xOTHER", seller="0xUS", addr="0xus")
    # We are on the passive (maker) side here -> NOT our taker print.
    assert not is_our_taker("sell", buyer="0xUS", seller="0xOTHER", addr="0xus")


# ----------------------------- order clustering -----------------------------


def _p(ts_s: float, side: str, px: float, sz: float) -> Print:
    return Print(int(ts_s * NS), side, px, sz)  # type: ignore[arg-type]


def test_cluster_splits_on_gap_above_floor() -> None:
    # Two sub-fills at the same instant + a third 0.73s later = TWO orders
    # (gap 0.73 > 0.5s threshold). Sub-fills (gap 0) stay merged.
    prints = [
        _p(0.0, "buy", 0.95, 100),
        _p(0.0, "buy", 0.96, 50),
        _p(0.73, "buy", 0.95, 80),
    ]
    clusters = cluster_orders(prints, max_gap_ns=NS // 2)
    assert [len(c) for c in clusters] == [2, 1]


def test_cluster_merges_within_threshold() -> None:
    prints = [_p(0.0, "buy", 0.95, 100), _p(0.3, "buy", 0.96, 50)]
    clusters = cluster_orders(prints, max_gap_ns=NS // 2)
    assert len(clusters) == 1 and len(clusters[0]) == 2


def test_cluster_splits_on_side_flip() -> None:
    prints = [_p(0.0, "buy", 0.95, 100), _p(0.0, "sell", 0.90, 50)]
    clusters = cluster_orders(prints, max_gap_ns=NS)
    assert [c[0].side for c in clusters] == ["buy", "sell"]


def test_cluster_empty() -> None:
    assert cluster_orders([], max_gap_ns=NS) == []


# ------------------------------- book join ----------------------------------


def test_book_before_index_strictly_before() -> None:
    ts = [10, 20, 30]
    assert book_before_index(ts, 25) == 1  # snapshot @20
    assert book_before_index(ts, 30) == 1  # strictly before 30 -> @20, not @30
    assert book_before_index(ts, 31) == 2
    assert book_before_index(ts, 10) == -1  # nothing strictly before the first
    assert book_before_index([], 5) == -1


# --------------------------- displayed-depth math ---------------------------


def _book() -> BookSnap:
    # asks ascending, bids descending (best-first), 3 levels each.
    return BookSnap(
        ts_ns=100,
        bid_px=(0.90, 0.89, 0.88),
        bid_sz=(40.0, 60.0, 80.0),
        ask_px=(0.95, 0.96, 0.97),
        ask_sz=(100.0, 200.0, 300.0),
    )


def test_displayed_metrics_buy_top_and_at_limit() -> None:
    dm = displayed_metrics(_book(), "buy", limit_px=0.96)
    assert dm is not None
    assert dm.disp_top == 100.0  # best ask size only
    assert dm.disp_at_limit == 300.0  # 0.95 + 0.96 levels (100 + 200)
    assert abs(dm.width - 0.05) < 1e-9
    assert dm.best_ask == 0.95 and dm.best_bid == 0.90


def test_displayed_metrics_buy_limit_at_touch_only() -> None:
    dm = displayed_metrics(_book(), "buy", limit_px=0.95)
    assert dm is not None
    assert dm.disp_top == 100.0
    assert dm.disp_at_limit == 100.0  # only the touch is marketable


def test_displayed_metrics_sell_side() -> None:
    dm = displayed_metrics(_book(), "sell", limit_px=0.89)
    assert dm is not None
    assert dm.disp_top == 40.0  # best bid size
    assert dm.disp_at_limit == 100.0  # bids at 0.90 and 0.89 (40 + 60)


def test_displayed_metrics_empty_side_returns_none() -> None:
    empty = BookSnap(1, (), (), (0.95,), (100.0,))
    assert displayed_metrics(empty, "sell", 0.9) is None  # no bids
    assert displayed_metrics(empty, "buy", 0.96) is not None


# ----------------------- end-to-end order metric ----------------------------


def test_order_metric_ratio_top_can_exceed_one_when_walking_levels() -> None:
    # One IOC eats both the 100 @0.95 and 50 of the 200 @0.96 => filled 150 vs
    # top-level 100 => ratio_top 1.5; ratio_at_limit 150/300 = 0.5.
    order = [Print(50, "buy", 0.95, 100.0), Print(50, "buy", 0.96, 50.0)]
    om = order_metric(order, _book(), slot="v31", leg="#1670", kind="bucket")
    assert om is not None
    assert om.filled == 150.0
    assert om.limit_px == 0.96
    assert abs(om.ratio_top - 1.5) < 1e-9
    assert abs(om.ratio_at_limit - 0.5) < 1e-9
    assert om.width_bucket == "mid"


def test_order_metric_partial_fill_below_top() -> None:
    # Filled 30 against a 100-share touch => hittable fraction 0.30.
    order = [Print(50, "buy", 0.95, 30.0)]
    om = order_metric(order, _book(), slot="v1", leg="#2200", kind="binary")
    assert om is not None
    assert abs(om.ratio_top - 0.30) < 1e-9
    assert abs(om.ratio_at_limit - 0.30) < 1e-9


# --------------------------------- stats ------------------------------------


def test_width_bucket_thresholds() -> None:
    assert width_bucket(0.005) == "tight"
    assert width_bucket(0.05) == "mid"
    assert width_bucket(0.20) == "wide"
    assert width_bucket(float("nan")) == "unknown"


def test_summarize_percentiles() -> None:
    d = summarize([float(i) for i in range(1, 11)])  # 1..10
    assert d.n == 10
    assert abs(d.median - 5.5) < 1e-9
    assert abs(d.p10 - 1.9) < 1e-9
    assert abs(d.p90 - 9.1) < 1e-9


def test_summarize_drops_none_and_nan() -> None:
    d = summarize([1.0, None, float("nan"), 3.0])  # type: ignore[list-item]
    assert d.n == 2 and d.median == 2.0


def test_clip_summary_row() -> None:
    # group, kind, n, p10, p25, p50, p75, p90, p99, max, mean
    row = _clip_summary_row("bucket", "bucket", [float(i) for i in range(1, 101)])
    assert row[0] == "bucket" and row[1] == "bucket" and row[2] == 100
    assert row[4] == "25.75"  # p25 over 1..100
    assert row[5] == "50.50"  # median
    assert row[-2] == "100.00"  # max
    assert row[-1] == "50.50"  # mean


def test_clip_summary_row_empty() -> None:
    row = _clip_summary_row("x", "", [])
    assert row[2] == 0  # n=0, percentiles are nan but the row is still emitted
