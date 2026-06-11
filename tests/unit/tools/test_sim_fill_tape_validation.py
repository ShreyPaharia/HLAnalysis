"""SHR-106 — independent sim-fill validation against the recorded HL trade tape.

Unit-tests the pure matching / aggregation logic: given a sim IOC fill and the
recorded trade prints for its symbol, decide how much *actually traded*
at-or-through the limit in the fill's window, and flag the fill as
phantom-liquidity when its size exceeds that contemporaneous volume.

All tests are filesystem-free — they build small in-memory prints frames and
``FillRecord``s, so they exercise the logic, not the recorder layout.
"""
from __future__ import annotations

import pandas as pd

from tools.sim_fill_tape_validation import (
    CellAgg,
    FillRecord,
    aggregate,
    classify_fill,
    hittable_prints,
    ledger_verdicts,
    load_sim_fills,
    run_matrix,
    validate_cell,
)

S = 1_000_000_000  # one second in ns


def _prints(rows) -> pd.DataFrame:
    """rows: list of (ts_ns, price, size, side)."""
    return pd.DataFrame(rows, columns=["ts_ns", "price", "size", "side"])


# --------------------------------------------------------------------------- #
# hittable_prints — price + side + window filtering
# --------------------------------------------------------------------------- #
def test_hittable_buy_keeps_prints_at_or_below_limit():
    # A marketable BUY at 0.90 lifts asks; real buy-aggressor prints at <=0.90
    # are the competing liquidity. 0.95 is above the limit -> excluded.
    pr = _prints([
        (100 * S, 0.88, 10.0, "buy"),
        (100 * S, 0.90, 20.0, "buy"),
        (100 * S, 0.95, 30.0, "buy"),
    ])
    hp = hittable_prints(
        pr, side="buy", price=0.90, fill_ts_ns=100 * S, window_ns=S,
    )
    assert sorted(hp["size"].tolist()) == [10.0, 20.0]


def test_hittable_sell_keeps_prints_at_or_above_limit():
    pr = _prints([
        (100 * S, 0.10, 10.0, "sell"),
        (100 * S, 0.20, 20.0, "sell"),
        (100 * S, 0.25, 30.0, "sell"),
    ])
    hp = hittable_prints(
        pr, side="sell", price=0.20, fill_ts_ns=100 * S, window_ns=S,
    )
    assert sorted(hp["size"].tolist()) == [20.0, 30.0]


def test_hittable_filters_outside_time_window():
    pr = _prints([
        (98 * S, 0.90, 11.0, "buy"),   # 2s before -> out (window 1s)
        (100 * S, 0.90, 22.0, "buy"),  # in
        (102 * S, 0.90, 33.0, "buy"),  # 2s after -> out
    ])
    hp = hittable_prints(
        pr, side="buy", price=0.90, fill_ts_ns=100 * S, window_ns=S,
    )
    assert hp["size"].tolist() == [22.0]


def test_hittable_match_aggressor_excludes_opposite_side():
    # A sim BUY: only real buy-aggressor prints count when match_aggressor.
    pr = _prints([
        (100 * S, 0.90, 10.0, "buy"),
        (100 * S, 0.90, 99.0, "sell"),
    ])
    matched = hittable_prints(
        pr, side="buy", price=0.90, fill_ts_ns=100 * S, window_ns=S,
        match_aggressor=True,
    )
    assert matched["size"].tolist() == [10.0]
    both = hittable_prints(
        pr, side="buy", price=0.90, fill_ts_ns=100 * S, window_ns=S,
        match_aggressor=False,
    )
    assert sorted(both["size"].tolist()) == [10.0, 99.0]


def test_hittable_price_tol_boundary():
    # A print a hair above the limit is kept only within price_tol.
    pr = _prints([(100 * S, 0.9000005, 5.0, "buy")])
    keep = hittable_prints(
        pr, side="buy", price=0.90, fill_ts_ns=100 * S, window_ns=S,
        price_tol=1e-6,
    )
    assert keep["size"].tolist() == [5.0]
    drop = hittable_prints(
        pr, side="buy", price=0.90, fill_ts_ns=100 * S, window_ns=S,
        price_tol=1e-9,
    )
    assert drop.empty


# --------------------------------------------------------------------------- #
# classify_fill — phantom verdict
# --------------------------------------------------------------------------- #
def _fill(size, side="buy", price=0.90, ts=100 * S, symbol="#1670"):
    return FillRecord(
        cloid="c1", ts_ns=ts, side=side, price=price, size=size,
        symbol=symbol, question_id="Q31",
    )


def test_classify_phantom_when_size_exceeds_hittable():
    pr = _prints([(100 * S, 0.90, 30.0, "buy")])
    v = classify_fill(_fill(100.0), pr, window_ns=S)
    assert v.is_phantom is True
    assert v.hittable_size == 30.0
    assert v.phantom_excess == 70.0
    assert v.phantom_notional == 70.0 * 0.90
    assert v.tape_covered is True


def test_classify_not_phantom_when_market_traded_enough():
    pr = _prints([
        (100 * S, 0.90, 60.0, "buy"),
        (100 * S, 0.89, 60.0, "buy"),
    ])
    v = classify_fill(_fill(100.0), pr, window_ns=S)
    assert v.is_phantom is False
    assert v.hittable_size == 120.0
    assert v.phantom_excess == 0.0


def test_classify_no_print_in_window_is_phantom():
    # Nearest print is far away (open-burst fill): zero contemporaneous flow.
    pr = _prints([(100 * S, 0.90, 500.0, "buy")])
    v = classify_fill(_fill(57.0, ts=1000 * S), pr, window_ns=S)
    assert v.is_phantom is True
    assert v.hittable_size == 0.0
    assert v.phantom_excess == 57.0
    assert v.tape_covered is True  # symbol HAD prints, just none in window


def test_classify_no_tape_marks_uncovered():
    empty = _prints([])
    v = classify_fill(_fill(57.0), empty, window_ns=S)
    assert v.tape_covered is False
    assert v.is_phantom is True
    assert v.hittable_size == 0.0


# --------------------------------------------------------------------------- #
# ledger_verdicts — non-double-counted volume crediting across re-fires
# --------------------------------------------------------------------------- #
def test_ledger_does_not_double_count_overlapping_refire():
    # The #2230 shape: two BUY 100@0.90 fired 0.5s apart, but only 100 really
    # traded at 0.90 in the window. Independent matching would excuse both
    # (each sees 100); the ledger credits the first and flags the second.
    pr = _prints([
        (100 * S, 0.90, 60.0, "buy"),
        (100 * S + 10, 0.90, 40.0, "buy"),
    ])
    fills = [_fill(100.0, ts=100 * S), _fill(100.0, ts=100 * S + S // 2)]
    v1, v2 = ledger_verdicts(fills, pr, window_ns=S)
    assert v1.tape_filled == 100.0 and v1.phantom_excess == 0.0 and not v1.is_phantom
    assert v2.tape_filled == 0.0 and v2.phantom_excess == 100.0 and v2.is_phantom


def test_ledger_partial_credit_then_starve():
    # 80 traded; first fill takes 80 (phantom 20), second is fully starved.
    pr = _prints([(100 * S, 0.90, 80.0, "buy")])
    fills = [_fill(100.0, ts=100 * S), _fill(50.0, ts=100 * S + S // 2)]
    v1, v2 = ledger_verdicts(fills, pr, window_ns=S)
    assert v1.tape_filled == 80.0 and v1.phantom_excess == 20.0
    assert v2.tape_filled == 0.0 and v2.phantom_excess == 50.0


def test_ledger_processes_earliest_first_regardless_of_input_order():
    pr = _prints([(100 * S, 0.90, 100.0, "buy")])
    late = _fill(100.0, ts=100 * S + S // 2)
    early = _fill(100.0, ts=100 * S)
    v_for = {v.ts_ns: v for v in ledger_verdicts([late, early], pr, window_ns=S)}
    assert v_for[early.ts_ns].phantom_excess == 0.0  # earliest got the volume
    assert v_for[late.ts_ns].phantom_excess == 100.0


def test_ledger_no_tape_all_phantom():
    fills = [_fill(57.0, ts=100 * S), _fill(13.0, ts=200 * S)]
    vs = ledger_verdicts(fills, _prints([]), window_ns=S)
    assert all(v.is_phantom and not v.tape_covered for v in vs)
    assert [v.phantom_excess for v in vs] == [57.0, 13.0]


# --------------------------------------------------------------------------- #
# aggregate — per-cell rollup
# --------------------------------------------------------------------------- #
def test_aggregate_rolls_up_counts_sizes_and_fractions():
    pr_thin = _prints([(100 * S, 0.90, 30.0, "buy")])
    pr_thick = _prints([(100 * S, 0.90, 200.0, "buy")])
    empty = _prints([])
    verdicts = [
        classify_fill(_fill(100.0), pr_thin, window_ns=S),    # phantom, excess 70
        classify_fill(_fill(50.0), pr_thick, window_ns=S),    # ok
        classify_fill(_fill(40.0), empty, window_ns=S),       # phantom no-tape, excess 40
    ]
    agg = aggregate(verdicts)
    assert isinstance(agg, CellAgg)
    assert agg.n_fills == 3
    assert agg.n_phantom == 2
    assert agg.n_no_tape == 1
    assert agg.sim_size == 190.0
    assert agg.phantom_excess_size == 110.0
    assert agg.hittable_size == 230.0
    assert abs(agg.phantom_fill_frac - 2 / 3) < 1e-12
    assert abs(agg.phantom_size_frac - 110.0 / 190.0) < 1e-12


def test_aggregate_empty_is_all_zero_no_div0():
    agg = aggregate([])
    assert agg.n_fills == 0
    assert agg.phantom_fill_frac == 0.0
    assert agg.phantom_size_frac == 0.0


# --------------------------------------------------------------------------- #
# IO layer — sim-fill loading + matrix wiring
# --------------------------------------------------------------------------- #
def _write_fills(run_dir, rows):
    run_dir.mkdir(parents=True, exist_ok=True)
    cols = ["cloid", "ts_ns", "side", "price", "size", "question_id", "symbol", "is_hedge"]
    pd.DataFrame(rows, columns=cols).to_parquet(run_dir / "fills.parquet", index=False)


def test_load_sim_fills_excludes_settle_and_hedge(tmp_path):
    run = tmp_path / "ioc_v31_bucket_0607"
    _write_fills(run, [
        ("c1", 100 * S, "buy", 0.90, 57.0, "Q31", "#1670", False),   # keep
        ("settle", 200 * S, "buy", 1.00, 57.0, "Q31", "#1670", False),  # drop (settle)
        ("c2", 300 * S, "sell", 0.50, 10.0, "Q31", "BTC", True),     # drop (hedge)
    ])
    fills = load_sim_fills(run)
    assert [f.cloid for f in fills] == ["c1"]


def test_load_sim_fills_missing_dir_is_empty(tmp_path):
    assert load_sim_fills(tmp_path / "nope") == []


def test_run_matrix_marks_missing_cells_none(tmp_path):
    # Only one cell present; the other 15 should come back None.
    run = tmp_path / "ioc_v31_bucket_0607"
    _write_fills(run, [("c1", 100 * S, "buy", 0.90, 57.0, "Q31", "#1670", False)])

    def loader(symbol):
        return _prints([(100 * S, 0.90, 30.0, "buy")])  # 30 < 57 -> phantom

    cells, verdicts = run_matrix(
        runs_dir=tmp_path, prefix="ioc_", tape_loader=loader, window_ns=S,
    )
    assert cells[("0607", "v31", "bucket")].n_phantom == 1
    assert cells[("0606", "v1", "binary")] is None
    assert len(verdicts) == 1


def test_validate_cell_uses_injected_loader():
    fills = [FillRecord("c1", 100 * S, "buy", 0.90, 100.0, "#1670", "Q31")]
    agg, verdicts = validate_cell(
        fills, lambda sym: _prints([(100 * S, 0.90, 40.0, "buy")]), window_ns=S,
    )
    assert agg.n_fills == 1 and agg.n_phantom == 1
    assert verdicts[0].phantom_excess == 60.0
