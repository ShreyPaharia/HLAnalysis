from __future__ import annotations

from pathlib import Path

import pytest

from hlanalysis.backtest.data.synthetic import (
    SyntheticDataSource,
    build_dummy_enter_strategy,
    make_default_binary_question,
)
from hlanalysis.backtest.runner.hftbt_runner import RunConfig, run_one_question


@pytest.fixture(scope="module")
def synthetic_source() -> tuple[SyntheticDataSource, object]:
    sq = make_default_binary_question(start_ts_ns=0)
    ds = SyntheticDataSource()
    ds.add_question(sq)
    return ds, sq


def test_run_yes_wins_records_enter_and_settle(synthetic_source, tmp_path):
    ds, sq = synthetic_source
    strat = build_dummy_enter_strategy({"size": 10.0})
    cfg = RunConfig(
        scanner_interval_seconds=60,
        tick_size=0.001,
        lot_size=1.0,
        slippage_bps=0.0,  # exact bookkeeping
        fee_taker=0.0,
        book_depth_assumption=10_000.0,
    )
    res = run_one_question(
        strat,
        ds,
        sq.descriptor,
        cfg,
        diagnostics_dir=tmp_path / "diag",
        fills_dir=tmp_path / "fills",
        strike=sq.strike,
    )
    # 1 ENTER + 1 settle fill
    assert len(res.fills) == 2, [f for f in res.fills]
    enter = res.fills[0]
    settle = res.fills[1]
    assert enter.side == "buy"
    assert enter.size == 10.0
    assert 0.30 < enter.price < 0.50  # first scan tick, YES ask ≈ 0.32-0.40
    assert settle.cloid == "settle"
    assert settle.price == 1.0  # YES won
    assert settle.size == 10.0
    # Realized P&L = settle.notional - enter.notional - fees
    expected_pnl = settle.price * settle.size - enter.price * enter.size
    assert res.realized_pnl_usd is not None
    assert res.realized_pnl_usd == pytest.approx(expected_pnl, abs=1e-9)


def test_run_no_outcome_settles_zero(tmp_path):
    """If the resolved outcome is NO but we held YES, settle price is 0."""
    sq = make_default_binary_question(start_ts_ns=0, outcome="no")
    ds = SyntheticDataSource()
    ds.add_question(sq)
    strat = build_dummy_enter_strategy({"size": 5.0})
    cfg = RunConfig(slippage_bps=0.0, fee_taker=0.0)
    res = run_one_question(
        strat,
        ds,
        sq.descriptor,
        cfg,
        diagnostics_dir=tmp_path / "diag",
        fills_dir=tmp_path / "fills",
        strike=sq.strike,
    )
    settle = res.fills[-1]
    assert settle.cloid == "settle"
    assert settle.price == 0.0  # YES position, NO outcome → 0
    assert res.realized_pnl_usd is not None
    assert res.realized_pnl_usd < 0  # paid for YES, got nothing


def test_run_persists_parquet_artifacts(synthetic_source, tmp_path):
    ds, sq = synthetic_source
    strat = build_dummy_enter_strategy({"size": 10.0})
    cfg = RunConfig(slippage_bps=0.0, fee_taker=0.0)
    run_one_question(
        strat,
        ds,
        sq.descriptor,
        cfg,
        diagnostics_dir=tmp_path / "diag",
        fills_dir=tmp_path / "fills",
        strike=sq.strike,
    )
    diag = tmp_path / "diag" / f"{sq.descriptor.question_id}.parquet"
    fills = tmp_path / "fills" / f"{sq.descriptor.question_id}.parquet"
    assert diag.exists()
    assert fills.exists()

    import pyarrow.parquet as pq

    d = pq.read_table(diag).to_pydict()
    f = pq.read_table(fills).to_pydict()
    # Diagnostics: ts_ns monotonic and one row per scan tick (10 ticks over 10 min)
    assert len(d["ts_ns"]) > 0
    assert d["ts_ns"] == sorted(d["ts_ns"])
    # Fills: at least one entry + a settle row with resolved_outcome populated
    assert "settle" in f["cloid"]
    settle_idx = f["cloid"].index("settle")
    assert f["resolved_outcome"][settle_idx] == "yes"
