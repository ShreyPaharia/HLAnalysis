from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

from hlanalysis.backtest.runner.result import FillRow, write_fills


def test_is_hedge_column_round_trip(tmp_path: Path) -> None:
    rows = [
        FillRow(
            cloid="c1", ts_ns=1, side="buy", price=1.0, size=2.0, fee=0.0,
            question_id="q", question_idx=0, symbol="YES",
            entry_p_model=None, entry_edge_chosen_side=None,
            entry_sigma=None, entry_tau_yr=None,
            realized_pnl_at_settle=0.0, resolved_outcome=None, is_hedge=False,
        ),
        FillRow(
            cloid="c2", ts_ns=2, side="sell", price=100.0, size=0.05, fee=0.001,
            question_id="q", question_idx=0, symbol="BTC-PERP",
            entry_p_model=None, entry_edge_chosen_side=None,
            entry_sigma=None, entry_tau_yr=None,
            realized_pnl_at_settle=0.0, resolved_outcome=None, is_hedge=True,
        ),
    ]
    p = tmp_path / "fills.parquet"
    write_fills(rows, p)
    tbl = pq.read_table(p)
    assert tbl.column_names[-1] == "is_hedge"
    assert tbl["is_hedge"].to_pylist() == [False, True]
