"""SHR-90 — emit the sim per-market hand-off (the SimMarket JSON that
sim_fidelity_report consumes) from a backtest run's fills + diagnostics."""

from __future__ import annotations

import pandas as pd

from tools.sim_markets_from_run import build_sim_markets


def _fills(rows):
    return pd.DataFrame(
        rows,
        columns=[
            "cloid",
            "symbol",
            "question_idx",
            "ts_ns",
            "realized_pnl_at_settle",
            "is_hedge",
        ],
    )


def _diag(rows):
    return pd.DataFrame(rows, columns=["question_idx", "ts_ns", "action", "sigma", "ref_price"])


def test_one_record_per_traded_symbol_with_entry_inputs():
    fills = _fills(
        [
            ("a", "#2250", 1000225, 100, -103.2, False),
            ("b", "#2250", 1000225, 200, -103.2, False),
            ("c", "#1640", 1000164, 50, 81.7, False),
        ]
    )
    diag = _diag(
        [
            (1000225, 90, "hold", float("nan"), 63500.0),
            (1000225, 100, "enter", 0.131, 63579.0),  # first entry → representative inputs
            (1000225, 150, "enter", 0.590, 63200.0),
            (1000164, 40, "enter", 0.200, 41000.0),
        ]
    )
    out = {m["symbol"]: m for m in build_sim_markets(fills, diag)}

    assert set(out) == {"#2250", "#1640"}
    assert out["#2250"]["question_idx"] == 1000225
    assert out["#2250"]["realized_pnl"] == -103.2
    assert out["#2250"]["n_fills"] == 2
    assert out["#2250"]["traded"] is True
    assert out["#2250"]["sigma"] == 0.131  # first ENTER row, not the hold NaN
    assert out["#2250"]["reference_price"] == 63579.0
    assert out["#1640"]["n_fills"] == 1


def test_hedge_fills_excluded_from_count():
    fills = _fills(
        [
            ("a", "#2250", 1000225, 100, -5.0, False),
            ("h", "#2250", 1000225, 110, -5.0, True),  # hedge leg — not a strategy fill
        ]
    )
    out = build_sim_markets(fills, _diag([]))
    assert out[0]["n_fills"] == 1


def test_missing_diagnostics_yields_null_inputs():
    fills = _fills([("a", "#2250", 1000225, 100, -5.0, False)])
    out = build_sim_markets(fills, _diag([]))
    assert out[0]["sigma"] is None
    assert out[0]["reference_price"] is None
