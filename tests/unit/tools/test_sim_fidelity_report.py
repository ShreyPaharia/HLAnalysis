"""SHR-90 — the sim-fidelity CLI: load sim markets from JSON, generate the
report from the live DB + sim markets, append the monitoring time series, and
gate the process exit code on trustworthiness.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hlanalysis.engine.state import (
    FILL_SOURCE_VENUE,
    Fill,
    StateDAL,
    TradeJournalRow,
)
from hlanalysis.parity.sources import SimMarket
from tools.sim_fidelity_report import (
    append_timeseries,
    generate_report,
    load_sim_markets_from_json,
    main,
)


def _dal(tmp_path: Path) -> StateDAL:
    dal = StateDAL(tmp_path / "state.db")
    dal.run_migrations()
    return dal


def _seed_live_market(dal: StateDAL, *, q: int, closed_pnl: float, settle: float, sigma: float):
    dal.add_journal_decision(TradeJournalRow(
        cloid=f"c{q}", question_idx=q, decision_ts_ns=1_000, action="enter",
        side="buy", symbol="#10", sigma=sigma, reference_price=100.0,
        fill_px=0.5, fill_sz=1.0, fill_ts_ns=1_500,
    ))
    dal.append_fill(Fill(
        fill_id=f"f{q}", cloid=f"c{q}", question_idx=q, symbol="#10", side="buy",
        price=0.5, size=1.0, fee=0.0, ts_ns=1_500, closed_pnl=closed_pnl,
        source=FILL_SOURCE_VENUE,
    ))
    if settle:
        dal.record_settlement(question_idx=q, symbol="#10", realized_pnl=settle, ts_ns=2_000)


# --------------------------------------------------------------------------- #
# load_sim_markets_from_json
# --------------------------------------------------------------------------- #
def test_load_sim_markets_from_json(tmp_path):
    p = tmp_path / "sim.json"
    p.write_text(json.dumps([
        {"question_idx": 1, "symbol": "#10", "realized_pnl": 10.0,
         "traded": True, "n_fills": 2, "sigma": 0.5, "reference_price": 100.0},
        {"question_idx": 2, "symbol": "#10", "realized_pnl": 0.0,
         "traded": False, "n_fills": 0},
    ]))
    sims = load_sim_markets_from_json(p)
    assert len(sims) == 2
    assert sims[0] == SimMarket(1, "#10", 10.0, True, 0.5, 100.0, 2)
    assert sims[1].traded is False
    assert sims[1].sigma is None


# --------------------------------------------------------------------------- #
# generate_report
# --------------------------------------------------------------------------- #
def test_generate_report_from_db_and_sims(tmp_path):
    dal = _dal(tmp_path)
    _seed_live_market(dal, q=1, closed_pnl=0.0, settle=7.0, sigma=0.5)
    sims = [SimMarket(1, "#10", 10.0, True, 0.5, 100.0, 1)]
    report = generate_report([dal.db_path], sims, run_ts_ns=1234)
    assert report.n_markets == 1
    assert report.total_live_pnl == pytest.approx(7.0)
    assert report.total_sim_pnl == pytest.approx(10.0)
    assert report.attribution.execution == pytest.approx(3.0)
    assert report.run_ts_ns == 1234


def test_generate_report_merges_multiple_slot_dbs(tmp_path):
    dal_a = StateDAL(tmp_path / "a" / "state.db"); dal_a.run_migrations()
    dal_b = StateDAL(tmp_path / "b" / "state.db"); dal_b.run_migrations()
    _seed_live_market(dal_a, q=1, closed_pnl=0.0, settle=7.0, sigma=0.5)
    _seed_live_market(dal_b, q=2, closed_pnl=0.0, settle=8.0, sigma=0.5)
    sims = [
        SimMarket(1, "#10", 7.0, True, 0.5, 100.0, 1),
        SimMarket(2, "#10", 8.0, True, 0.5, 100.0, 1),
    ]
    report = generate_report([dal_a.db_path, dal_b.db_path], sims, run_ts_ns=1)
    assert report.n_markets == 2
    assert report.total_residual == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# append_timeseries
# --------------------------------------------------------------------------- #
def test_append_timeseries_writes_one_jsonl_row_per_call(tmp_path):
    dal = _dal(tmp_path)
    _seed_live_market(dal, q=1, closed_pnl=0.0, settle=7.0, sigma=0.5)
    sims = [SimMarket(1, "#10", 10.0, True, 0.5, 100.0, 1)]
    ts_path = tmp_path / "history.jsonl"
    r1 = generate_report([dal.db_path], sims, run_ts_ns=1)
    r2 = generate_report([dal.db_path], sims, run_ts_ns=2)
    append_timeseries(ts_path, r1)
    append_timeseries(ts_path, r2)
    lines = ts_path.read_text().strip().splitlines()
    assert len(lines) == 2
    row0 = json.loads(lines[0])
    assert row0["run_ts_ns"] == 1
    assert row0["residual_ratio"] == pytest.approx(3.0 / 7.0)
    assert "input_skew" in row0 and "execution" in row0 and "unmodeled_halt" in row0


# --------------------------------------------------------------------------- #
# main — CLI wiring + gate exit code
# --------------------------------------------------------------------------- #
def test_main_writes_json_and_gates_exit_code(tmp_path):
    dal = _dal(tmp_path)
    # Big execution residual → should FAIL the gate (exit 1).
    _seed_live_market(dal, q=1, closed_pnl=0.0, settle=100.0, sigma=0.5)
    sim_path = tmp_path / "sim.json"
    sim_path.write_text(json.dumps([
        {"question_idx": 1, "symbol": "#10", "realized_pnl": 10.0,
         "traded": True, "n_fills": 1, "sigma": 0.5, "reference_price": 100.0},
    ]))
    out_path = tmp_path / "report.json"
    ts_path = tmp_path / "history.jsonl"
    rc = main([
        "--db", str(dal.db_path),
        "--sim", str(sim_path),
        "--json", str(out_path),
        "--timeseries", str(ts_path),
        "--run-ts-ns", "777",
        "--threshold", "0.1",
    ])
    assert rc == 1  # residual_ratio 0.9 > 0.1 → not trustworthy
    doc = json.loads(out_path.read_text())
    assert doc["summary"]["n_markets"] == 1
    assert doc["summary"]["run_ts_ns"] == 777
    assert doc["markets"][0]["question_idx"] == 1
    # timeseries row appended
    assert len(ts_path.read_text().strip().splitlines()) == 1


def test_main_passes_gate_on_parity(tmp_path):
    dal = _dal(tmp_path)
    _seed_live_market(dal, q=1, closed_pnl=0.0, settle=100.0, sigma=0.5)
    sim_path = tmp_path / "sim.json"
    sim_path.write_text(json.dumps([
        {"question_idx": 1, "symbol": "#10", "realized_pnl": 100.0,
         "traded": True, "n_fills": 1, "sigma": 0.5, "reference_price": 100.0},
    ]))
    rc = main([
        "--db", str(dal.db_path),
        "--sim", str(sim_path),
        "--run-ts-ns", "1",
    ])
    assert rc == 0


def test_generate_report_by_symbol_matches_mismatched_question_idx(tmp_path):
    """SHR-90: reconcile a backtest by symbol — live venue fills carry q=-1 and
    the sim uses a different question_idx, but the symbol matches."""
    from hlanalysis.engine.state import FILL_SOURCE_VENUE, Fill, StateDAL
    dal = StateDAL(tmp_path / "state.db")
    dal.run_migrations()
    # live: a venue fill on #10 with the unattributed -1 placeholder
    dal.append_fill(Fill(
        fill_id="f1", cloid="c1", question_idx=-1, symbol="#10", side="sell",
        price=0.8, size=100.0, fee=0.0, ts_ns=1_000, closed_pnl=5.0,
        source=FILL_SOURCE_VENUE,
    ))
    sim = [SimMarket(question_idx=1000001, symbol="#10", realized_pnl=2.0,
                     traded=True, sigma=None, reference_price=None, n_fills=2)]
    rep = generate_report([dal.db_path], sim, run_ts_ns=1, key="symbol")
    s = rep.to_dict()["summary"]
    # one matched market, residual = sim 2.0 - live 5.0 = -3.0
    assert s["n_markets"] == 1
    assert s["total_residual"] == pytest.approx(-3.0)
