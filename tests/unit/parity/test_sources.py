"""SHR-90 — building DecisionPairs from normalized per-market sim/live records,
and the read-only loaders over the live trade-journal DB and sim RunResults.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hlanalysis.backtest.runner.result import Fill as SimFill, RunResult
from hlanalysis.engine.state import (
    FILL_SOURCE_VENUE,
    Fill,
    StateDAL,
    TradeJournalRow,
)
from hlanalysis.parity.sources import (
    LiveMarket,
    SimMarket,
    build_pairs,
    load_live_markets_from_db,
    reconcile_markets,
    sim_market_from_run,
)
from hlanalysis.parity.validation import (
    BUCKET_EXECUTION,
    BUCKET_INPUT_SKEW,
    BUCKET_UNMODELED_HALT,
)


# --------------------------------------------------------------------------- #
# build_pairs — matching markets + flag derivation
# --------------------------------------------------------------------------- #
def test_matched_market_pairs_both_legs():
    live = [LiveMarket(question_idx=1, symbol="BTC", realized_pnl=7.0, traded=True,
                       halt_active=False, sigma=0.5, reference_price=100.0, n_fills=2)]
    sim = [SimMarket(question_idx=1, symbol="BTC", realized_pnl=10.0, traded=True,
                     sigma=0.5, reference_price=100.0, n_fills=2)]
    pairs = build_pairs(live, sim)
    assert len(pairs) == 1
    p = pairs[0]
    assert p.sim is not None and p.live is not None
    assert p.sim.realized_pnl == 10.0
    assert p.live.realized_pnl == 7.0
    assert p.is_matched


def test_live_only_market_under_halt():
    live = [LiveMarket(question_idx=1, symbol="BTC", realized_pnl=4.0, traded=True,
                       halt_active=True, sigma=0.5, reference_price=100.0, n_fills=1)]
    sim = [SimMarket(question_idx=1, symbol="BTC", realized_pnl=0.0, traded=False,
                     sigma=0.5, reference_price=100.0, n_fills=0)]
    pairs = build_pairs(live, sim)
    p = pairs[0]
    assert p.sim is None
    assert p.live is not None
    assert p.live_halt_active is True


def test_input_divergence_detected_from_sigma():
    live = [LiveMarket(question_idx=1, symbol="BTC", realized_pnl=8.0, traded=True,
                       halt_active=False, sigma=0.5, reference_price=100.0, n_fills=1)]
    # sim did not trade and its σ is materially different → input skew
    sim = [SimMarket(question_idx=1, symbol="BTC", realized_pnl=0.0, traded=False,
                     sigma=0.9, reference_price=100.0, n_fills=0)]
    pairs = build_pairs(live, sim, sigma_rel_tol=0.05)
    p = pairs[0]
    assert p.inputs_diverged is True


def test_no_input_divergence_when_inputs_close():
    live = [LiveMarket(question_idx=1, symbol="BTC", realized_pnl=8.0, traded=True,
                       halt_active=False, sigma=0.500, reference_price=100.0, n_fills=1)]
    sim = [SimMarket(question_idx=1, symbol="BTC", realized_pnl=0.0, traded=False,
                     sigma=0.501, reference_price=100.01, n_fills=0)]
    pairs = build_pairs(live, sim, sigma_rel_tol=0.05, ref_rel_tol=0.05)
    assert pairs[0].inputs_diverged is False


def test_sim_only_market_creates_live_none_pair():
    live: list[LiveMarket] = []
    sim = [SimMarket(question_idx=9, symbol="ETH", realized_pnl=3.0, traded=True,
                     sigma=None, reference_price=None, n_fills=1)]
    pairs = build_pairs(live, sim)
    p = pairs[0]
    assert p.live is None
    assert p.sim is not None and p.sim.realized_pnl == 3.0


def test_reconcile_markets_end_to_end_attribution():
    live = [
        LiveMarket(1, "BTC", 7.0, True, False, 0.5, 100.0, 2),
        LiveMarket(2, "ETH", 8.0, True, False, 0.5, 100.0, 1),
        LiveMarket(3, "SOL", 4.0, True, True, 0.5, 100.0, 1),
    ]
    sim = [
        SimMarket(1, "BTC", 10.0, True, 0.5, 100.0, 2),   # matched → execution +3
        SimMarket(2, "ETH", 0.0, False, 0.9, 100.0, 0),   # live-only, σ diverged → input-skew -8
        SimMarket(3, "SOL", 0.0, False, 0.5, 100.0, 0),   # live-only, halt → unmodeled-halt -4
    ]
    markets = reconcile_markets(live, sim, sigma_rel_tol=0.05)
    by_q = {m.question_idx: m for m in markets}
    assert by_q[1].attribution.dominant_bucket == BUCKET_EXECUTION
    assert by_q[2].attribution.dominant_bucket == BUCKET_INPUT_SKEW
    assert by_q[3].attribution.dominant_bucket == BUCKET_UNMODELED_HALT
    assert by_q[1].attribution.execution == pytest.approx(3.0)
    assert by_q[2].attribution.input_skew == pytest.approx(-8.0)
    assert by_q[3].attribution.unmodeled_halt == pytest.approx(-4.0)


# --------------------------------------------------------------------------- #
# sim_market_from_run — RunResult adapter
# --------------------------------------------------------------------------- #
def test_sim_market_from_run_reads_pnl_and_fills():
    rr = RunResult(
        fills=[SimFill(cloid="c1", symbol="BTC", side="buy", price=0.5, size=1.0, fee=0.0, partial=False)],
        n_decisions=1,
        realized_pnl_usd=12.5,
    )
    sm = sim_market_from_run(question_idx=5, symbol="BTC", result=rr, sigma=0.4, reference_price=99.0)
    assert sm.question_idx == 5
    assert sm.realized_pnl == 12.5
    assert sm.traded is True
    assert sm.n_fills == 1
    assert sm.sigma == 0.4


def test_sim_market_from_run_no_fills_not_traded():
    rr = RunResult(fills=[], n_decisions=0, realized_pnl_usd=None)
    sm = sim_market_from_run(question_idx=5, symbol="BTC", result=rr)
    assert sm.traded is False
    assert sm.realized_pnl == 0.0
    assert sm.n_fills == 0


# --------------------------------------------------------------------------- #
# load_live_markets_from_db — read-only over the engine state DB
# --------------------------------------------------------------------------- #
def _dal(tmp_path: Path) -> StateDAL:
    dal = StateDAL(tmp_path / "state.db")
    dal.run_migrations()
    return dal


def test_load_live_markets_joins_journal_fills_and_settlement(tmp_path):
    dal = _dal(tmp_path)
    # An entry decision (journal) for question 1, with a venue fill + settlement.
    dal.add_journal_decision(TradeJournalRow(
        cloid="c1", question_idx=1, decision_ts_ns=1_000, action="enter",
        side="buy", symbol="#10", sigma=0.6, reference_price=100.0,
        halt_json=None, fill_px=0.5, fill_sz=1.0, fill_ts_ns=1_500,
    ))
    dal.append_fill(Fill(
        fill_id="f1", cloid="c1", question_idx=1, symbol="#10", side="buy",
        price=0.5, size=1.0, fee=0.0, ts_ns=1_500, closed_pnl=0.0,
        source=FILL_SOURCE_VENUE,
    ))
    dal.record_settlement(question_idx=1, symbol="#10", realized_pnl=5.0, ts_ns=2_000)

    markets = load_live_markets_from_db(dal.db_path)
    assert len(markets) == 1
    m = markets[0]
    assert m.question_idx == 1
    assert m.symbol == "#10"
    assert m.realized_pnl == pytest.approx(5.0)   # closed_pnl 0 + settlement 5
    assert m.traded is True
    assert m.halt_active is False
    assert m.sigma == pytest.approx(0.6)
    assert m.n_fills == 1


def test_load_live_markets_flags_halt_from_journal(tmp_path):
    dal = _dal(tmp_path)
    dal.add_journal_decision(TradeJournalRow(
        cloid="c1", question_idx=2, decision_ts_ns=1_000, action="enter",
        side="buy", symbol="#10", sigma=0.6, reference_price=100.0,
        halt_json='{"restart_blocked": false, "daily_loss_halted": true, '
                  '"reject_breaker_tripped": false, "stale_reference": false}',
    ))
    markets = load_live_markets_from_db(dal.db_path)
    assert markets[0].halt_active is True


def test_load_live_markets_rejected_only_not_traded(tmp_path):
    dal = _dal(tmp_path)
    dal.add_journal_decision(TradeJournalRow(
        cloid="c1", question_idx=3, decision_ts_ns=1_000, action="enter",
        side="buy", symbol="#10", reject_reason="not_marketable",
    ))
    markets = load_live_markets_from_db(dal.db_path)
    assert markets[0].traded is False
    assert markets[0].realized_pnl == 0.0
    assert markets[0].n_fills == 0
