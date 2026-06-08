"""SHR-90 standing sim-vs-live validation pipeline — core reconciliation +
residual attribution.

Synthetic sim/live decision pairs exercise the three attribution buckets
(input-skew / execution / unmodeled-halt) and the invariant that the buckets
PARTITION the per-market PnL residual exactly.
"""
from __future__ import annotations

import json

import pytest

from hlanalysis.parity.validation import (
    BUCKET_INPUT_SKEW,
    DecisionPair,
    TradeLeg,
    attribute_residual,
    build_report,
    format_report,
    reconcile_market,
)


def _leg(pnl: float, *, price: float | None = 0.5, size: float = 1.0) -> TradeLeg:
    return TradeLeg(realized_pnl=pnl, fill_price=price, fill_size=size)


# --------------------------------------------------------------------------- #
# attribute_residual — the three buckets
# --------------------------------------------------------------------------- #
def test_matched_decision_attributes_to_execution():
    # Both sim and live traded the same decision; the PnL gap is fill-level
    # (real venue vs simulated fill) → execution.
    pairs = [DecisionPair(key="q1:buy", sim=_leg(10.0), live=_leg(7.0))]
    a = attribute_residual(pairs)
    assert a.execution == pytest.approx(3.0)
    assert a.input_skew == 0.0
    assert a.unmodeled_halt == 0.0


def test_one_sided_under_live_halt_attributes_to_halt():
    # Sim entered, live did NOT — and the journal shows a live halt gate was
    # active at decision time → the divergence is an unmodeled halt.
    pairs = [
        DecisionPair(key="q1:buy", sim=_leg(5.0), live=None, live_halt_active=True),
    ]
    a = attribute_residual(pairs)
    assert a.unmodeled_halt == pytest.approx(5.0)
    assert a.execution == 0.0
    assert a.input_skew == 0.0


def test_one_sided_with_diverged_inputs_attributes_to_input_skew():
    # Live entered, sim did not; no halt, but the sim's evaluate() inputs
    # differed materially from the journal's → different decision = input skew.
    pairs = [
        DecisionPair(key="q1:buy", sim=None, live=_leg(8.0), inputs_diverged=True),
    ]
    a = attribute_residual(pairs)
    assert a.input_skew == pytest.approx(-8.0)  # sim 0 - live 8
    assert a.execution == 0.0
    assert a.unmodeled_halt == 0.0


def test_one_sided_no_halt_no_divergence_falls_back_to_execution():
    # Same inputs, no halt, yet only one side traded → a reject / marketability
    # difference, i.e. execution.
    pairs = [DecisionPair(key="q1:buy", sim=None, live=_leg(4.0))]
    a = attribute_residual(pairs)
    assert a.execution == pytest.approx(-4.0)
    assert a.input_skew == 0.0
    assert a.unmodeled_halt == 0.0


def test_buckets_partition_total_residual_exactly():
    pairs = [
        DecisionPair(key="a", sim=_leg(10.0), live=_leg(7.0)),               # exec +3
        DecisionPair(key="b", sim=_leg(5.0), live=None, live_halt_active=True),  # halt +5
        DecisionPair(key="c", sim=None, live=_leg(8.0), inputs_diverged=True),   # skew -8
        DecisionPair(key="d", sim=None, live=_leg(4.0)),                     # exec -4
    ]
    a = attribute_residual(pairs)
    total_residual = sum(
        (p.sim.realized_pnl if p.sim else 0.0)
        - (p.live.realized_pnl if p.live else 0.0)
        for p in pairs
    )
    assert a.total == pytest.approx(total_residual)
    assert a.input_skew + a.execution + a.unmodeled_halt == pytest.approx(a.total)
    assert a.dominant_bucket == BUCKET_INPUT_SKEW  # |−8| is the largest


def test_halt_takes_precedence_over_input_divergence():
    # A one-sided decision that is BOTH halted and input-diverged is booked to
    # halt (the gate is the proximate cause — live could not have traded).
    pairs = [
        DecisionPair(
            key="q1", sim=_leg(6.0), live=None,
            live_halt_active=True, inputs_diverged=True,
        )
    ]
    a = attribute_residual(pairs)
    assert a.unmodeled_halt == pytest.approx(6.0)
    assert a.input_skew == 0.0


def test_dominant_bucket_none_when_no_residual():
    pairs = [DecisionPair(key="a", sim=_leg(5.0), live=_leg(5.0))]
    a = attribute_residual(pairs)
    assert a.total == pytest.approx(0.0)
    assert a.dominant_bucket is None


# --------------------------------------------------------------------------- #
# reconcile_market
# --------------------------------------------------------------------------- #
def test_reconcile_market_aggregates_pnl_and_counts():
    pairs = [
        DecisionPair(key="a", sim=_leg(10.0), live=_leg(7.0)),
        DecisionPair(key="b", sim=_leg(5.0), live=None, live_halt_active=True),
        DecisionPair(key="c", sim=None, live=_leg(8.0), inputs_diverged=True),
    ]
    m = reconcile_market(question_idx=42, symbol="BTC", pairs=pairs)
    assert m.question_idx == 42
    assert m.symbol == "BTC"
    assert m.sim_pnl == pytest.approx(15.0)
    assert m.live_pnl == pytest.approx(15.0)
    assert m.residual == pytest.approx(0.0)
    assert m.n_matched == 1
    assert m.n_sim_only == 1
    assert m.n_live_only == 1
    # residual is exactly partitioned by the attribution even when it nets to 0
    assert m.attribution.execution == pytest.approx(3.0)
    assert m.attribution.unmodeled_halt == pytest.approx(5.0)
    assert m.attribution.input_skew == pytest.approx(-8.0)


def test_reconcile_market_empty_is_perfect_parity():
    m = reconcile_market(question_idx=1, symbol="ETH", pairs=[])
    assert m.sim_pnl == 0.0
    assert m.live_pnl == 0.0
    assert m.residual == 0.0
    assert m.attribution.total == 0.0


def test_market_to_dict_is_json_serialisable():
    pairs = [DecisionPair(key="a", sim=_leg(10.0), live=_leg(7.0))]
    m = reconcile_market(question_idx=7, symbol="SOL", pairs=pairs)
    d = m.to_dict()
    s = json.dumps(d)  # must not raise
    back = json.loads(s)
    assert back["question_idx"] == 7
    assert back["residual"] == pytest.approx(3.0)
    assert back["attribution"]["execution"] == pytest.approx(3.0)


# --------------------------------------------------------------------------- #
# build_report — aggregate + the monitored number + gate verdict
# --------------------------------------------------------------------------- #
def _market(qidx, symbol, pairs):
    return reconcile_market(question_idx=qidx, symbol=symbol, pairs=pairs)


def test_report_aggregates_residual_and_buckets():
    markets = [
        _market(1, "BTC", [DecisionPair(key="a", sim=_leg(10.0), live=_leg(7.0))]),
        _market(2, "ETH", [DecisionPair(key="b", sim=None, live=_leg(8.0), inputs_diverged=True)]),
    ]
    r = build_report(markets, run_ts_ns=1_000)
    assert r.n_markets == 2
    assert r.total_residual == pytest.approx(3.0 - 8.0)
    assert r.abs_residual == pytest.approx(3.0 + 8.0)
    assert r.attribution.execution == pytest.approx(3.0)
    assert r.attribution.input_skew == pytest.approx(-8.0)
    # the monitored number: |attributed residual| relative to live PnL magnitude
    assert r.residual_ratio == pytest.approx(11.0 / 15.0)
    assert r.dominant_bucket == BUCKET_INPUT_SKEW


def test_report_trustworthy_gate_passes_under_threshold():
    # Near-perfect parity: tiny execution residual on a large live PnL.
    markets = [
        _market(1, "BTC", [DecisionPair(key="a", sim=_leg(100.5), live=_leg(100.0))]),
    ]
    r = build_report(markets, run_ts_ns=1, residual_ratio_threshold=0.1)
    assert r.residual_ratio == pytest.approx(0.5 / 100.0)
    assert r.trustworthy is True


def test_report_trustworthy_gate_fails_over_threshold():
    markets = [
        _market(1, "BTC", [DecisionPair(key="a", sim=_leg(50.0), live=_leg(100.0))]),
    ]
    r = build_report(markets, run_ts_ns=1, residual_ratio_threshold=0.1)
    assert r.residual_ratio == pytest.approx(0.5)
    assert r.trustworthy is False


def test_report_empty_is_trustworthy_zero_ratio():
    r = build_report([], run_ts_ns=1)
    assert r.n_markets == 0
    assert r.residual_ratio == 0.0
    assert r.trustworthy is True


def test_report_to_dict_round_trips_json():
    markets = [
        _market(1, "BTC", [DecisionPair(key="a", sim=_leg(10.0), live=_leg(7.0))]),
    ]
    r = build_report(markets, run_ts_ns=999)
    d = r.to_dict()
    s = json.dumps(d)
    back = json.loads(s)
    assert back["summary"]["n_markets"] == 1
    assert back["summary"]["run_ts_ns"] == 999
    assert len(back["markets"]) == 1
    assert back["markets"][0]["question_idx"] == 1


def test_format_report_renders_human_summary():
    markets = [
        _market(1, "BTC", [DecisionPair(key="a", sim=_leg(10.0), live=_leg(7.0))]),
        _market(2, "ETH", [DecisionPair(key="b", sim=None, live=_leg(8.0), inputs_diverged=True)]),
    ]
    r = build_report(markets, run_ts_ns=1)
    text = format_report(r)
    assert "markets" in text.lower()
    assert "input" in text.lower()
    assert "execution" in text.lower()
    assert "halt" in text.lower()
    # per-market lines present
    assert "BTC" in text
    assert "ETH" in text
