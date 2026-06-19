"""Tests for the --decision-trace-out feature.

Verifies that DecisionTraceWriter and build_trace_row produce JSONL output
with the canonical schema, and that run_one_question writes trace rows when
a writer is passed.
"""

from __future__ import annotations

import json

import pytest

from hlanalysis.backtest.data.synthetic import (
    SyntheticDataSource,
    build_dummy_enter_strategy,
    make_default_binary_question,
)
from hlanalysis.backtest.runner._decision_trace import DecisionTraceWriter, build_trace_row
from hlanalysis.backtest.runner.hftbt_runner import RunConfig, run_one_question
from hlanalysis.strategy.types import (
    Action,
    BookState,
    Decision,
    Diagnostic,
    OrderIntent,
    QuestionView,
)

# ---------------------------------------------------------------------------
# Canonical schema — all keys that must appear in every JSONL row.
# ---------------------------------------------------------------------------

_CANONICAL_KEYS = {
    "ts_ns",
    "question_idx",
    "klass",
    "strategy_id",
    "action",
    "reason",
    "chosen_symbol",
    "chosen_side",
    "reference_price",
    "sigma",
    "p_model",
    "edge",
    "safety_d_entry",
    "safety_d_exit",
    "tte_s",
    "favorite_side",
    "intended_size",
    "intended_price",
    "bid_px",
    "bid_sz",
    "ask_px",
    "ask_sz",
    "position_qty",
    "position_avg_entry",
    "config_hash",
    "diag_fields",
}


# ---------------------------------------------------------------------------
# Unit tests for build_trace_row
# ---------------------------------------------------------------------------


def _make_question_view(*, question_idx: int = 1, klass: str = "priceBinary") -> QuestionView:
    return QuestionView(
        question_idx=question_idx,
        yes_symbol="@YES",
        no_symbol="@NO",
        strike=100000.0,
        expiry_ns=int(1e18),
        underlying="BTC",
        klass=klass,
        period="1d",
        settled=False,
        leg_symbols=("@YES", "@NO"),
        venue="hyperliquid",
    )


def _make_book(*, bid: float = 0.85, ask: float = 0.87, sz: float = 100.0) -> BookState:
    return BookState(
        symbol="@YES",
        bid_px=bid,
        bid_sz=sz,
        ask_px=ask,
        ask_sz=sz,
        last_trade_ts_ns=0,
        last_l2_ts_ns=0,
    )


def _make_enter_decision() -> Decision:
    diag = Diagnostic(
        "info",
        "edge",
        (
            ("p_model", "0.9000"),
            ("edge_yes", "0.0250"),
            ("sigma", "0.0150"),
            ("tau_yr", "0.002740"),
            ("ln_sk", "0.0123"),
        ),
    )
    intent = OrderIntent(
        question_idx=1,
        symbol="@YES",
        side="buy",
        size=100.0,
        limit_price=0.87,
        cloid="test-cloid",
    )
    return Decision(action=Action.ENTER, intents=(intent,), diagnostics=(diag,))


def _make_hold_decision(reason: str = "no_favorite") -> Decision:
    return Decision(
        action=Action.HOLD,
        diagnostics=(Diagnostic("info", reason),),
    )


class TestBuildTraceRow:
    def test_all_canonical_keys_present(self):
        """Every row must contain exactly the canonical schema keys."""
        qv = _make_question_view()
        decision = _make_enter_decision()
        row = build_trace_row(
            ts_ns=1_000_000_000,
            question=qv,
            strategy_id="v3_theta_harvester",
            decision=decision,
            reference_price=100500.0,
            books={"@YES": _make_book()},
            position=None,
            config_hash="abcdef012345",
        )
        assert set(row.keys()) == _CANONICAL_KEYS, (
            f"Missing: {_CANONICAL_KEYS - set(row.keys())}  Extra: {set(row.keys()) - _CANONICAL_KEYS}"
        )

    def test_enter_decision_extracts_fields(self):
        """An ENTER decision extracts intent, edge, sigma, p_model correctly."""
        qv = _make_question_view()
        decision = _make_enter_decision()
        row = build_trace_row(
            ts_ns=1_000_000_000,
            question=qv,
            strategy_id="v3_theta_harvester",
            decision=decision,
            reference_price=100500.0,
            books={"@YES": _make_book(bid=0.85, ask=0.87)},
            position=None,
            config_hash="abc123",
        )
        assert row["action"] == "enter"
        assert row["reason"] == "edge"
        assert row["chosen_symbol"] == "@YES"
        assert row["chosen_side"] == "buy"
        assert row["favorite_side"] == "yes"
        assert row["intended_size"] == pytest.approx(100.0)
        assert row["intended_price"] == pytest.approx(0.87)
        assert row["p_model"] == pytest.approx(0.9, abs=1e-4)
        assert row["sigma"] == pytest.approx(0.015, abs=1e-6)
        assert row["edge"] == pytest.approx(0.025, abs=1e-4)
        assert row["bid_px"] == pytest.approx(0.85)
        assert row["ask_px"] == pytest.approx(0.87)
        assert row["bid_sz"] == pytest.approx(100.0)
        assert row["ask_sz"] == pytest.approx(100.0)
        assert row["config_hash"] == "abc123"
        assert row["strategy_id"] == "v3_theta_harvester"
        assert row["question_idx"] == 1
        assert row["klass"] == "priceBinary"

    def test_hold_decision_nulls(self):
        """A HOLD decision with no intents should have None for chosen fields."""
        qv = _make_question_view()
        decision = _make_hold_decision("no_favorite")
        row = build_trace_row(
            ts_ns=5_000_000_000,
            question=qv,
            strategy_id="v3_theta_harvester",
            decision=decision,
            reference_price=99000.0,
            books={},
            position=None,
            config_hash="",
        )
        assert row["action"] == "hold"
        assert row["reason"] == "no_favorite"
        assert row["chosen_symbol"] == ""
        assert row["chosen_side"] == ""
        assert row["intended_size"] is None
        assert row["intended_price"] is None
        assert row["sigma"] is None
        assert row["p_model"] is None
        assert row["edge"] is None
        assert row["position_qty"] is None
        assert row["position_avg_entry"] is None

    def test_diag_fields_is_merged_dict(self):
        """diag_fields should be a flat dict of all kv pairs from all diagnostics."""
        qv = _make_question_view()
        decision = Decision(
            action=Action.HOLD,
            diagnostics=(
                Diagnostic("info", "first", (("k1", "v1"), ("k2", "v2"))),
                Diagnostic("info", "second", (("k3", "v3"),)),
            ),
        )
        row = build_trace_row(
            ts_ns=0,
            question=qv,
            strategy_id="test",
            decision=decision,
            reference_price=0.0,
            books={},
            position=None,
            config_hash="",
        )
        assert isinstance(row["diag_fields"], dict)
        assert row["diag_fields"]["k1"] == "v1"
        assert row["diag_fields"]["k2"] == "v2"
        assert row["diag_fields"]["k3"] == "v3"

    def test_tte_s_derived_from_expiry(self):
        """When no tte_s in diags, it is derived from expiry_ns - ts_ns."""
        expiry_ns = 10_000_000_000
        ts_ns = 3_000_000_000
        qv = QuestionView(
            question_idx=1,
            yes_symbol="@YES",
            no_symbol="@NO",
            strike=1000.0,
            expiry_ns=expiry_ns,
            underlying="BTC",
            klass="priceBinary",
            period="1d",
            settled=False,
            leg_symbols=("@YES", "@NO"),
            venue="hyperliquid",
        )
        row = build_trace_row(
            ts_ns=ts_ns,
            question=qv,
            strategy_id="test",
            decision=_make_hold_decision(),
            reference_price=0.0,
            books={},
            position=None,
            config_hash="",
        )
        assert row["tte_s"] == pytest.approx(7.0)  # (10 - 3) seconds


# ---------------------------------------------------------------------------
# Integration test: run_one_question writes trace rows
# ---------------------------------------------------------------------------


class TestDecisionTraceWriterIntegration:
    def test_writer_context_manager_creates_file(self, tmp_path):
        out = tmp_path / "trace.jsonl"
        with DecisionTraceWriter(out) as w:
            w.write({"ts_ns": 1, "action": "hold"})
        assert out.exists()
        rows = [json.loads(ln) for ln in out.read_text().splitlines() if ln.strip()]
        assert len(rows) == 1
        assert rows[0]["action"] == "hold"

    def test_writer_appends_across_calls(self, tmp_path):
        out = tmp_path / "trace.jsonl"
        with DecisionTraceWriter(out) as w:
            for i in range(3):
                w.write({"ts_ns": i, "action": "hold"})
        rows = [json.loads(ln) for ln in out.read_text().splitlines() if ln.strip()]
        assert len(rows) == 3

    def test_run_one_question_writes_trace(self, tmp_path):
        """run_one_question with a trace writer produces JSONL with canonical keys."""
        out = tmp_path / "trace.jsonl"

        sq = make_default_binary_question(start_ts_ns=0)
        ds = SyntheticDataSource()
        ds.add_question(sq)
        cfg = RunConfig(
            scanner_interval_seconds=60,
            slippage_bps=0.0,
            fee_taker=0.0,
            order_latency_ms=0.0,
            ioc_marketability_recheck=False,
        )
        strategy = build_dummy_enter_strategy({"size": 10.0})

        with DecisionTraceWriter(out) as writer:
            run_one_question(
                strategy,
                ds,
                sq.descriptor,
                cfg,
                strike=sq.strike,
                decision_trace_writer=writer,
                decision_trace_strategy_id="test_strategy",
                decision_trace_config_hash="deadbeef",
            )

        assert out.exists(), "trace file was not created"
        lines = [ln for ln in out.read_text().splitlines() if ln.strip()]
        assert len(lines) > 0, "trace file has no rows"

        for line in lines:
            row = json.loads(line)
            missing = _CANONICAL_KEYS - set(row.keys())
            assert not missing, f"Row missing canonical keys: {missing}\nRow: {row}"

        # Spot-check a few field types.
        row0 = json.loads(lines[0])
        assert isinstance(row0["ts_ns"], int)
        assert isinstance(row0["question_idx"], int)
        assert isinstance(row0["action"], str)
        assert row0["strategy_id"] == "test_strategy"
        assert row0["config_hash"] == "deadbeef"
        assert isinstance(row0["diag_fields"], dict)

    def test_no_trace_writer_no_overhead(self, tmp_path):
        """When decision_trace_writer is None, run_one_question completes normally."""
        sq = make_default_binary_question(start_ts_ns=0)
        ds = SyntheticDataSource()
        ds.add_question(sq)
        cfg = RunConfig(
            scanner_interval_seconds=60,
            slippage_bps=0.0,
            fee_taker=0.0,
            order_latency_ms=0.0,
            ioc_marketability_recheck=False,
        )
        strategy = build_dummy_enter_strategy({"size": 10.0})
        # No trace writer — should run without error.
        result = run_one_question(
            strategy,
            ds,
            sq.descriptor,
            cfg,
            strike=sq.strike,
            decision_trace_writer=None,
        )
        assert result is not None
