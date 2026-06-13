"""SHR-83 — durable live-engine trade journal.

The journal records ONE row per decision/order, progressively populated across
the decision → send → fill → (reject) lifecycle, plus the slot halt-state at
decision time and the evaluate() inputs (book-at-decision top-N, σ,
recent_returns summary, recent_volume_usd, ref price). It is write-only,
best-effort, and must NEVER raise into the caller (the engine hot path).
"""

from __future__ import annotations

import json

import pytest

from hlanalysis.engine.state import StateDAL
from hlanalysis.engine.trade_journal import HaltSnapshot, TradeJournal
from hlanalysis.strategy.types import BookState, Diagnostic


def _book() -> BookState:
    return BookState(
        symbol="@30",
        bid_px=0.94,
        bid_sz=10.0,
        ask_px=0.95,
        ask_sz=8.0,
        last_trade_ts_ns=5,
        last_l2_ts_ns=6,
        bid_levels=((0.94, 10.0), (0.93, 20.0), (0.92, 30.0), (0.91, 40.0)),
        ask_levels=((0.95, 8.0), (0.96, 16.0), (0.97, 24.0), (0.98, 32.0)),
    )


def _journal(tmp_path) -> TradeJournal:
    dal = StateDAL(tmp_path / "state.db")
    dal.run_migrations()
    return TradeJournal(dal)


def test_record_decision_captures_inputs_book_sigma_and_halt(tmp_path):
    j = _journal(tmp_path)
    halt = HaltSnapshot(
        restart_blocked=False,
        daily_loss_halted=True,
        realized_pnl_today=-12.5,
        daily_loss_cap_usd=200.0,
        reject_breaker_tripped=True,
        stale_reference=False,
    )
    j.record_decision(
        cloid="hla-v1-abc",
        question_idx=42,
        decision_ts_ns=1000,
        action="enter",
        side="buy",
        symbol="@30",
        intended_size=10.0,
        intended_price=0.95,
        book=_book(),
        reference_price=80_300.0,
        recent_volume_usd=5_000.0,
        recent_returns=(0.001, -0.002, 0.0015, -0.0005),
        diagnostics=(Diagnostic("info", "entry", (("vol", "0.1234"),)),),
        halt=halt,
        top_n=3,
    )
    row = j.dal.get_journal_row("hla-v1-abc")
    assert row is not None
    assert row.question_idx == 42
    assert row.decision_ts_ns == 1000
    assert row.action == "enter"
    assert row.side == "buy" and row.symbol == "@30"
    assert row.intended_size == 10.0 and row.intended_price == 0.95
    assert row.reference_price == 80_300.0
    assert row.recent_volume_usd == 5_000.0
    # sigma extracted from the diagnostic's "vol" field.
    assert row.sigma == pytest.approx(0.1234)
    # send/fill/reject not yet populated.
    assert row.send_ts_ns is None
    assert row.fill_ts_ns is None
    assert row.reject_reason is None

    # book snapshot is top-N truncated.
    book = json.loads(row.book_json)
    assert book["bid_px"] == 0.94 and book["ask_px"] == 0.95
    assert len(book["bid_levels"]) == 3
    assert len(book["ask_levels"]) == 3
    assert book["bid_levels"][0] == [0.94, 10.0]

    # recent_returns stored as a summary, not the raw array.
    summ = json.loads(row.returns_summary_json)
    assert summ["n"] == 4
    assert "mean" in summ and "std" in summ and summ["last"] == -0.0005

    halt_d = json.loads(row.halt_json)
    assert halt_d["daily_loss_halted"] is True
    assert halt_d["reject_breaker_tripped"] is True
    assert halt_d["realized_pnl_today"] == -12.5


def test_lifecycle_decision_send_fill(tmp_path):
    j = _journal(tmp_path)
    j.record_decision(
        cloid="hla-v1-1",
        question_idx=1,
        decision_ts_ns=100,
        action="enter",
        side="buy",
        symbol="@30",
        intended_size=5.0,
        intended_price=0.9,
        book=_book(),
        reference_price=80_000.0,
        recent_volume_usd=1_000.0,
    )
    j.record_send(cloid="hla-v1-1", send_ts_ns=150)
    j.record_fill(cloid="hla-v1-1", fill_ts_ns=180, fill_px=0.901, fill_sz=5.0)
    row = j.dal.get_journal_row("hla-v1-1")
    assert row.send_ts_ns == 150
    assert row.fill_ts_ns == 180
    assert row.fill_px == pytest.approx(0.901)
    assert row.fill_sz == 5.0
    assert row.reject_reason is None


def test_lifecycle_decision_send_reject(tmp_path):
    j = _journal(tmp_path)
    j.record_decision(
        cloid="hla-v1-2",
        question_idx=2,
        decision_ts_ns=100,
        action="enter",
        side="buy",
        symbol="@31",
        intended_size=5.0,
        intended_price=0.9,
        book=_book(),
        reference_price=80_000.0,
        recent_volume_usd=1_000.0,
    )
    j.record_send(cloid="hla-v1-2", send_ts_ns=150)
    j.record_reject(cloid="hla-v1-2", reject_reason="insufficient margin")
    row = j.dal.get_journal_row("hla-v1-2")
    assert row.send_ts_ns == 150
    assert row.reject_reason == "insufficient margin"
    assert row.fill_ts_ns is None


def test_updates_to_unknown_cloid_are_noops(tmp_path):
    """A send/fill/reject for a cloid with no decision row must not create a
    partial row or raise — it is simply dropped."""
    j = _journal(tmp_path)
    j.record_send(cloid="ghost", send_ts_ns=1)
    j.record_fill(cloid="ghost", fill_ts_ns=2, fill_px=0.5, fill_sz=1.0)
    j.record_reject(cloid="ghost", reject_reason="x")
    assert j.dal.get_journal_row("ghost") is None


def test_sigma_is_null_when_no_annualized_sigma_in_diagnostics(tmp_path):
    """Exit rows (and any row whose diagnostics carry no vol/sigma field) must
    record sigma=NULL, NOT fall back to the raw returns stdev.  The two scales
    (annualised ≈ 0.4; returns-stdev ≈ 0.0002) are incomparable, so a
    mixed-scale column silently corrupts every consumer that compares rows."""
    j = _journal(tmp_path)
    # Simulate an exit decision: non-empty recent_returns but diagnostics that
    # carry NO 'sigma'/'vol'/'vol_sigma' field (typical for exit triggers).
    j.record_decision(
        cloid="hla-v1-exit-1",
        question_idx=3,
        decision_ts_ns=1,
        action="exit",
        recent_returns=(0.001, -0.002, 0.0015, -0.0005),
        diagnostics=(Diagnostic("warn", "exit_safety_d_below_min", (("safety_d", "0.35"), ("threshold", "0.50"))),),
    )
    row = j.dal.get_journal_row("hla-v1-exit-1")
    # sigma must be NULL — not the raw returns stdev (≈ 0.0013).
    assert row.sigma is None


def test_sigma_is_null_when_no_diagnostics_and_no_returns(tmp_path):
    """Truly empty decision (no diagnostics, no returns) → sigma NULL."""
    j = _journal(tmp_path)
    j.record_decision(
        cloid="hla-v1-3",
        question_idx=3,
        decision_ts_ns=1,
        action="enter",
    )
    row = j.dal.get_journal_row("hla-v1-3")
    assert row.sigma is None


def test_sigma_annualized_from_enter_diagnostics_unchanged(tmp_path):
    """Enter rows with the 'edge' diagnostic block must still record the
    annualised σ — the fix must not regress the happy path."""
    j = _journal(tmp_path)
    j.record_decision(
        cloid="hla-v1-enter-1",
        question_idx=4,
        decision_ts_ns=2,
        action="enter",
        side="buy",
        symbol="@30",
        recent_returns=(0.001, -0.002, 0.0015),
        diagnostics=(
            Diagnostic(
                "info",
                "edge",
                (
                    ("p_model", "0.9200"),
                    ("edge_yes", "0.0350"),
                    ("edge_no", "-1000000000.0000"),
                    ("sigma", "0.4123"),
                    ("tau_yr", "0.000027378"),
                    ("ln_sk", "-0.0012"),
                ),
            ),
        ),
    )
    row = j.dal.get_journal_row("hla-v1-enter-1")
    assert row.sigma == pytest.approx(0.4123)


def test_writes_are_best_effort_and_never_raise(tmp_path, monkeypatch):
    """A DB write failure inside the journal must be swallowed — the journal is
    off the hot path and must never break order submission."""
    j = _journal(tmp_path)

    def _boom(*a, **k):
        raise RuntimeError("disk gone")

    monkeypatch.setattr(j.dal, "add_journal_decision", _boom)
    monkeypatch.setattr(j.dal, "update_journal", _boom)
    # None of these should propagate.
    j.record_decision(cloid="hla-v1-4", question_idx=4, decision_ts_ns=1, action="enter")
    j.record_send(cloid="hla-v1-4", send_ts_ns=2)
    j.record_fill(cloid="hla-v1-4", fill_ts_ns=3, fill_px=0.5, fill_sz=1.0)
    j.record_reject(cloid="hla-v1-4", reject_reason="x")


def test_record_decision_is_insert_once_not_clobber(tmp_path):
    """A second decision row for the same cloid must not overwrite the first
    (cloids are unique per order; a duplicate is a no-op insert)."""
    j = _journal(tmp_path)
    j.record_decision(cloid="hla-v1-5", question_idx=5, decision_ts_ns=1, action="enter", reference_price=1.0)
    j.record_decision(cloid="hla-v1-5", question_idx=5, decision_ts_ns=2, action="exit", reference_price=2.0)
    row = j.dal.get_journal_row("hla-v1-5")
    assert row.decision_ts_ns == 1
    assert row.action == "enter"
