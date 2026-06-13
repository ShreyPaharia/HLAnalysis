from __future__ import annotations

from datetime import datetime, timezone

from hlanalysis.strategy.render import (
    outcome_description,
    question_description,
    settlement_pnl_usd,
)
from hlanalysis.strategy.types import QuestionView


_EXPIRY_NS = int(datetime(2026, 5, 9, 6, 0, tzinfo=timezone.utc).timestamp() * 1e9)


def _binary() -> QuestionView:
    return QuestionView(
        question_idx=1_000_010,
        yes_symbol="#100",
        no_symbol="#101",
        strike=79583.0,
        expiry_ns=_EXPIRY_NS,
        underlying="BTC",
        klass="priceBinary",
        period="1d",
        leg_symbols=("#100", "#101"),
        kv=(("class", "priceBinary"), ("underlying", "BTC"), ("targetPrice", "79583"), ("expiry", "20260509-0600")),
    )


def _bucket() -> QuestionView:
    return QuestionView(
        question_idx=1,
        yes_symbol="#120",
        no_symbol="#121",
        strike=float("nan"),
        expiry_ns=_EXPIRY_NS,
        underlying="BTC",
        klass="priceBucket",
        period="1d",
        leg_symbols=("#120", "#121", "#130", "#131", "#140", "#141"),
        kv=(
            ("class", "priceBucket"),
            ("underlying", "BTC"),
            ("priceThresholds", "77991,81174"),
            ("expiry", "20260509-0600"),
        ),
    )


def test_priceBinary_question_and_legs():
    qv = _binary()
    assert question_description(qv) == "BTC > $79,583 by 2026-05-09 06:00 UTC"
    assert outcome_description(qv, "#100") == "YES (BTC > $79,583)"
    assert outcome_description(qv, "#101") == "NO (BTC ≤ $79,583)"


def test_priceBucket_question_and_each_leg():
    qv = _bucket()
    assert question_description(qv) == "BTC bucketed by $77,991 / $81,174 by 2026-05-09 06:00 UTC"
    assert outcome_description(qv, "#120") == "YES (BTC < $77,991)"
    assert outcome_description(qv, "#121") == "NO (BTC ≥ $77,991)"
    assert outcome_description(qv, "#130") == "YES ($77,991 ≤ BTC < $81,174)"
    assert outcome_description(qv, "#131") == "NO (BTC NOT in [$77,991, $81,174))"
    assert outcome_description(qv, "#140") == "YES (BTC > $81,174)"
    assert outcome_description(qv, "#141") == "NO (BTC ≤ $81,174)"


def test_unknown_symbol_falls_back():
    qv = _binary()
    assert "#999" in outcome_description(qv, "#999")


def _pm_binary_no_strike() -> QuestionView:
    # Mirrors what the PM normalize emits for daily up/down markets (no
    # static strike in metadata, only a `question_name` label).
    return QuestionView(
        question_idx=42,
        yes_symbol="tok-yes",
        no_symbol="tok-no",
        strike=float("nan"),
        expiry_ns=_EXPIRY_NS,
        underlying="BTC",
        klass="priceBinary",
        period="",
        name="Will BTC go up or down on May 8?",
        leg_symbols=("tok-yes", "tok-no"),
        kv=(
            ("class", "priceBinary"),
            ("underlying", "BTC"),
            ("expiry", "20260509-0600"),
            ("question_name", "Will BTC go up or down on May 8?"),
        ),
    )


def test_priceBinary_pm_no_strike_uses_question_name():
    qv = _pm_binary_no_strike()
    out = question_description(qv)
    assert "Will BTC go up or down on May 8?" in out
    assert "2026-05-09 06:00 UTC" in out
    # No empty "$" rendering — the old bug was "BTC > $ by ".
    assert "BTC > $ by" not in out
    assert "$ by " not in out


def test_settlement_pnl_yes_wins_using_settled_symbol():
    qv = QuestionView(
        question_idx=1,
        yes_symbol="#100",
        no_symbol="#101",
        strike=79583.0,
        expiry_ns=_EXPIRY_NS,
        underlying="BTC",
        klass="priceBinary",
        period="1d",
        settled=True,
        settled_side="yes",
        settled_symbol="#100",
        leg_symbols=("#100", "#101"),
    )
    # Held the winner: payout 1.0 per share, entered at 0.40 → +0.60/share.
    assert settlement_pnl_usd(qv, "#100", qty=500.0, avg_entry=0.40) == 300.0
    # Held the loser: payout 0.0 per share, entered at 0.60 → -0.60/share.
    assert settlement_pnl_usd(qv, "#101", qty=500.0, avg_entry=0.60) == -300.0


def test_settlement_pnl_unknown_winner_falls_back_to_prior_realized():
    qv = QuestionView(
        question_idx=1,
        yes_symbol="#100",
        no_symbol="#101",
        strike=79583.0,
        expiry_ns=_EXPIRY_NS,
        underlying="BTC",
        klass="priceBinary",
        period="1d",
        settled=True,
        settled_side=None,
        settled_symbol="",
        leg_symbols=("#100", "#101"),
    )
    assert (
        settlement_pnl_usd(
            qv,
            "#100",
            qty=500.0,
            avg_entry=0.40,
            prior_realized=12.5,
        )
        == 12.5
    )


def test_settlement_pnl_bucket_uses_exact_winning_leg():
    # Bucket: leg layout = (yes_o0, no_o0, yes_o1, no_o1, yes_o2, no_o2).
    # Outcome 1 won → SettlementEvent stamps settled_symbol="#10" (YES of o1).
    qv = QuestionView(
        question_idx=2,
        yes_symbol="#0",
        no_symbol="#1",
        strike=float("nan"),
        expiry_ns=_EXPIRY_NS,
        underlying="BTC",
        klass="priceBucket",
        period="1d",
        settled=True,
        settled_side="yes",
        settled_symbol="#10",
        leg_symbols=("#0", "#1", "#10", "#11", "#20", "#21"),
    )
    # Held YES of o1 → won.
    assert settlement_pnl_usd(qv, "#10", qty=100.0, avg_entry=0.30) == 70.0
    # Held YES of o0 → lost (different outcome even though side is "yes").
    assert settlement_pnl_usd(qv, "#0", qty=100.0, avg_entry=0.30) == -30.0
