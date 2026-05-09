from __future__ import annotations

from datetime import datetime, timezone

from hlanalysis.strategy.render import outcome_description, question_description
from hlanalysis.strategy.types import QuestionView


_EXPIRY_NS = int(datetime(2026, 5, 9, 6, 0, tzinfo=timezone.utc).timestamp() * 1e9)


def _binary() -> QuestionView:
    return QuestionView(
        question_idx=1_000_010, yes_symbol="#100", no_symbol="#101",
        strike=79583.0, expiry_ns=_EXPIRY_NS,
        underlying="BTC", klass="priceBinary", period="1d",
        leg_symbols=("#100", "#101"),
        kv=(("class", "priceBinary"), ("underlying", "BTC"),
            ("targetPrice", "79583"), ("expiry", "20260509-0600")),
    )


def _bucket() -> QuestionView:
    return QuestionView(
        question_idx=1, yes_symbol="#120", no_symbol="#121",
        strike=float("nan"), expiry_ns=_EXPIRY_NS,
        underlying="BTC", klass="priceBucket", period="1d",
        leg_symbols=("#120","#121","#130","#131","#140","#141"),
        kv=(("class","priceBucket"),("underlying","BTC"),
            ("priceThresholds","77991,81174"),("expiry","20260509-0600")),
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
