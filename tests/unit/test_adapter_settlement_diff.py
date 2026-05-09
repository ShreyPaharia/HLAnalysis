"""Unit test for HyperliquidAdapter._detect_polled_settlements.

Verifies that polled outcomeMeta diffs produce SettlementEvent rows for:
  (a) existing question whose settledNamedOutcomes grew, and
  (b) question that disappeared from outcomeMeta.questions[] entirely.
"""

from __future__ import annotations

from hlanalysis.adapters.hyperliquid import HyperliquidAdapter
from hlanalysis.config import Subscription
from hlanalysis.events import Mechanism, ProductType, SettlementEvent


def _adapter() -> HyperliquidAdapter:
    return HyperliquidAdapter()


def _btc_bucket_template() -> Subscription:
    return Subscription(
        venue="hyperliquid",
        product_type=ProductType.PREDICTION_BINARY,
        mechanism=Mechanism.CLOB,
        symbol="*",
        channels=["trades", "book"],
        match={"underlying": "BTC", "class": "priceBucket"},
    )


def _question(qidx: int, named: list[int], settled: list[int]) -> dict:
    return {
        "question": qidx,
        "name": "Recurring",
        "description": "class:priceBucket|underlying:BTC|priceThresholds:77000,82000|expiry:20260509-0600|period:1d",
        "namedOutcomes": named,
        "settledNamedOutcomes": settled,
    }


def test_no_settlement_when_meta_unchanged():
    a = _adapter()
    tmpl = [_btc_bucket_template()]
    payload = {"questions": [_question(1, [12, 13, 14], [])]}
    # Prime snapshot
    a._detect_polled_settlements(tmpl, payload)
    # Same payload again — no settlements
    out = a._detect_polled_settlements(tmpl, payload)
    assert out == []


def test_settlement_when_settledNamedOutcomes_grows():
    a = _adapter()
    tmpl = [_btc_bucket_template()]
    a._detect_polled_settlements(tmpl, {"questions": [_question(1, [12, 13, 14], [])]})
    out = a._detect_polled_settlements(tmpl, {"questions": [_question(1, [12, 13, 14], [13])]})
    assert len(out) == 1
    ev = out[0]
    assert isinstance(ev, SettlementEvent)
    assert ev.symbol == "#130"  # YES leg of outcome 13


def test_settlement_when_question_disappears():
    a = _adapter()
    tmpl = [_btc_bucket_template()]
    # Snapshot question 1 active with no settled outcomes
    a._detect_polled_settlements(tmpl, {"questions": [_question(1, [12, 13, 14], [])]})
    # Now question 1 gone (HL retired post-settle)
    out = a._detect_polled_settlements(tmpl, {"questions": []})
    # Three outcomes were unsettled at disappearance → three settlements
    syms = sorted(ev.symbol for ev in out)
    assert syms == ["#120", "#130", "#140"]


def test_settlement_only_for_matched_template():
    a = _adapter()
    # Template only matches priceBucket; a priceBinary question disappearing
    # should not emit (template filter rejects it).
    tmpl = [_btc_bucket_template()]
    a._detect_polled_settlements(tmpl, {"questions": [
        {
            "question": 99,
            "name": "Other",
            "description": "class:priceBinary|underlying:BTC|targetPrice:80000|expiry:20260509-0600|period:1d",
            "namedOutcomes": [10],
            "settledNamedOutcomes": [],
        },
    ]})
    out = a._detect_polled_settlements(tmpl, {"questions": []})
    assert out == []  # template doesn't match priceBinary


def test_settlement_on_first_poll_does_not_fire():
    """First call primes the snapshot — no diff to compute."""
    a = _adapter()
    tmpl = [_btc_bucket_template()]
    out = a._detect_polled_settlements(
        tmpl, {"questions": [_question(1, [12, 13, 14], [13])]}
    )
    # First call: nothing to diff against, so we log the state but don't emit.
    # Acceptable as "primed".
    assert out == []
