import json
from hlanalysis.adapters.polymarket_normalize import (
    parse_bucket_event,
    parse_gamma_event_to_bucket_question_meta,
)


def _mk(strike, yes, no, cond):
    return {
        "conditionId": cond,
        "groupItemTitle": f"${strike:,.0f}",
        "clobTokenIds": json.dumps([yes, no]),
        "outcomePrices": json.dumps(["0", "0"]),
    }


def _event():
    return {
        "slug": "btc-multi-strikes-weekly-2026-06-12",
        "startDate": "2026-06-08T16:00:00Z",
        "endDate": "2026-06-12T16:00:00Z",
        # deliberately out of strike order to prove sorting
        "markets": [
            _mk(90000, "y90", "n90", "c90"),
            _mk(80000, "y80", "n80", "c80"),
        ],
    }


def test_parse_bucket_event_sorts_by_strike_and_flattens_legs():
    rec = parse_bucket_event(_event())
    assert rec["thresholds"] == [80000.0, 90000.0]
    assert rec["leg_tokens"] == [["y80", "n80"], ["y90", "n90"]]


def test_bucket_question_meta_has_ladder_kv_and_all_legs():
    qm = parse_gamma_event_to_bucket_question_meta(
        _event(),
        series_slug="btc-multi-strikes-weekly",
        local_recv_ts=123,
        underlying="BTC",
    )
    kv = dict(zip(qm.keys, qm.values))
    assert kv["class"] == "priceBucket"
    assert kv["underlying"] == "BTC"
    assert kv["bucketLayout"] == "above_ladder"
    assert kv["priceThresholds"] == "80000,90000"
    # flat canonical order yes0,no0,yes1,no1
    assert kv["leg_token_ids"] == "y80,n80,y90,n90"
    assert kv["series_slug"] == "btc-multi-strikes-weekly"
    # one named outcome per strike
    assert sorted(qm.named_outcome_idxs) == [0, 1]
    # symbol routes WS frames; first leg token
    assert qm.symbol == "y80"


def test_parse_bucket_event_none_when_under_two_legs():
    ev = _event()
    ev["markets"] = ev["markets"][:1]
    assert parse_bucket_event(ev) is None
