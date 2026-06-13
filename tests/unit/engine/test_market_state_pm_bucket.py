from hlanalysis.engine.market_state import MarketState
from hlanalysis.events import QuestionMetaEvent, ProductType, Mechanism


def _bucket_meta():
    keys = [
        "class",
        "underlying",
        "leg_token_ids",
        "priceThresholds",
        "bucketLayout",
        "expiry_ns",
        "series_slug",
        "condition_id",
    ]
    values = [
        "priceBucket",
        "BTC",
        "y80,n80,y90,n90",
        "80000,90000",
        "above_ladder",
        "0",
        "btc-multi-strikes-weekly",
        "evt1",
    ]
    return QuestionMetaEvent(
        venue="polymarket",
        product_type=ProductType.PREDICTION_BINARY,
        mechanism=Mechanism.CLOB,
        symbol="y80",
        exchange_ts=0,
        local_recv_ts=1,
        question_idx=7,
        named_outcome_idxs=[0, 1],
        keys=keys,
        values=values,
    )


def test_pm_bucket_builds_flat_leg_symbols():
    ms = MarketState()
    ms.apply(_bucket_meta())
    q = ms.question(7)
    assert q.klass == "priceBucket"
    assert q.leg_symbols == ("y80", "n80", "y90", "n90")


def test_pm_binary_path_unchanged():
    ms = MarketState()
    keys = ["class", "underlying", "yes_token_id", "no_token_id", "series_slug"]
    values = ["priceBinary", "BTC", "yA", "nA", "btc-up-or-down-daily"]
    ev = QuestionMetaEvent(
        venue="polymarket",
        product_type=ProductType.PREDICTION_BINARY,
        mechanism=Mechanism.CLOB,
        symbol="yA",
        exchange_ts=0,
        local_recv_ts=1,
        question_idx=8,
        named_outcome_idxs=[0, 1],
        keys=keys,
        values=values,
    )
    ms.apply(ev)
    q = ms.question(8)
    assert q.leg_symbols == ("yA", "nA")
