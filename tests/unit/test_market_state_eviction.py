from hlanalysis.engine.market_state import MarketState
from hlanalysis.events import Mechanism, ProductType, QuestionMetaEvent


def _meta(idx, expiry_ns, venue="hyperliquid"):
    return QuestionMetaEvent(
        question_idx=idx,
        venue=venue,
        product_type=ProductType.PREDICTION_BINARY,
        mechanism=Mechanism.CLOB,
        symbol=f"#test{idx}",
        exchange_ts=0,
        local_recv_ts=0,
        named_outcome_idxs=[0],
        keys=["class", "underlying", "expiry_ns"],
        values=["priceBinary", "BTC", str(expiry_ns)],
    )


def test_evict_settled_questions_removes_old_settled_only():
    ms = MarketState()
    ms.apply(_meta(1, expiry_ns=1_000))
    ms.apply(_meta(2, expiry_ns=10_000))
    ms.mark_question_settled(1)                 # settled, old
    # qidx 2 not settled
    n = ms.evict_settled_questions(now_ns=1_000 + 7 * 3600 * 1_000_000_000,
                                   retain_after_settle_ns=3600 * 1_000_000_000)
    idxs = {q.question_idx for q in ms.all_questions()}
    assert n == 1
    assert idxs == {2}                          # settled+old gone, unsettled kept


def test_evict_keeps_recently_settled():
    ms = MarketState()
    ms.apply(_meta(1, expiry_ns=1_000))
    ms.mark_question_settled(1)
    # now only 1 minute after expiry -> within retain window -> kept
    n = ms.evict_settled_questions(now_ns=1_000 + 60 * 1_000_000_000,
                                   retain_after_settle_ns=3600 * 1_000_000_000)
    assert n == 0
    assert {q.question_idx for q in ms.all_questions()} == {1}
