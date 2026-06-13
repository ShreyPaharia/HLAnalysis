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
    ms.mark_question_settled(1)  # settled, old
    # qidx 2 not settled
    n = ms.evict_settled_questions(now_ns=1_000 + 7 * 3600 * 1_000_000_000, retain_after_settle_ns=3600 * 1_000_000_000)
    idxs = {q.question_idx for q in ms.all_questions()}
    assert n == 1
    assert idxs == {2}  # settled+old gone, unsettled kept


def test_evict_keeps_recently_settled():
    ms = MarketState()
    ms.apply(_meta(1, expiry_ns=1_000))
    ms.mark_question_settled(1)
    # now only 1 minute after expiry -> within retain window -> kept
    n = ms.evict_settled_questions(now_ns=1_000 + 60 * 1_000_000_000, retain_after_settle_ns=3600 * 1_000_000_000)
    assert n == 0
    assert {q.question_idx for q in ms.all_questions()} == {1}


def _meta_outcome(idx, expiry_ns, outcome_idx, venue="hyperliquid"):
    """Like _meta but lets the caller pick the named_outcome_idx so each
    question gets distinct leg symbols (#N0 / #N1)."""
    return QuestionMetaEvent(
        question_idx=idx,
        venue=venue,
        product_type=ProductType.PREDICTION_BINARY,
        mechanism=Mechanism.CLOB,
        symbol=f"#test{idx}",
        exchange_ts=0,
        local_recv_ts=0,
        named_outcome_idxs=[outcome_idx],
        keys=["class", "underlying", "expiry_ns"],
        values=["priceBinary", "BTC", str(expiry_ns)],
    )


def test_evict_invalidates_symbol_cache():
    """evict_settled_questions must clear the symbol→question cache so that
    evicted legs no longer appear in the next symbol_to_question_map() call."""
    ms = MarketState()
    # outcome_idx=0 → legs #00, #01; outcome_idx=1 → legs #10, #11
    ms.apply(_meta_outcome(1, expiry_ns=1_000, outcome_idx=0))  # settled+old
    ms.apply(_meta_outcome(2, expiry_ns=10_000, outcome_idx=1))  # live

    # outcome_idx=0 → f"#{10*0+s}" → "#0" (YES) and "#1" (NO)
    # outcome_idx=1 → f"#{10*1+s}" → "#10" (YES) and "#11" (NO)
    evicted_legs = ("#0", "#1")
    live_legs = ("#10", "#11")

    # Prime the cache before eviction.
    cache_before = ms.symbol_to_question_map()
    assert all(sym in cache_before for sym in evicted_legs)
    assert ms._sym_to_q_cache is not None  # cache is warm

    ms.mark_question_settled(1)
    n = ms.evict_settled_questions(
        now_ns=1_000 + 7 * 3600 * 1_000_000_000,
        retain_after_settle_ns=3600 * 1_000_000_000,
    )
    assert n == 1

    # After eviction the cache must have been rebuilt; evicted legs gone.
    cache_after = ms.symbol_to_question_map()
    assert all(sym not in cache_after for sym in evicted_legs)
    # Live question's legs are still present.
    assert all(sym in cache_after for sym in live_legs)


def test_evict_skips_zero_expiry_settled():
    """A settled question with expiry_ns == 0 must NOT be evicted — the guard
    `q.expiry_ns` (falsy) is intentional and documents the contract."""
    ms = MarketState()
    # expiry_ns=0 → the fallback int(...) in _update_question also yields 0
    ms.apply(_meta_outcome(1, expiry_ns=0, outcome_idx=0))
    ms.mark_question_settled(1)

    # Any eviction window — the zero-expiry guard fires first.
    n = ms.evict_settled_questions(
        now_ns=999 * 3600 * 1_000_000_000,
        retain_after_settle_ns=0,
    )
    assert n == 0
    assert {q.question_idx for q in ms.all_questions()} == {1}
