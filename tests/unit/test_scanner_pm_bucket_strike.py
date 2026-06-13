"""Regression: the scanner's PM NaN-strike skip must NOT swallow priceBucket
questions (SHR-109 sibling — the bucket-only v31_pm_eth_ms slot saw 0 decisions).

PM up/down (priceBinary) markets get their strike captured asynchronously, so a
NaN strike there means "not priced yet → skip". But PM multi-strike buckets
(priceBucket) have NO single strike — every leg's winning region comes from
`priceThresholds`, so `question.strike` is *always* NaN by construction. The old
blanket `q.venue == polymarket and q.strike is NaN → continue` skipped every
bucket leg, so a bucket-only slot evaluated nothing and emitted no decisions.

These tests pin: a NaN-strike PM bucket reaches the strategy; a NaN-strike PM
binary does not.
"""

from __future__ import annotations

from datetime import datetime, timezone

from hlanalysis.engine.config import (
    AllowlistEntry,
    GlobalRiskConfig,
    StrategyConfig,
)
from hlanalysis.engine.market_state import MarketState
from hlanalysis.engine.scanner import Scanner
from hlanalysis.engine.state import StateDAL
from hlanalysis.events import BboEvent, Mechanism, ProductType, QuestionMetaEvent
from hlanalysis.strategy.types import Action, Decision

_REF = "ETHUSDT_SPOT"
_BUCKET_QIDX = 700001
_BINARY_QIDX = 700002


class _RecordingStrategy:
    """Records every question_idx that reaches evaluate(), always HOLDs."""

    def __init__(self) -> None:
        self.seen: list[int] = []

    def evaluate(self, *, question, **_kw) -> Decision:
        self.seen.append(question.question_idx)
        return Decision(action=Action.HOLD, diagnostics=())


def _expiry(now_ns: int) -> str:
    return datetime.fromtimestamp((now_ns + 3 * 24 * 3600 * 1_000_000_000) / 1e9, tz=timezone.utc).strftime(
        "%Y%m%d-%H%M"
    )


def _seed(now_ns: int) -> MarketState:
    ms = MarketState()
    # ETH spot reference: bbo-sourced so last_mark resolves from the BBO mid.
    ms.set_reference_source(_REF, "bbo")
    ms.apply(
        BboEvent(
            venue="binance",
            product_type=ProductType.SPOT,
            mechanism=Mechanism.CLOB,
            symbol=_REF,
            exchange_ts=now_ns,
            local_recv_ts=now_ns,
            bid_px=3000.0,
            bid_sz=5.0,
            ask_px=3000.2,
            ask_sz=5.0,
        )
    )

    # PM multi-strike bucket (no "strike" key → q.strike is NaN by construction).
    ms.apply(
        QuestionMetaEvent(
            venue="polymarket",
            product_type=ProductType.PREDICTION_BINARY,
            mechanism=Mechanism.CLOB,
            symbol="ETHL0",
            exchange_ts=now_ns,
            local_recv_ts=now_ns,
            question_idx=_BUCKET_QIDX,
            named_outcome_idxs=[0, 1],
            keys=["class", "underlying", "leg_token_ids", "priceThresholds", "bucketLayout", "expiry", "series_slug"],
            values=[
                "priceBucket",
                "ETH",
                "ETHL0,ETHL0N,ETHL1,ETHL1N",
                "2500,3500",
                "above_ladder",
                _expiry(now_ns),
                "ethereum-multi-strikes-weekly",
            ],
        )
    )
    ms.apply(
        BboEvent(
            venue="polymarket",
            product_type=ProductType.PREDICTION_BINARY,
            mechanism=Mechanism.CLOB,
            symbol="ETHL0",
            exchange_ts=now_ns,
            local_recv_ts=now_ns,
            bid_px=0.90,
            bid_sz=50.0,
            ask_px=0.92,
            ask_sz=50.0,
        )
    )

    # PM up/down binary with NO captured strike (strike_ref_ts_ns present, strike
    # absent → NaN). Must still be skipped (cannot be priced yet).
    ms.apply(
        QuestionMetaEvent(
            venue="polymarket",
            product_type=ProductType.PREDICTION_BINARY,
            mechanism=Mechanism.CLOB,
            symbol="ETHYES",
            exchange_ts=now_ns,
            local_recv_ts=now_ns,
            question_idx=_BINARY_QIDX,
            named_outcome_idxs=[0],
            keys=["class", "underlying", "expiry", "yes_token_id", "no_token_id", "series_slug", "strike_ref_ts_ns"],
            values=[
                "priceBinary",
                "ETH",
                _expiry(now_ns),
                "ETHYES",
                "ETHNO",
                "eth-up-or-down-daily",
                str(now_ns - 120_000_000_000),
            ],
        )
    )
    ms.apply(
        BboEvent(
            venue="polymarket",
            product_type=ProductType.PREDICTION_BINARY,
            mechanism=Mechanism.CLOB,
            symbol="ETHYES",
            exchange_ts=now_ns,
            local_recv_ts=now_ns,
            bid_px=0.55,
            bid_sz=50.0,
            ask_px=0.57,
            ask_sz=50.0,
        )
    )
    return ms


def _cfg() -> StrategyConfig:
    bucket = AllowlistEntry(
        match={
            "class": "priceBucket",
            "underlying": "ETH",
            "venue": "polymarket",
            "series_slug": "ethereum-multi-strikes-weekly",
        },
        max_position_usd=50,
        stop_loss_pct=None,
        tte_min_seconds=0,
        tte_max_seconds=604800,
        price_extreme_threshold=0.0,
        distance_from_strike_usd_min=0,
        vol_max=100,
    )
    binary = AllowlistEntry(
        match={
            "class": "priceBinary",
            "underlying": "ETH",
            "venue": "polymarket",
            "series_slug": "eth-up-or-down-daily",
        },
        max_position_usd=50,
        stop_loss_pct=None,
        tte_min_seconds=0,
        tte_max_seconds=604800,
        price_extreme_threshold=0.0,
        distance_from_strike_usd_min=0,
        vol_max=100,
    )
    return StrategyConfig(
        name="theta_harvester",
        paper_mode=True,
        allowlist=[bucket, binary],
        blocklist_question_idxs=[],
        defaults=bucket,
        **{
            "global": GlobalRiskConfig(
                max_total_inventory_usd=500,
                max_concurrent_positions=5,
                daily_loss_cap_usd=200,
                max_strike_distance_pct=100,
                min_recent_volume_usd=0,
                stale_data_halt_seconds=5,
                reconcile_interval_seconds=60,
            )
        },
    )


def test_pm_bucket_with_nan_strike_reaches_strategy(tmp_path):
    now = 1_700_000_000_000_000_000
    ms = _seed(now)
    assert ms.question(_BUCKET_QIDX).strike != ms.question(_BUCKET_QIDX).strike  # NaN
    dal = StateDAL(tmp_path / "state.db")
    dal.run_migrations()
    strat = _RecordingStrategy()
    scanner = Scanner(
        strategy=strat,
        cfg=_cfg(),
        market_state=ms,
        dal=dal,
        kill_switch_path=tmp_path / "halt",
        last_reconcile_ns=now,
        reference_symbol=_REF,
    )
    scanner.scan(now_ns=now)
    assert _BUCKET_QIDX in strat.seen, (
        "PM priceBucket with NaN strike must be evaluated (it is priced off priceThresholds, not question.strike)"
    )


def test_pm_binary_with_nan_strike_still_skipped(tmp_path):
    now = 1_700_000_000_000_000_000
    ms = _seed(now)
    dal = StateDAL(tmp_path / "state.db")
    dal.run_migrations()
    strat = _RecordingStrategy()
    scanner = Scanner(
        strategy=strat,
        cfg=_cfg(),
        market_state=ms,
        dal=dal,
        kill_switch_path=tmp_path / "halt",
        last_reconcile_ns=now,
        reference_symbol=_REF,
    )
    scanner.scan(now_ns=now)
    assert _BINARY_QIDX not in strat.seen, (
        "PM priceBinary with an uncaptured (NaN) strike cannot be priced yet and must still be skipped"
    )
