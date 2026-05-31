# tests/unit/test_replay_smoke.py
from __future__ import annotations

from hlanalysis.engine.replay import ReplayRunner
from hlanalysis.events import (
    BboEvent, MarkEvent, Mechanism, ProductType, QuestionMetaEvent,
)
from hlanalysis.strategy.late_resolution import (
    LateResolutionConfig, LateResolutionStrategy,
)
from hlanalysis.strategy.types import Action


def _events(now_ns: int):
    yield QuestionMetaEvent(
        venue="hyperliquid", product_type=ProductType.PREDICTION_BINARY,
        mechanism=Mechanism.CLOB, symbol="qmeta",
        exchange_ts=now_ns - 60_000_000_000, local_recv_ts=now_ns - 60_000_000_000,
        question_idx=42, named_outcome_idxs=[3],
        keys=["class", "underlying", "period", "expiry", "strike"],
        # Expiry 10 min after now_ns=1_700_000_000_000_000_000 (2023-11-14 22:13 UTC)
        values=["priceBinary", "BTC", "1h", "20231114-2223", "80000"],
    )
    # Seed marks (low-vol). 2026-05-21: MarketState now buckets marks to 1m
    # windows, so space these 60s apart.
    for i in range(32):
        ts = now_ns - (32 - i) * 60_000_000_000
        yield MarkEvent(
            venue="hyperliquid", product_type=ProductType.PERP, mechanism=Mechanism.CLOB,
            symbol="BTC", exchange_ts=ts, local_recv_ts=ts,
            mark_px=80_300.0 + i * 0.1,
        )
    yield BboEvent(
        venue="hyperliquid", product_type=ProductType.PREDICTION_BINARY,
        mechanism=Mechanism.CLOB, symbol="#30",
        exchange_ts=now_ns, local_recv_ts=now_ns,
        bid_px=0.95, bid_sz=10.0, ask_px=0.96, ask_sz=10.0,
    )
    yield BboEvent(
        venue="hyperliquid", product_type=ProductType.PREDICTION_BINARY,
        mechanism=Mechanism.CLOB, symbol="#31",
        exchange_ts=now_ns, local_recv_ts=now_ns,
        bid_px=0.04, bid_sz=10.0, ask_px=0.05, ask_sz=10.0,
    )


def test_replay_runs_strategy_and_emits_enter_decision_for_seeded_setup():
    cfg = LateResolutionConfig(
        tte_min_seconds=60, tte_max_seconds=1800,
        price_extreme_threshold=0.95, distance_from_strike_usd_min=200.0,
        vol_max=0.5, max_position_usd=100.0, stop_loss_pct=10.0,
        max_strike_distance_pct=10.0, min_recent_volume_usd=0.0,  # disable for the smoke test
        stale_data_halt_seconds=5,
    )
    runner = ReplayRunner(strategy=LateResolutionStrategy(cfg))
    # Override: late-resolution requires recent_volume_usd ≥ min; we disabled the min
    decisions = list(runner.run_iter(_events(1_700_000_000_000_000_000)))
    enters = [d for d in decisions if d.action is Action.ENTER]
    assert enters, f"expected at least one ENTER decision, got {[d.action for d in decisions]}"


def _bbo_sourced_events(now_ns: int):
    """A PM-style replay where σ is sourced from the BTCUSDT BBO mid rather
    than a MarkEvent."""
    # A real PM question carries the ERC-1155 CLOB token ids; the engine uses
    # those as the leg symbols (PM WS book frames are keyed by them too).
    yield QuestionMetaEvent(
        venue="polymarket", product_type=ProductType.PREDICTION_BINARY,
        mechanism=Mechanism.CLOB, symbol="YES_TOKEN",
        exchange_ts=now_ns - 60_000_000_000, local_recv_ts=now_ns - 60_000_000_000,
        question_idx=42, named_outcome_idxs=[0, 1],
        keys=["class", "underlying", "period", "expiry", "strike",
              "yes_token_id", "no_token_id"],
        values=["priceBinary", "BTC", "1h", "20231114-2223", "80000",
                "YES_TOKEN", "NO_TOKEN"],
    )
    # Reference σ feed: dense BTCUSDT BBO ticks (no MarkEvent).
    for i in range(40):
        ts = now_ns - (40 - i) * 60_000_000_000
        bid = 80_300.0 + i * 0.1
        yield BboEvent(
            venue="binance", product_type=ProductType.PERP,
            mechanism=Mechanism.CLOB, symbol="BTCUSDT",
            exchange_ts=ts, local_recv_ts=ts,
            bid_px=bid, bid_sz=5.0, ask_px=bid + 2.0, ask_sz=5.0,
        )
    yield BboEvent(
        venue="polymarket", product_type=ProductType.PREDICTION_BINARY,
        mechanism=Mechanism.CLOB, symbol="YES_TOKEN",
        exchange_ts=now_ns, local_recv_ts=now_ns,
        bid_px=0.95, bid_sz=10.0, ask_px=0.96, ask_sz=10.0,
    )
    yield BboEvent(
        venue="polymarket", product_type=ProductType.PREDICTION_BINARY,
        mechanism=Mechanism.CLOB, symbol="NO_TOKEN",
        exchange_ts=now_ns, local_recv_ts=now_ns,
        bid_px=0.04, bid_sz=10.0, ask_px=0.05, ask_sz=10.0,
    )


def test_replay_bbo_sourced_reference_drives_sigma_from_mid():
    cfg = LateResolutionConfig(
        tte_min_seconds=60, tte_max_seconds=1800,
        price_extreme_threshold=0.95, distance_from_strike_usd_min=200.0,
        vol_max=0.5, max_position_usd=100.0, stop_loss_pct=10.0,
        max_strike_distance_pct=10.0, min_recent_volume_usd=0.0,
        stale_data_halt_seconds=5,
    )
    now = 1_700_000_000_000_000_000
    runner = ReplayRunner(
        strategy=LateResolutionStrategy(cfg),
        reference_symbol="BTCUSDT",
        reference_sigma_source="bbo",
    )
    decisions = list(runner.run_iter(_bbo_sourced_events(now)))
    enters = [d for d in decisions if d.action is Action.ENTER]
    assert enters, "expected an ENTER driven by bbo-sourced σ"
    # last reference price comes from the final BBO mid, not any mark.
    last_bid = 80_300.0 + 39 * 0.1
    assert runner._market.last_mark("BTCUSDT") == last_bid + 1.0  # mid of [bid, bid+2]
    assert runner._market.recent_hl_bars("BTCUSDT", n=10) != ()
