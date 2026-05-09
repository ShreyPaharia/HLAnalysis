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
    # Seed marks (low-vol)
    for i in range(32):
        yield MarkEvent(
            venue="hyperliquid", product_type=ProductType.PERP, mechanism=Mechanism.CLOB,
            symbol="BTC", exchange_ts=now_ns - (32 - i) * 1_000_000,
            local_recv_ts=now_ns - (32 - i) * 1_000_000,
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
