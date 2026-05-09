from __future__ import annotations

import pytest

from hlanalysis.engine.config import (
    AllowlistEntry, GlobalRiskConfig, StrategyConfig,
)
from hlanalysis.engine.market_state import MarketState
from hlanalysis.engine.scanner import Scanner
from hlanalysis.engine.state import StateDAL
from hlanalysis.events import (
    BboEvent, MarkEvent, Mechanism, ProductType, QuestionMetaEvent,
)
from hlanalysis.strategy.late_resolution import (
    LateResolutionConfig, LateResolutionStrategy,
)
from hlanalysis.strategy.types import Action


def _strategy_cfg() -> StrategyConfig:
    entry = AllowlistEntry(
        match={"class": "priceBinary", "underlying": "BTC", "period": "1h"},
        max_position_usd=100, stop_loss_pct=10, tte_min_seconds=60,
        tte_max_seconds=1800, price_extreme_threshold=0.95,
        distance_from_strike_usd_min=200, vol_max=0.5,
    )
    return StrategyConfig(
        name="late_resolution", paper_mode=True,
        allowlist=[entry], blocklist_question_idxs=[],
        defaults=entry,
        **{"global": GlobalRiskConfig(
            max_total_inventory_usd=500, max_concurrent_positions=5,
            daily_loss_cap_usd=200, max_strike_distance_pct=10,
            min_recent_volume_usd=0,  # disable for tests
            stale_data_halt_seconds=5, reconcile_interval_seconds=60,
        )},
    )


def _seed_market(now_ns: int) -> MarketState:
    ms = MarketState()
    # Use a near-future expiry: 10 minutes after now (20231114-2223 for now=1_700_000_000_000_000_000)
    from datetime import datetime, timezone
    expiry_str = datetime.fromtimestamp(
        (now_ns + 10 * 60 * 1_000_000_000) / 1e9, tz=timezone.utc
    ).strftime('%Y%m%d-%H%M')
    ms.apply(QuestionMetaEvent(
        venue="hyperliquid", product_type=ProductType.PREDICTION_BINARY,
        mechanism=Mechanism.CLOB, symbol="qmeta",
        exchange_ts=now_ns - 60_000_000_000, local_recv_ts=now_ns - 60_000_000_000,
        question_idx=42, named_outcome_idxs=[3],
        keys=["class", "underlying", "period", "expiry", "strike"],
        values=["priceBinary", "BTC", "1h", expiry_str, "80000"],
    ))
    for i in range(8):
        ms.apply(MarkEvent(
            venue="hyperliquid", product_type=ProductType.PERP, mechanism=Mechanism.CLOB,
            symbol="BTC", exchange_ts=now_ns - (8 - i) * 1_000_000,
            local_recv_ts=now_ns - (8 - i) * 1_000_000, mark_px=80_300.0 + i * 0.01,
        ))
    ms.apply(BboEvent(
        venue="hyperliquid", product_type=ProductType.PREDICTION_BINARY,
        mechanism=Mechanism.CLOB, symbol="#30",
        exchange_ts=now_ns, local_recv_ts=now_ns,
        bid_px=0.95, bid_sz=10.0, ask_px=0.96, ask_sz=10.0,
    ))
    ms.apply(BboEvent(
        venue="hyperliquid", product_type=ProductType.PREDICTION_BINARY,
        mechanism=Mechanism.CLOB, symbol="#31",
        exchange_ts=now_ns, local_recv_ts=now_ns,
        bid_px=0.04, bid_sz=10.0, ask_px=0.05, ask_sz=10.0,
    ))
    return ms


def test_scanner_emits_enter_for_allowlisted_question(tmp_path):
    now = 1_700_000_000_000_000_000
    ms = _seed_market(now)
    dal = StateDAL(tmp_path / "state.db")
    dal.run_migrations()
    cfg = _strategy_cfg()
    rcfg = LateResolutionConfig(
        tte_min_seconds=60, tte_max_seconds=1800,
        price_extreme_threshold=0.95, distance_from_strike_usd_min=200.0,
        vol_max=0.5, max_position_usd=100.0, stop_loss_pct=10.0,
        max_strike_distance_pct=10.0, min_recent_volume_usd=0.0,
        stale_data_halt_seconds=5,
    )
    scanner = Scanner(
        strategy=LateResolutionStrategy(rcfg),
        cfg=cfg, market_state=ms, dal=dal, kill_switch_path=tmp_path / "halt",
        last_reconcile_ns=now,
    )
    decisions = scanner.scan(now_ns=now)
    assert any(d.decision.action is Action.ENTER for d in decisions)


def test_scanner_skips_blocklisted_question(tmp_path):
    now = 1_700_000_000_000_000_000
    ms = _seed_market(now)
    dal = StateDAL(tmp_path / "state.db")
    dal.run_migrations()
    cfg = _strategy_cfg().model_copy(update={"blocklist_question_idxs": [42]})
    rcfg = LateResolutionConfig(
        tte_min_seconds=60, tte_max_seconds=1800,
        price_extreme_threshold=0.95, distance_from_strike_usd_min=200.0,
        vol_max=0.5, max_position_usd=100.0, stop_loss_pct=10.0,
        max_strike_distance_pct=10.0, min_recent_volume_usd=0.0,
        stale_data_halt_seconds=5,
    )
    scanner = Scanner(
        strategy=LateResolutionStrategy(rcfg),
        cfg=cfg, market_state=ms, dal=dal, kill_switch_path=tmp_path / "halt",
        last_reconcile_ns=now,
    )
    decisions = scanner.scan(now_ns=now)
    enters = [d for d in decisions if d.decision.action is Action.ENTER]
    assert enters == []
