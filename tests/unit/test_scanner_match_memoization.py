"""P4: the scanner must not re-run allowlist matching for every question on
every tick. It memoizes the match verdict per question_idx (the match fields —
class/underlying/period/venue/series_slug — are immutable per question, and the
allowlist/blocklist are static per process), so non-tradeable questions are
skipped with a dict lookup instead of a full match each scan. Decisions must be
bit-identical to the un-memoized behaviour.
"""
from __future__ import annotations

from datetime import datetime, timezone

import hlanalysis.engine.scanner as scanner_mod
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


def _cfg() -> StrategyConfig:
    entry = AllowlistEntry(
        match={"class": "priceBinary", "underlying": "BTC", "period": "1h"},
        max_position_usd=100, stop_loss_pct=10, tte_min_seconds=60,
        tte_max_seconds=1800, price_extreme_threshold=0.95,
        distance_from_strike_usd_min=200, vol_max=0.5,
    )
    return StrategyConfig(
        name="late_resolution", paper_mode=True,
        allowlist=[entry], blocklist_question_idxs=[], defaults=entry,
        **{"global": GlobalRiskConfig(
            max_total_inventory_usd=500, max_concurrent_positions=5,
            daily_loss_cap_usd=200, max_strike_distance_pct=10,
            min_recent_volume_usd=0, stale_data_halt_seconds=5,
            reconcile_interval_seconds=60,
        )},
    )


def _seed_two_questions(now_ns: int) -> MarketState:
    """One BTC/1h question (matches the allowlist) + one ETH/1h question
    (does not). Both fully quoted so the matching one would ENTER."""
    ms = MarketState()
    expiry_str = datetime.fromtimestamp(
        (now_ns + 10 * 60 * 1_000_000_000) / 1e9, tz=timezone.utc
    ).strftime('%Y%m%d-%H%M')
    # Matching: BTC
    ms.apply(QuestionMetaEvent(
        venue="hyperliquid", product_type=ProductType.PREDICTION_BINARY,
        mechanism=Mechanism.CLOB, symbol="qmeta",
        exchange_ts=now_ns - 60_000_000_000, local_recv_ts=now_ns - 60_000_000_000,
        question_idx=42, named_outcome_idxs=[3],
        keys=["class", "underlying", "period", "expiry", "strike"],
        values=["priceBinary", "BTC", "1h", expiry_str, "80000"],
    ))
    # Non-matching: ETH (different underlying → allowlist won't match)
    ms.apply(QuestionMetaEvent(
        venue="hyperliquid", product_type=ProductType.PREDICTION_BINARY,
        mechanism=Mechanism.CLOB, symbol="qmeta2",
        exchange_ts=now_ns - 60_000_000_000, local_recv_ts=now_ns - 60_000_000_000,
        question_idx=43, named_outcome_idxs=[5],
        keys=["class", "underlying", "period", "expiry", "strike"],
        values=["priceBinary", "ETH", "1h", expiry_str, "3000"],
    ))
    for i in range(8):
        ts = now_ns - (8 - i) * 60_000_000_000
        ms.apply(MarkEvent(
            venue="hyperliquid", product_type=ProductType.PERP,
            mechanism=Mechanism.CLOB, symbol="BTC",
            exchange_ts=ts, local_recv_ts=ts, mark_px=80_300.0 + i * 0.01,
        ))
    # Books for the matching question's legs (#30/#31) and the ETH legs (#50/#51).
    for sym, bid, ask in [("#30", 0.95, 0.96), ("#31", 0.04, 0.05),
                          ("#50", 0.60, 0.61), ("#51", 0.39, 0.40)]:
        ms.apply(BboEvent(
            venue="hyperliquid", product_type=ProductType.PREDICTION_BINARY,
            mechanism=Mechanism.CLOB, symbol=sym,
            exchange_ts=now_ns, local_recv_ts=now_ns,
            bid_px=bid, bid_sz=10.0, ask_px=ask, ask_sz=10.0,
        ))
    return ms


def _scanner(ms, tmp_path):
    dal = StateDAL(tmp_path / "state.db")
    dal.run_migrations()
    rcfg = LateResolutionConfig(
        tte_min_seconds=60, tte_max_seconds=1800,
        price_extreme_threshold=0.95, distance_from_strike_usd_min=200.0,
        vol_max=0.5, max_position_usd=100.0, stop_loss_pct=10.0,
        max_strike_distance_pct=10.0, min_recent_volume_usd=0.0,
        stale_data_halt_seconds=5,
    )
    return Scanner(
        strategy=LateResolutionStrategy(rcfg), cfg=_cfg(),
        market_state=ms, dal=dal, kill_switch_path=tmp_path / "halt",
        last_reconcile_ns=0,
    )


def test_match_question_memoized_across_ticks(tmp_path, monkeypatch):
    now = 1_700_000_000_000_000_000
    ms = _seed_two_questions(now)
    scanner = _scanner(ms, tmp_path)

    calls: list[int] = []
    real = scanner_mod.match_question

    def _counting(cfg, *, question_idx, fields):
        calls.append(question_idx)
        return real(cfg, question_idx=question_idx, fields=fields)

    monkeypatch.setattr(scanner_mod, "match_question", _counting)

    scanner.scan(now_ns=now)
    scanner.scan(now_ns=now)
    scanner.scan(now_ns=now)

    # Each question is matched exactly ONCE despite three scans (memoized).
    assert sorted(calls) == [42, 43], f"expected one match per qidx, got {calls}"


def test_decisions_unchanged_with_non_tradeable_question(tmp_path):
    now = 1_700_000_000_000_000_000
    ms = _seed_two_questions(now)
    scanner = _scanner(ms, tmp_path)
    decisions = scanner.scan(now_ns=now)
    # The matching BTC question (42) enters; the non-matching ETH question (43)
    # never produces a decision.
    assert any(d.decision.action is Action.ENTER for d in decisions)
    assert all(d.inputs.question.question_idx == 42 for d in decisions)
