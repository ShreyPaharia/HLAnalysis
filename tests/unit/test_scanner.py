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
    # 2026-05-21: MarketState now buckets marks to 1m windows. Space
    # timestamps 60s apart so each MarkEvent populates its own bucket.
    for i in range(8):
        ts = now_ns - (8 - i) * 60_000_000_000
        ms.apply(MarkEvent(
            venue="hyperliquid", product_type=ProductType.PERP, mechanism=Mechanism.CLOB,
            symbol="BTC", exchange_ts=ts,
            local_recv_ts=ts, mark_px=80_300.0 + i * 0.01,
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


# --- Daily PnL-window boundary (06:00 UTC for HL HIP-4) ---

from hlanalysis.engine.scanner import Scanner as _Scanner
import datetime as _dt


def test_daily_window_start_at_midnight_utc_is_today_when_after_midnight():
    # 2026-05-19 12:34:56 UTC → window started at 2026-05-19 00:00:00 UTC.
    now_dt = _dt.datetime(2026, 5, 19, 12, 34, 56, tzinfo=_dt.timezone.utc)
    now_ns = int(now_dt.timestamp() * 1_000_000_000)
    start_ns = _Scanner._daily_window_start_ns(now_ns, hour=0)
    expected = _dt.datetime(2026, 5, 19, 0, 0, 0, tzinfo=_dt.timezone.utc)
    assert start_ns == int(expected.timestamp() * 1_000_000_000)


def test_daily_window_start_at_06_utc_rolls_back_when_before_boundary():
    # 2026-05-19 03:00:00 UTC with hour=6 → window started at 2026-05-18 06:00:00 UTC,
    # not at 2026-05-19 06:00:00 UTC (which is in the FUTURE relative to now).
    now_dt = _dt.datetime(2026, 5, 19, 3, 0, 0, tzinfo=_dt.timezone.utc)
    now_ns = int(now_dt.timestamp() * 1_000_000_000)
    start_ns = _Scanner._daily_window_start_ns(now_ns, hour=6)
    expected = _dt.datetime(2026, 5, 18, 6, 0, 0, tzinfo=_dt.timezone.utc)
    assert start_ns == int(expected.timestamp() * 1_000_000_000)


def test_daily_window_start_at_06_utc_uses_today_when_past_boundary():
    # 2026-05-19 07:00:00 UTC with hour=6 → window started at 2026-05-19 06:00:00 UTC.
    # The HIP-4 settlement happened at 06:00:00–06:00:06 today; PnL from
    # earlier 24h closes is now in the previous window.
    now_dt = _dt.datetime(2026, 5, 19, 7, 0, 0, tzinfo=_dt.timezone.utc)
    now_ns = int(now_dt.timestamp() * 1_000_000_000)
    start_ns = _Scanner._daily_window_start_ns(now_ns, hour=6)
    expected = _dt.datetime(2026, 5, 19, 6, 0, 0, tzinfo=_dt.timezone.utc)
    assert start_ns == int(expected.timestamp() * 1_000_000_000)


def test_legacy_utc_midnight_helper_still_works():
    # Backward-compatibility: existing callers (tests, older code) that call
    # _utc_midnight_ns(now) must still get the legacy UTC-midnight behavior.
    now_dt = _dt.datetime(2026, 5, 19, 12, 34, 56, tzinfo=_dt.timezone.utc)
    now_ns = int(now_dt.timestamp() * 1_000_000_000)
    legacy = _Scanner._utc_midnight_ns(now_ns)
    new = _Scanner._daily_window_start_ns(now_ns, hour=0)
    assert legacy == new


# --- Gate-decision log: chosen-leg book snapshot for bucket markets ---


def test_gate_log_snapshot_uses_chosen_leg_book_when_diagnosed(tmp_path):
    """For bucket markets, the strategy can route to any one of N YES legs.
    The gate-decision log must snapshot THAT leg's book (not an arbitrary
    first leg), so operators can correlate the price columns with edge_yes
    without doing inverse arithmetic.
    """
    import json
    from hlanalysis.strategy.types import (
        Action, BookState, Decision, Diagnostic, QuestionView,
    )

    now = 1_700_000_000_000_000_000
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
    log_path = tmp_path / "gate_decisions.jsonl"
    scanner = Scanner(
        strategy=LateResolutionStrategy(rcfg),
        cfg=cfg, market_state=MarketState(), dal=dal,
        kill_switch_path=tmp_path / "halt",
        last_reconcile_ns=now,
        gate_log_path=log_path,
    )

    bucket_q = QuestionView(
        question_idx=14, yes_symbol="", no_symbol="", strike=0.0,
        expiry_ns=now + 3600 * 1_000_000_000,
        underlying="BTC", klass="priceBucket", period="1d",
        leg_symbols=("#700", "#701", "#780", "#781"),
    )
    # First leg is a long-shot (price 0.01); chosen leg #780 is the favorite (0.95).
    books = {
        "#700": BookState(symbol="#700", bid_px=0.00301, bid_sz=10.0,
                          ask_px=0.01, ask_sz=10.0,
                          last_trade_ts_ns=now, last_l2_ts_ns=now),
        "#780": BookState(symbol="#780", bid_px=0.94, bid_sz=10.0,
                          ask_px=0.95, ask_sz=10.0,
                          last_trade_ts_ns=now, last_l2_ts_ns=now),
    }
    decision = Decision(action=Action.HOLD, diagnostics=(
        Diagnostic("info", "edge", (
            ("p_model", "0.7847"), ("edge_yes", "-0.1706"),
            ("chosen_leg", "#780"),
        )),
    ))

    scanner._maybe_log_gate_transition(
        question=bucket_q, decision=decision, books=books, now_ns=now,
    )
    row = json.loads(log_path.read_text().strip())
    # The chosen leg's book is what should appear in the snapshot columns.
    assert row["ask_px"] == 0.95, f"ask_px should be chosen leg's ask, got {row['ask_px']}"
    assert row["bid_px"] == 0.94
    assert row["reason"] == "edge"


def test_gate_log_snapshot_falls_back_to_first_leg_when_no_chosen_leg(tmp_path):
    """When the diagnostic doesn't carry chosen_leg (binary path, or other
    reasons like tte_out_of_window), preserve the legacy first-leg snapshot."""
    import json
    from hlanalysis.strategy.types import (
        Action, BookState, Decision, Diagnostic, QuestionView,
    )

    now = 1_700_000_000_000_000_000
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
    log_path = tmp_path / "gate_decisions.jsonl"
    scanner = Scanner(
        strategy=LateResolutionStrategy(rcfg),
        cfg=cfg, market_state=MarketState(), dal=dal,
        kill_switch_path=tmp_path / "halt",
        last_reconcile_ns=now,
        gate_log_path=log_path,
    )

    q = QuestionView(
        question_idx=42, yes_symbol="@30", no_symbol="@31", strike=80_000.0,
        expiry_ns=now + 600 * 1_000_000_000,
        underlying="BTC", klass="priceBinary", period="1h",
    )
    books = {
        "@30": BookState(symbol="@30", bid_px=0.95, bid_sz=10.0,
                        ask_px=0.96, ask_sz=10.0,
                        last_trade_ts_ns=now, last_l2_ts_ns=now),
    }
    decision = Decision(action=Action.HOLD, diagnostics=(
        Diagnostic("info", "tte_out_of_window", (("tte_s", "8000"),)),
    ))
    scanner._maybe_log_gate_transition(
        question=q, decision=decision, books=books, now_ns=now,
    )
    row = json.loads(log_path.read_text().strip())
    assert row["ask_px"] == 0.96
    assert row["reason"] == "tte_out_of_window"
