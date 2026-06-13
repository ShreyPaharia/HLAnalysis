from __future__ import annotations

import pytest

from hlanalysis.engine.config import (
    AllowlistEntry,
    GlobalRiskConfig,
    StrategyConfig,
)
from hlanalysis.engine.market_state import MarketState
from hlanalysis.engine.scanner import Scanner
from hlanalysis.engine.state import StateDAL
from hlanalysis.events import (
    BboEvent,
    MarkEvent,
    Mechanism,
    ProductType,
    QuestionMetaEvent,
)
from hlanalysis.strategy.late_resolution import (
    LateResolutionConfig,
    LateResolutionStrategy,
)
from hlanalysis.strategy.types import Action


def _strategy_cfg() -> StrategyConfig:
    entry = AllowlistEntry(
        match={"class": "priceBinary", "underlying": "BTC", "period": "1h"},
        max_position_usd=100,
        stop_loss_pct=10,
        tte_min_seconds=60,
        tte_max_seconds=1800,
        price_extreme_threshold=0.95,
        distance_from_strike_usd_min=200,
        vol_max=0.5,
    )
    return StrategyConfig(
        name="late_resolution",
        paper_mode=True,
        allowlist=[entry],
        blocklist_question_idxs=[],
        defaults=entry,
        **{
            "global": GlobalRiskConfig(
                max_total_inventory_usd=500,
                max_concurrent_positions=5,
                daily_loss_cap_usd=200,
                max_strike_distance_pct=10,
                min_recent_volume_usd=0,  # disable for tests
                stale_data_halt_seconds=5,
                reconcile_interval_seconds=60,
            )
        },
    )


def _seed_market(now_ns: int) -> MarketState:
    ms = MarketState()
    # Use a near-future expiry: 10 minutes after now (20231114-2223 for now=1_700_000_000_000_000_000)
    from datetime import datetime, timezone

    expiry_str = datetime.fromtimestamp((now_ns + 10 * 60 * 1_000_000_000) / 1e9, tz=timezone.utc).strftime(
        "%Y%m%d-%H%M"
    )
    ms.apply(
        QuestionMetaEvent(
            venue="hyperliquid",
            product_type=ProductType.PREDICTION_BINARY,
            mechanism=Mechanism.CLOB,
            symbol="qmeta",
            exchange_ts=now_ns - 60_000_000_000,
            local_recv_ts=now_ns - 60_000_000_000,
            question_idx=42,
            named_outcome_idxs=[3],
            keys=["class", "underlying", "period", "expiry", "strike"],
            values=["priceBinary", "BTC", "1h", expiry_str, "80000"],
        )
    )
    # 2026-05-21: MarketState now buckets marks to 1m windows. Space
    # timestamps 60s apart so each MarkEvent populates its own bucket.
    for i in range(8):
        ts = now_ns - (8 - i) * 60_000_000_000
        ms.apply(
            MarkEvent(
                venue="hyperliquid",
                product_type=ProductType.PERP,
                mechanism=Mechanism.CLOB,
                symbol="BTC",
                exchange_ts=ts,
                local_recv_ts=ts,
                mark_px=80_300.0 + i * 0.01,
            )
        )
    ms.apply(
        BboEvent(
            venue="hyperliquid",
            product_type=ProductType.PREDICTION_BINARY,
            mechanism=Mechanism.CLOB,
            symbol="#30",
            exchange_ts=now_ns,
            local_recv_ts=now_ns,
            bid_px=0.95,
            bid_sz=10.0,
            ask_px=0.96,
            ask_sz=10.0,
        )
    )
    ms.apply(
        BboEvent(
            venue="hyperliquid",
            product_type=ProductType.PREDICTION_BINARY,
            mechanism=Mechanism.CLOB,
            symbol="#31",
            exchange_ts=now_ns,
            local_recv_ts=now_ns,
            bid_px=0.04,
            bid_sz=10.0,
            ask_px=0.05,
            ask_sz=10.0,
        )
    )
    return ms


def test_scanner_emits_enter_for_allowlisted_question(tmp_path):
    now = 1_700_000_000_000_000_000
    ms = _seed_market(now)
    dal = StateDAL(tmp_path / "state.db")
    dal.run_migrations()
    cfg = _strategy_cfg()
    rcfg = LateResolutionConfig(
        tte_min_seconds=60,
        tte_max_seconds=1800,
        price_extreme_threshold=0.95,
        distance_from_strike_usd_min=200.0,
        vol_max=0.5,
        max_position_usd=100.0,
        stop_loss_pct=10.0,
        max_strike_distance_pct=10.0,
        min_recent_volume_usd=0.0,
        stale_data_halt_seconds=5,
    )
    scanner = Scanner(
        strategy=LateResolutionStrategy(rcfg),
        cfg=cfg,
        market_state=ms,
        dal=dal,
        kill_switch_path=tmp_path / "halt",
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
        tte_min_seconds=60,
        tte_max_seconds=1800,
        price_extreme_threshold=0.95,
        distance_from_strike_usd_min=200.0,
        vol_max=0.5,
        max_position_usd=100.0,
        stop_loss_pct=10.0,
        max_strike_distance_pct=10.0,
        min_recent_volume_usd=0.0,
        stale_data_halt_seconds=5,
    )
    scanner = Scanner(
        strategy=LateResolutionStrategy(rcfg),
        cfg=cfg,
        market_state=ms,
        dal=dal,
        kill_switch_path=tmp_path / "halt",
        last_reconcile_ns=now,
    )
    decisions = scanner.scan(now_ns=now)
    enters = [d for d in decisions if d.decision.action is Action.ENTER]
    assert enters == []


def _cfg_with_match(match: dict) -> StrategyConfig:
    cfg = _strategy_cfg()
    entry = cfg.allowlist[0].model_copy(update={"match": match})
    return cfg.model_copy(update={"allowlist": [entry]})


def _scanner_for(cfg: StrategyConfig, ms: MarketState, tmp_path, now: int) -> Scanner:
    dal = StateDAL(tmp_path / "state.db")
    dal.run_migrations()
    rcfg = LateResolutionConfig(
        tte_min_seconds=60,
        tte_max_seconds=1800,
        price_extreme_threshold=0.95,
        distance_from_strike_usd_min=200.0,
        vol_max=0.5,
        max_position_usd=100.0,
        stop_loss_pct=10.0,
        max_strike_distance_pct=10.0,
        min_recent_volume_usd=0.0,
        stale_data_halt_seconds=5,
    )
    return Scanner(
        strategy=LateResolutionStrategy(rcfg),
        cfg=cfg,
        market_state=ms,
        dal=dal,
        kill_switch_path=tmp_path / "halt",
        last_reconcile_ns=now,
    )


def test_scanner_matches_when_allowlist_scopes_to_question_venue(tmp_path):
    # The seeded question is a hyperliquid one; an allowlist that pins
    # venue=hyperliquid must still match it (venue is exposed to match_question).
    now = 1_700_000_000_000_000_000
    ms = _seed_market(now)
    cfg = _cfg_with_match(
        {"class": "priceBinary", "underlying": "BTC", "period": "1h", "venue": "hyperliquid"},
    )
    scanner = _scanner_for(cfg, ms, tmp_path, now)
    decisions = scanner.scan(now_ns=now)
    assert any(d.decision.action is Action.ENTER for d in decisions)


def test_scanner_skips_question_from_other_venue(tmp_path):
    # A Polymarket-scoped allowlist (venue + series_slug) must NOT match the
    # seeded hyperliquid question — this is the live prod bug where PM slots
    # grabbed HL questions and shipped HL leg symbols as PM token ids.
    now = 1_700_000_000_000_000_000
    ms = _seed_market(now)
    cfg = _cfg_with_match(
        {"class": "priceBinary", "underlying": "BTC", "venue": "polymarket", "series_slug": "btc-up-or-down-daily"},
    )
    scanner = _scanner_for(cfg, ms, tmp_path, now)
    decisions = scanner.scan(now_ns=now)
    assert all(d.decision.action is not Action.ENTER for d in decisions)


def _seed_pm_updown(now_ns: int, *, strike_ref_ts_ns: int) -> MarketState:
    ms = MarketState()
    ms.apply(
        MarkEvent(
            venue="binance",
            product_type=ProductType.PERP,
            mechanism=Mechanism.CLOB,
            symbol="BTC",
            exchange_ts=now_ns,
            local_recv_ts=now_ns,
            mark_px=74_000.0,
        )
    )
    ms.apply(
        QuestionMetaEvent(
            venue="polymarket",
            product_type=ProductType.PREDICTION_BINARY,
            mechanism=Mechanism.CLOB,
            symbol="YES_TOKEN",
            exchange_ts=now_ns,
            local_recv_ts=now_ns,
            question_idx=909100,
            named_outcome_idxs=[0, 1],
            keys=["class", "underlying", "series_slug", "yes_token_id", "no_token_id", "strike_ref_ts_ns"],
            values=["priceBinary", "BTC", "btc-up-or-down-daily", "YES_TOKEN", "NO_TOKEN", str(strike_ref_ts_ns)],
        )
    )
    return ms


def test_scanner_reloads_persisted_pm_strike(tmp_path):
    # The scanner is pure/sync and must not capture the strike itself. Once
    # EngineRuntime._maybe_capture_pm_strike has persisted the strike to the
    # DAL, the next scan tick must reload it into the shared MarketState.
    now = 1_700_000_000_000_000_000
    ms = _seed_pm_updown(now, strike_ref_ts_ns=now - 120_000_000_000)
    cfg = _cfg_with_match({"class": "priceBinary", "underlying": "BTC"})
    scanner = _scanner_for(cfg, ms, tmp_path, now)
    scanner.dal.set_pm_strike(909100, 73_644.92)  # as if the runtime captured it
    scanner.scan(now_ns=now)
    assert ms.question(909100).strike == 73_644.92


# --- Daily PnL-window boundary (06:00 UTC for HL HIP-4) ---

from hlanalysis.engine.scanner import Scanner as _Scanner
import datetime as _dt


def test_scan_uses_injected_realized_pnl_and_skips_provider(tmp_path):
    """SHR-41: the runtime pre-fetches realized PnL off the event loop and
    injects it, so scan() must use the injected value and NOT call its
    (blocking) pnl_provider."""
    now = 1_700_000_000_000_000_000
    ms = _seed_market(now)
    cfg = _strategy_cfg()
    scanner = _scanner_for(cfg, ms, tmp_path, now)
    calls: list[int] = []
    scanner._pnl_provider = lambda ns: (calls.append(ns), 0.0)[1]

    scanner.scan(now_ns=now, realized_pnl_today=-123.0)

    assert calls == [], "pnl_provider was called despite an injected value"


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


# --- Returns-window length derived from cfg, not hard-coded ---


def test_scanner_recent_returns_n_uses_max_vol_lookback_seconds():
    """Scanner must request enough 1m returns to cover vol_lookback_seconds
    across defaults + every allowlist entry. The legacy n=32 silently
    capped the strategy's σ window at 32 min regardless of YAML config."""
    from hlanalysis.engine.scanner import Scanner

    cfg = _strategy_cfg()
    # Override defaults + allowlist to a 1h lookback (60 samples expected).
    cfg = cfg.model_copy(
        update={
            "defaults": cfg.defaults.model_copy(update={"vol_lookback_seconds": 3600}),
            "allowlist": [
                cfg.allowlist[0].model_copy(update={"vol_lookback_seconds": 3600}),
            ],
        }
    )
    n = Scanner._required_returns_n(cfg)
    assert n >= 60, f"expected >=60 bars for 3600s lookback, got {n}"


def test_scanner_recent_returns_n_picks_max_across_allowlist_entries():
    """If priceBinary asks for 1800s but priceBucket asks for 7200s,
    scanner must size for the larger window so the bucket strategy gets
    the samples it needs."""
    from hlanalysis.engine.scanner import Scanner
    from hlanalysis.engine.config import AllowlistEntry

    cfg = _strategy_cfg()
    long_entry = AllowlistEntry(
        match={"class": "priceBucket", "underlying": "BTC", "period": "1d"},
        max_position_usd=100,
        stop_loss_pct=10,
        tte_min_seconds=0,
        tte_max_seconds=86400,
        price_extreme_threshold=0.85,
        distance_from_strike_usd_min=0,
        vol_max=100,
        vol_lookback_seconds=7200,
    )
    short_entry = cfg.allowlist[0].model_copy(update={"vol_lookback_seconds": 1800})
    cfg = cfg.model_copy(
        update={
            "defaults": cfg.defaults.model_copy(update={"vol_lookback_seconds": 1800}),
            "allowlist": [short_entry, long_entry],
        }
    )
    n = Scanner._required_returns_n(cfg)
    assert n >= 120, f"expected >=120 bars for 7200s lookback, got {n}"


def test_scanner_recent_returns_n_includes_theta_drift_lookback():
    """For theta_harvester, drift_lookback_seconds can exceed vol_lookback_seconds;
    scanner must size for whichever is larger."""
    from hlanalysis.engine.scanner import Scanner
    from hlanalysis.engine.config import ThetaParams

    cfg = _strategy_cfg().model_copy(
        update={
            "strategy_type": "theta_harvester",
            "theta": ThetaParams(
                vol_lookback_seconds=1800,
                drift_lookback_seconds=7200,
                vol_sampling_dt_seconds=60,
            ),
        }
    )
    n = Scanner._required_returns_n(cfg)
    assert n >= 120, f"expected >=120 bars for 7200s drift lookback, got {n}"


def test_scanner_recent_returns_n_floors_at_32():
    """Legacy behavior: never return fewer than 32 bars so existing strategies
    that read recent_returns directly (e.g. v3.4 LM gate) keep working."""
    from hlanalysis.engine.scanner import Scanner

    cfg = _strategy_cfg().model_copy(
        update={
            "defaults": _strategy_cfg().defaults.model_copy(update={"vol_lookback_seconds": 600}),
            "allowlist": [
                _strategy_cfg().allowlist[0].model_copy(update={"vol_lookback_seconds": 600}),
            ],
        }
    )
    n = Scanner._required_returns_n(cfg)
    assert n == 32  # 600/60 = 10 → floored to 32


# --- Gate-decision log: chosen-leg book snapshot for bucket markets ---


def test_gate_log_snapshot_uses_chosen_leg_book_when_diagnosed(tmp_path):
    """For bucket markets, the strategy can route to any one of N YES legs.
    The gate-decision log must snapshot THAT leg's book (not an arbitrary
    first leg), so operators can correlate the price columns with edge_yes
    without doing inverse arithmetic.
    """
    import json
    from hlanalysis.strategy.types import (
        Action,
        BookState,
        Decision,
        Diagnostic,
        QuestionView,
    )

    now = 1_700_000_000_000_000_000
    dal = StateDAL(tmp_path / "state.db")
    dal.run_migrations()
    cfg = _strategy_cfg()
    rcfg = LateResolutionConfig(
        tte_min_seconds=60,
        tte_max_seconds=1800,
        price_extreme_threshold=0.95,
        distance_from_strike_usd_min=200.0,
        vol_max=0.5,
        max_position_usd=100.0,
        stop_loss_pct=10.0,
        max_strike_distance_pct=10.0,
        min_recent_volume_usd=0.0,
        stale_data_halt_seconds=5,
    )
    log_path = tmp_path / "gate_decisions.jsonl"
    scanner = Scanner(
        strategy=LateResolutionStrategy(rcfg),
        cfg=cfg,
        market_state=MarketState(),
        dal=dal,
        kill_switch_path=tmp_path / "halt",
        last_reconcile_ns=now,
        gate_log_path=log_path,
    )

    bucket_q = QuestionView(
        question_idx=14,
        yes_symbol="",
        no_symbol="",
        strike=0.0,
        expiry_ns=now + 3600 * 1_000_000_000,
        underlying="BTC",
        klass="priceBucket",
        period="1d",
        leg_symbols=("#700", "#701", "#780", "#781"),
    )
    # First leg is a long-shot (price 0.01); chosen leg #780 is the favorite (0.95).
    books = {
        "#700": BookState(
            symbol="#700",
            bid_px=0.00301,
            bid_sz=10.0,
            ask_px=0.01,
            ask_sz=10.0,
            last_trade_ts_ns=now,
            last_l2_ts_ns=now,
        ),
        "#780": BookState(
            symbol="#780", bid_px=0.94, bid_sz=10.0, ask_px=0.95, ask_sz=10.0, last_trade_ts_ns=now, last_l2_ts_ns=now
        ),
    }
    decision = Decision(
        action=Action.HOLD,
        diagnostics=(
            Diagnostic(
                "info",
                "edge",
                (
                    ("p_model", "0.7847"),
                    ("edge_yes", "-0.1706"),
                    ("chosen_leg", "#780"),
                ),
            ),
        ),
    )

    scanner._maybe_log_gate_transition(
        question=bucket_q,
        decision=decision,
        books=books,
        now_ns=now,
    )
    row = json.loads(log_path.read_text().strip())
    # The chosen leg's book is what should appear in the snapshot columns.
    assert row["ask_px"] == 0.95, f"ask_px should be chosen leg's ask, got {row['ask_px']}"
    assert row["bid_px"] == 0.94
    assert row["reason"] == "edge"


def test_gate_log_snapshot_uses_held_position_book_when_no_chosen_leg(tmp_path):
    """For held-position paths (topup_hold, exit_eval, vol_insufficient_data
    while position is open) the strategy doesn't emit `chosen_leg` in the
    diag — the relevant leg is implicit from the held position. Snapshot
    must read THAT leg's book, not an arbitrary first leg of the question."""
    import json
    from hlanalysis.engine.state import Position as DalPosition
    from hlanalysis.strategy.types import (
        Action,
        BookState,
        Decision,
        Diagnostic,
        QuestionView,
    )

    now = 1_700_000_000_000_000_000
    dal = StateDAL(tmp_path / "state.db")
    dal.run_migrations()
    cfg = _strategy_cfg()
    rcfg = LateResolutionConfig(
        tte_min_seconds=60,
        tte_max_seconds=1800,
        price_extreme_threshold=0.95,
        distance_from_strike_usd_min=200.0,
        vol_max=0.5,
        max_position_usd=100.0,
        stop_loss_pct=10.0,
        max_strike_distance_pct=10.0,
        min_recent_volume_usd=0.0,
        stale_data_halt_seconds=5,
    )
    log_path = tmp_path / "gate_decisions.jsonl"
    scanner = Scanner(
        strategy=LateResolutionStrategy(rcfg),
        cfg=cfg,
        market_state=MarketState(),
        dal=dal,
        kill_switch_path=tmp_path / "halt",
        last_reconcile_ns=now,
        gate_log_path=log_path,
    )

    bucket_q = QuestionView(
        question_idx=14,
        yes_symbol="",
        no_symbol="",
        strike=0.0,
        expiry_ns=now + 3600 * 1_000_000_000,
        underlying="BTC",
        klass="priceBucket",
        period="1d",
        leg_symbols=("#700", "#701", "#780", "#781"),
    )
    books = {
        "#700": BookState(
            symbol="#700", bid_px=0.018, bid_sz=10.0, ask_px=0.04, ask_sz=10.0, last_trade_ts_ns=now, last_l2_ts_ns=now
        ),
        "#780": BookState(
            symbol="#780", bid_px=0.94, bid_sz=10.0, ask_px=0.95, ask_sz=10.0, last_trade_ts_ns=now, last_l2_ts_ns=now
        ),
    }
    # Topup-hold style decision: held position exists, no chosen_leg in diag.
    held_position = DalPosition(
        question_idx=14,
        symbol="#780",
        qty=100.0,
        avg_entry=0.93,
        realized_pnl=0.0,
        last_update_ts_ns=now,
        stop_loss_price=0.0,
    )
    decision = Decision(
        action=Action.HOLD,
        diagnostics=(
            Diagnostic(
                "info",
                "hold",
                (
                    ("edge_held", "0.0500"),
                    ("held_p", "0.99"),
                    ("reason", "not_needed"),
                ),
            ),
        ),
    )

    scanner._maybe_log_gate_transition(
        question=bucket_q,
        decision=decision,
        books=books,
        now_ns=now,
        position=held_position,
    )
    row = json.loads(log_path.read_text().strip())
    assert row["ask_px"] == 0.95, f"held-position snapshot should read #780's ask, got {row['ask_px']}"
    assert row["bid_px"] == 0.94


def test_gate_log_snapshot_prefers_binary_favorite_leg_when_no_chosen_leg(tmp_path):
    """For binary HOLDs without an explicit chosen_leg in the diagnostic
    (bid_notional_too_thin, no_favorite, edge), the snapshot should record
    the favourite (higher-mid) leg — that's the leg every leg-aware gate
    actually reasoned about. Was: PM gate log surfaced the underdog YES
    book (0.06–0.10) while the strategy was blocking on the NO favourite
    (0.90+), making the row mathematically inconsistent with the diag."""
    import json
    from hlanalysis.strategy.types import (
        Action,
        BookState,
        Decision,
        Diagnostic,
        QuestionView,
    )

    now = 1_700_000_000_000_000_000
    dal = StateDAL(tmp_path / "state.db")
    dal.run_migrations()
    cfg = _strategy_cfg()
    rcfg = LateResolutionConfig(
        tte_min_seconds=60,
        tte_max_seconds=1800,
        price_extreme_threshold=0.95,
        distance_from_strike_usd_min=200.0,
        vol_max=0.5,
        max_position_usd=100.0,
        stop_loss_pct=10.0,
        max_strike_distance_pct=10.0,
        min_recent_volume_usd=0.0,
        stale_data_halt_seconds=5,
    )
    log_path = tmp_path / "gate_decisions.jsonl"
    scanner = Scanner(
        strategy=LateResolutionStrategy(rcfg),
        cfg=cfg,
        market_state=MarketState(),
        dal=dal,
        kill_switch_path=tmp_path / "halt",
        last_reconcile_ns=now,
        gate_log_path=log_path,
    )

    q = QuestionView(
        question_idx=42,
        yes_symbol="@yes",
        no_symbol="@no",
        strike=80_000.0,
        expiry_ns=now + 600 * 1_000_000_000,
        underlying="BTC",
        klass="priceBinary",
        period="1h",
    )
    # NO is the favorite (mid 0.94), YES is the underdog (mid 0.07).
    books = {
        "@yes": BookState(
            symbol="@yes", bid_px=0.06, bid_sz=200.0, ask_px=0.08, ask_sz=10.0, last_trade_ts_ns=now, last_l2_ts_ns=now
        ),
        "@no": BookState(
            symbol="@no", bid_px=0.93, bid_sz=5.0, ask_px=0.95, ask_sz=10.0, last_trade_ts_ns=now, last_l2_ts_ns=now
        ),
    }
    decision = Decision(
        action=Action.HOLD, diagnostics=(Diagnostic("info", "bid_notional_too_thin", (("min_usd", "10.00"),)),)
    )
    scanner._maybe_log_gate_transition(
        question=q,
        decision=decision,
        books=books,
        now_ns=now,
    )
    row = json.loads(log_path.read_text().strip())
    assert row["ask_px"] == 0.95, f"should log NO favourite ask, got {row['ask_px']}"
    assert row["bid_px"] == 0.93
    assert row["bid_sz"] == 5.0  # the actually-failing notional, $4.65
    assert row["reason"] == "bid_notional_too_thin"


def test_gate_log_snapshot_falls_back_to_first_leg_when_no_chosen_leg(tmp_path):
    """When the diagnostic doesn't carry chosen_leg (binary path, or other
    reasons like tte_out_of_window), preserve the legacy first-leg snapshot."""
    import json
    from hlanalysis.strategy.types import (
        Action,
        BookState,
        Decision,
        Diagnostic,
        QuestionView,
    )

    now = 1_700_000_000_000_000_000
    dal = StateDAL(tmp_path / "state.db")
    dal.run_migrations()
    cfg = _strategy_cfg()
    rcfg = LateResolutionConfig(
        tte_min_seconds=60,
        tte_max_seconds=1800,
        price_extreme_threshold=0.95,
        distance_from_strike_usd_min=200.0,
        vol_max=0.5,
        max_position_usd=100.0,
        stop_loss_pct=10.0,
        max_strike_distance_pct=10.0,
        min_recent_volume_usd=0.0,
        stale_data_halt_seconds=5,
    )
    log_path = tmp_path / "gate_decisions.jsonl"
    scanner = Scanner(
        strategy=LateResolutionStrategy(rcfg),
        cfg=cfg,
        market_state=MarketState(),
        dal=dal,
        kill_switch_path=tmp_path / "halt",
        last_reconcile_ns=now,
        gate_log_path=log_path,
    )

    q = QuestionView(
        question_idx=42,
        yes_symbol="@30",
        no_symbol="@31",
        strike=80_000.0,
        expiry_ns=now + 600 * 1_000_000_000,
        underlying="BTC",
        klass="priceBinary",
        period="1h",
    )
    books = {
        "@30": BookState(
            symbol="@30", bid_px=0.95, bid_sz=10.0, ask_px=0.96, ask_sz=10.0, last_trade_ts_ns=now, last_l2_ts_ns=now
        ),
    }
    decision = Decision(
        action=Action.HOLD, diagnostics=(Diagnostic("info", "tte_out_of_window", (("tte_s", "8000"),)),)
    )
    scanner._maybe_log_gate_transition(
        question=q,
        decision=decision,
        books=books,
        now_ns=now,
    )
    row = json.loads(log_path.read_text().strip())
    assert row["ask_px"] == 0.96
    assert row["reason"] == "tte_out_of_window"


# ---- recent_hl_bars threading + dormant-Parkinson activation ---------------


class _RecordingStrategy(LateResolutionStrategy):
    """Wraps a real strategy but records the recent_hl_bars kwarg it was
    handed, so we can assert the Scanner actually threads it through (the
    dormant-Parkinson bug was the Scanner NEVER passing recent_hl_bars)."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.seen_hl_bars = None
        self.seen_returns = None

    def evaluate(self, *, recent_hl_bars=(), recent_returns=(), **kw):
        self.seen_hl_bars = recent_hl_bars
        self.seen_returns = recent_returns
        return super().evaluate(recent_hl_bars=recent_hl_bars, recent_returns=recent_returns, **kw)


def _rcfg(**overrides):
    base = dict(
        tte_min_seconds=60,
        tte_max_seconds=1800,
        price_extreme_threshold=0.95,
        distance_from_strike_usd_min=200.0,
        vol_max=0.5,
        max_position_usd=100.0,
        stop_loss_pct=10.0,
        max_strike_distance_pct=10.0,
        min_recent_volume_usd=0.0,
        stale_data_halt_seconds=5,
    )
    base.update(overrides)
    return LateResolutionConfig(**base)


def _seed_market_with_range(now_ns: int) -> MarketState:
    """Like _seed_market but each 60s bucket carries an intra-bucket H/L range
    while the bucket CLOSES return to a flat level — so close-to-close stdev σ
    is ~0 but Parkinson σ (range-based) is large."""
    ms = MarketState()
    from datetime import datetime, timezone

    expiry_str = datetime.fromtimestamp((now_ns + 10 * 60 * 1_000_000_000) / 1e9, tz=timezone.utc).strftime(
        "%Y%m%d-%H%M"
    )
    ms.apply(
        QuestionMetaEvent(
            venue="hyperliquid",
            product_type=ProductType.PREDICTION_BINARY,
            mechanism=Mechanism.CLOB,
            symbol="qmeta",
            exchange_ts=now_ns - 60_000_000_000,
            local_recv_ts=now_ns - 60_000_000_000,
            question_idx=42,
            named_outcome_idxs=[3],
            keys=["class", "underlying", "period", "expiry", "strike"],
            values=["priceBinary", "BTC", "1h", expiry_str, "80000"],
        )
    )
    one_min = 60_000_000_000
    # 8 buckets; each opens+closes at 80_300 (flat closes) but swings ±300 mid.
    for i in range(8):
        bucket_base = now_ns - (8 - i) * one_min
        for off, px in ((0, 80_300.0), (1, 80_600.0), (2, 80_000.0), (3, 80_300.0)):
            ms.apply(
                MarkEvent(
                    venue="hyperliquid",
                    product_type=ProductType.PERP,
                    mechanism=Mechanism.CLOB,
                    symbol="BTC",
                    exchange_ts=bucket_base + off,
                    local_recv_ts=bucket_base + off,
                    mark_px=px,
                )
            )
    ms.apply(
        BboEvent(
            venue="hyperliquid",
            product_type=ProductType.PREDICTION_BINARY,
            mechanism=Mechanism.CLOB,
            symbol="#30",
            exchange_ts=now_ns,
            local_recv_ts=now_ns,
            bid_px=0.95,
            bid_sz=10.0,
            ask_px=0.96,
            ask_sz=10.0,
        )
    )
    ms.apply(
        BboEvent(
            venue="hyperliquid",
            product_type=ProductType.PREDICTION_BINARY,
            mechanism=Mechanism.CLOB,
            symbol="#31",
            exchange_ts=now_ns,
            local_recv_ts=now_ns,
            bid_px=0.04,
            bid_sz=10.0,
            ask_px=0.05,
            ask_sz=10.0,
        )
    )
    return ms


def test_scanner_threads_recent_hl_bars_into_evaluate(tmp_path):
    """Regression for the dormant-Parkinson bug: the Scanner must pass
    recent_hl_bars to strategy.evaluate, matching MarketState.recent_hl_bars."""
    now = 1_700_000_000_000_000_000
    ms = _seed_market_with_range(now)
    dal = StateDAL(tmp_path / "state.db")
    dal.run_migrations()
    strat = _RecordingStrategy(_rcfg())
    scanner = Scanner(
        strategy=strat,
        cfg=_strategy_cfg(),
        market_state=ms,
        dal=dal,
        kill_switch_path=tmp_path / "halt",
        last_reconcile_ns=now,
    )
    scanner.scan(now_ns=now)
    assert strat.seen_hl_bars is not None
    assert len(strat.seen_hl_bars) > 0
    expected = ms.recent_hl_bars("BTC", n=scanner._recent_returns_n)
    assert strat.seen_hl_bars == expected
    # each bar carries a real range (high > low)
    assert all(h > l for (h, l) in strat.seen_hl_bars)


def test_parkinson_differs_from_stdev_on_live_scanner_path(tmp_path):
    """With intra-bucket H/L range, a vol_estimator=parkinson slot must reach a
    DIFFERENT decision than stdev on the live Scanner path. Here flat closes →
    stdev σ≈0 (enters) while Parkinson σ from the range exceeds vol_max (holds).
    """
    now = 1_700_000_000_000_000_000
    dal = StateDAL(tmp_path / "state.db")
    dal.run_migrations()
    # vol_max sits between stdev σ (~0) and Parkinson σ (range-based, ~0.003).
    vmax = 0.001

    def run(estimator: str):
        ms = _seed_market_with_range(now)
        scanner = Scanner(
            strategy=LateResolutionStrategy(_rcfg(vol_max=vmax, vol_estimator=estimator)),
            cfg=_strategy_cfg(),
            market_state=ms,
            dal=dal,
            kill_switch_path=tmp_path / "halt",
            last_reconcile_ns=now,
        )
        return scanner.scan(now_ns=now)

    stdev_dec = run("stdev")
    park_dec = run("parkinson")
    assert any(d.decision.action is Action.ENTER for d in stdev_dec)
    assert not any(d.decision.action is Action.ENTER for d in park_dec)


def test_parkinson_equals_stdev_on_flat_series(tmp_path):
    """Sanity: with a flat series (H==L per bucket) Parkinson degenerates and
    falls back to stdev, so the decision matches stdev — no spurious change."""
    now = 1_700_000_000_000_000_000
    dal = StateDAL(tmp_path / "state.db")
    dal.run_migrations()

    def run(estimator: str):
        ms = _seed_market(now)  # flat marks, no intra-bucket range
        scanner = Scanner(
            strategy=LateResolutionStrategy(_rcfg(vol_estimator=estimator)),
            cfg=_strategy_cfg(),
            market_state=ms,
            dal=dal,
            kill_switch_path=tmp_path / "halt",
            last_reconcile_ns=now,
        )
        return [d.decision.action for d in scanner.scan(now_ns=now)]

    assert run("stdev") == run("parkinson")


# ---- NaN-strike PM guard ---------------------------------------------------


def _seed_pm_updown_with_books(now_ns: int, *, strike_ref_ts_ns: int) -> MarketState:
    """Like _seed_pm_updown but also seeds:
    - multiple BTC marks (so recent_returns has ≥2 entries),
    - an expiry within the TTE window (600s from now),
    - BBO books for YES_TOKEN / NO_TOKEN at prices that pass the
      price_extreme_threshold=0.95 gate.

    With this setup the LateResolutionStrategy WOULD evaluate the question
    and reach ENTER — unless the scanner guard skips it due to NaN strike.
    The strike is intentionally NOT set (no `set_question_strike` call) so
    it stays NaN.
    """
    from datetime import datetime, timezone

    ms = MarketState()
    # Enough 60s-spaced BTC marks for recent_returns to have >2 entries.
    for i in range(10):
        ts = now_ns - (10 - i) * 60_000_000_000
        ms.apply(
            MarkEvent(
                venue="binance",
                product_type=ProductType.PERP,
                mechanism=Mechanism.CLOB,
                symbol="BTC",
                exchange_ts=ts,
                local_recv_ts=ts,
                mark_px=74_000.0 + i * 0.5,
            )
        )
    expiry_str = datetime.fromtimestamp(
        (now_ns + 600 * 1_000_000_000) / 1e9,
        tz=timezone.utc,
    ).strftime("%Y%m%d-%H%M")
    ms.apply(
        QuestionMetaEvent(
            venue="polymarket",
            product_type=ProductType.PREDICTION_BINARY,
            mechanism=Mechanism.CLOB,
            symbol="YES_TOKEN",
            exchange_ts=now_ns,
            local_recv_ts=now_ns,
            question_idx=909100,
            named_outcome_idxs=[0, 1],
            keys=["class", "underlying", "series_slug", "expiry", "yes_token_id", "no_token_id", "strike_ref_ts_ns"],
            values=[
                "priceBinary",
                "BTC",
                "btc-up-or-down-daily",
                expiry_str,
                "YES_TOKEN",
                "NO_TOKEN",
                str(strike_ref_ts_ns),
            ],
        )
    )
    # YES leg is the favourite at 0.96 ask — passes price_extreme_threshold=0.95.
    # Deep-enough bid so bid_notional gate (disabled at 0) is irrelevant.
    ms.apply(
        BboEvent(
            venue="polymarket",
            product_type=ProductType.PREDICTION_BINARY,
            mechanism=Mechanism.CLOB,
            symbol="YES_TOKEN",
            exchange_ts=now_ns,
            local_recv_ts=now_ns,
            bid_px=0.95,
            bid_sz=200.0,
            ask_px=0.96,
            ask_sz=200.0,
        )
    )
    ms.apply(
        BboEvent(
            venue="polymarket",
            product_type=ProductType.PREDICTION_BINARY,
            mechanism=Mechanism.CLOB,
            symbol="NO_TOKEN",
            exchange_ts=now_ns,
            local_recv_ts=now_ns,
            bid_px=0.03,
            bid_sz=200.0,
            ask_px=0.04,
            ask_sz=200.0,
        )
    )
    return ms


def test_scanner_skips_pm_question_with_unresolved_strike(tmp_path):
    """A PM question whose strike is still NaN (capture not yet fired) must
    produce NO ENTER decision. Without the guard the strategy evaluates the
    market, safety_d is NaN → skipped, and ENTER is emitted.

    TDD: write before Fix 1 → the test FAILS (ENTER emitted without guard).
    After adding the guard in Scanner.scan() → the test PASSES (question is
    skipped before strategy.evaluate is called).
    """
    import math

    now = 1_700_000_000_000_000_000
    ms = _seed_pm_updown_with_books(now, strike_ref_ts_ns=now - 120_000_000_000)
    cfg = _cfg_with_match({"class": "priceBinary", "underlying": "BTC"})
    scanner = _scanner_for(cfg, ms, tmp_path, now)
    # Precondition: strike must still be NaN (no capture/persist).
    assert math.isnan(ms.question(909100).strike), "precondition failed: expected NaN strike before guard test"
    decisions = scanner.scan(now_ns=now)
    assert all(d.decision.action is not Action.ENTER for d in decisions), (
        "scanner produced ENTER for a PM question with NaN strike — NaN-strike guard is missing"
    )
