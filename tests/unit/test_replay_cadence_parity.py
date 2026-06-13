# tests/unit/test_replay_cadence_parity.py
"""ReplayRunner single-cadence reads stay bit-identical after the (symbol, dt)
refactor: a dt-less read resolves to the sole registered cadence.

Also asserts the SHR-66 fix: ReplayRunner passes now_ns+lookback_seconds so
bars older than the lookback window are dropped after a feed gap (time-bounded
path), matching the backtest slice_window semantics."""

from __future__ import annotations

from hlanalysis.engine.market_state import MarketState
from hlanalysis.engine.replay import ReplayRunner
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


def test_dtless_read_equals_explicit_dt_for_single_cadence() -> None:
    ms = MarketState()
    ms.set_reference_cadence("BTC", sampling_dt_seconds=5, lookback_seconds=1800)
    for t in range(50):
        ts = t * 1_000_000_000
        ms.apply(
            MarkEvent(
                venue="hyperliquid",
                product_type=ProductType.PERP,
                mechanism=Mechanism.CLOB,
                symbol="BTC",
                exchange_ts=ts,
                local_recv_ts=ts,
                mark_px=100.0 + (t % 7) * 0.5,
            )
        )
    assert ms.recent_returns("BTC", n=32) == ms.recent_returns("BTC", n=32, dt=5)
    assert ms.recent_hl_bars("BTC", n=32) == ms.recent_hl_bars("BTC", n=32, dt=5)


def test_replay_runner_uses_time_bounded_path_after_feed_gap() -> None:
    """After a feed gap larger than the lookback window, ReplayRunner should
    return empty returns (time-bounded path drops stale bars) rather than
    returning old bars via the legacy COUNT path (SHR-66)."""
    cfg = LateResolutionConfig(
        tte_min_seconds=60,
        tte_max_seconds=86400,
        price_extreme_threshold=0.95,
        distance_from_strike_usd_min=200.0,
        vol_max=1.0,
        max_position_usd=100.0,
        stop_loss_pct=10.0,
        max_strike_distance_pct=50.0,
        min_recent_volume_usd=0.0,
        stale_data_halt_seconds=999_999,
    )
    # sampling_dt=60s, n=32 → lookback window = 32*60 = 1920s
    runner = ReplayRunner(strategy=LateResolutionStrategy(cfg), sampling_dt_seconds=60)

    # Seed 10 bars well before the gap start (> 1920s before now_ns).
    gap_end_ns = 10_000 * 1_000_000_000  # t=10000s
    seed_base_ns = 1_000 * 1_000_000_000  # t=1000s (9000s before gap_end)

    events: list = []
    events.append(
        QuestionMetaEvent(
            venue="hyperliquid",
            product_type=ProductType.PREDICTION_BINARY,
            mechanism=Mechanism.CLOB,
            symbol="qmeta",
            exchange_ts=seed_base_ns,
            local_recv_ts=seed_base_ns,
            question_idx=1,
            named_outcome_idxs=[0],
            keys=["class", "underlying", "period", "expiry", "strike"],
            values=["priceBinary", "BTC", "1h", "20500101-1200", "99999"],
        )
    )
    # 10 seed bars at t=1000..1009 (all > 1920s before gap_end=10000s)
    for i in range(10):
        ts = seed_base_ns + i * 60 * 1_000_000_000
        events.append(
            MarkEvent(
                venue="hyperliquid",
                product_type=ProductType.PERP,
                mechanism=Mechanism.CLOB,
                symbol="BTC",
                exchange_ts=ts,
                local_recv_ts=ts,
                mark_px=80000.0 + i,
            )
        )
    # Trigger: one BBO event at gap_end (9000s after the last seed bar)
    events.append(
        BboEvent(
            venue="hyperliquid",
            product_type=ProductType.PREDICTION_BINARY,
            mechanism=Mechanism.CLOB,
            symbol="#0",
            exchange_ts=gap_end_ns,
            local_recv_ts=gap_end_ns,
            bid_px=0.50,
            bid_sz=10.0,
            ask_px=0.51,
            ask_sz=10.0,
        )
    )

    # Feed events; capture what recent_returns the strategy sees at the BBO tick.
    captured: list[tuple[float, ...]] = []
    _orig_eval = runner._strategy.evaluate

    def _capturing_eval(**kwargs):  # type: ignore[override]
        captured.append(
            runner._market.recent_returns(
                runner._ref,
                n=runner._recent_returns_n,
                now_ns=kwargs.get("now_ns"),
                lookback_seconds=runner._recent_returns_n * runner._sampling_dt_seconds,
            )
        )
        return _orig_eval(**kwargs)

    runner._strategy.evaluate = _capturing_eval  # type: ignore[method-assign]
    list(runner.run_iter(events))

    # With the time-bounded path: 10 seed bars are all outside the 1920s window
    # (they are 9000–8460s before gap_end), so recent_returns must be empty.
    # With the legacy COUNT path it would return 9 returns from the stale bars.
    assert len(captured) > 0, "strategy.evaluate was never called"
    assert captured[-1] == (), (
        "expected empty returns after feed gap (time-bounded path), "
        f"got {len(captured[-1])} returns — legacy COUNT path still active"
    )
