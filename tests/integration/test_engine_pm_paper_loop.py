"""Engine with a v31_pm paper slot scans a stubbed market and emits at
least one ENTER decision, all without network. Validates that the
ExecutionClient seam is properly threaded through Router and Scanner for
a Polymarket-typed account."""
from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from pathlib import Path

import pytest

from hlanalysis.adapters.base import VenueAdapter
from hlanalysis.engine.config import (
    AccountConfig,
    AlertsConfig,
    AllowlistEntry,
    DeployConfig,
    GlobalRiskConfig,
    PolymarketAccount,
    StrategyConfig,
    TelegramConfig,
    ThetaParams,
)
from hlanalysis.engine.exec_client import ExecutionClient
from hlanalysis.engine.pm_client import PMClient
from hlanalysis.engine.runtime import EngineRuntime
from hlanalysis.events import (
    BboEvent,
    MarkEvent,
    Mechanism,
    ProductType,
    QuestionMetaEvent,
)


class _PMStubAdapter(VenueAdapter):
    venue = "polymarket"

    def supports(self, *_a, **_k) -> bool:
        return True

    async def stream(self, _subs) -> AsyncIterator:
        # Real wall-clock timestamps so stale-data halt doesn't fire.
        now = time.time_ns()
        # ~12h ahead, formatted as YYYYMMDD-HHMM (the engine's parser).
        expiry_str = datetime.fromtimestamp(
            (now + 12 * 3600 * 1_000_000_000) / 1e9, tz=timezone.utc,
        ).strftime("%Y%m%d-%H%M")

        # Real PM question: legs are the CLOB token ids (yes_token_id /
        # no_token_id), which the PM book frames are keyed by too.
        yield QuestionMetaEvent(
            venue="polymarket", product_type=ProductType.PREDICTION_BINARY,
            mechanism=Mechanism.CLOB, symbol="YES_TOKEN",
            exchange_ts=now - 90 * 60 * 1_000_000_000,
            local_recv_ts=now - 90 * 60 * 1_000_000_000,
            question_idx=909001, named_outcome_idxs=[0, 1],
            keys=["class", "underlying", "period", "expiry", "strike",
                  "yes_token_id", "no_token_id", "series_slug"],
            values=["priceBinary", "BTC", "1d", expiry_str, "70000",
                    "YES_TOKEN", "NO_TOKEN", "btc-up-or-down-daily"],
        )

        # 90 BTC marks at 60s cadence so theta_harvester's σ has enough data
        # (vol_lookback_seconds=3600 / vol_sampling_dt_seconds=60 = 60 samples
        # required; emit a few extra for headroom). Prices drift mildly so
        # σ is positive (tiny) — sufficient to pass the vol_clip_min=0.0 gate.
        for i in range(90):
            ts = now - (90 - i) * 60_000_000_000
            yield MarkEvent(
                venue="binance", product_type=ProductType.SPOT,
                mechanism=Mechanism.CLOB, symbol="BTC",
                exchange_ts=ts, local_recv_ts=ts,
                mark_px=80_000.0 + (i % 5) * 0.5,
            )

        # YES leg — favorite at 0.925 mid, deep enough bid_notional to pass
        # the 10 USD floor. BTC at 80k vs strike 70k → p_yes ≈ 1.0, so
        # edge ≈ 1 - 0.93 - 0.005 = 0.065 > edge_buffer 0.03 → ENTER.
        yield BboEvent(
            venue="polymarket", product_type=ProductType.PREDICTION_BINARY,
            mechanism=Mechanism.CLOB, symbol="YES_TOKEN",
            exchange_ts=now, local_recv_ts=now,
            bid_px=0.92, bid_sz=200.0, ask_px=0.93, ask_sz=200.0,
        )
        yield BboEvent(
            venue="polymarket", product_type=ProductType.PREDICTION_BINARY,
            mechanism=Mechanism.CLOB, symbol="NO_TOKEN",
            exchange_ts=now, local_recv_ts=now,
            bid_px=0.06, bid_sz=200.0, ask_px=0.07, ask_sz=200.0,
        )
        # Keep the adapter alive so the runtime's scan loop ticks at least once.
        await asyncio.sleep(2.5)


class _PMUpDownStubAdapter(VenueAdapter):
    """A PM 'BTC Up or Down' market: no static strike, just a reference
    candle ts (strike_ref_ts_ns). The reference minute is already closed
    (strike_ref_ts_ns is 90s in the past) so the engine must fetch the
    Binance spot 1m close rather than stamping the live mark."""

    venue = "polymarket"

    def supports(self, *_a, **_k) -> bool:
        return True

    async def stream(self, _subs) -> AsyncIterator:
        now = time.time_ns()
        expiry_str = datetime.fromtimestamp(
            (now + 12 * 3600 * 1_000_000_000) / 1e9, tz=timezone.utc,
        ).strftime("%Y%m%d-%H%M")
        # Live reference mark at 73_500 — this is the value the engine must
        # NOT capture (the new path uses the spot 1m close, not the live mark).
        yield MarkEvent(
            venue="binance", product_type=ProductType.SPOT,
            mechanism=Mechanism.CLOB, symbol="BTC",
            exchange_ts=now, local_recv_ts=now, mark_px=73_500.0,
        )
        # strike_ref_ts_ns is 90s in the past → the reference candle has
        # already closed (>60s), so the runtime's capture path should fire.
        yield QuestionMetaEvent(
            venue="polymarket", product_type=ProductType.PREDICTION_BINARY,
            mechanism=Mechanism.CLOB, symbol="YES_TOKEN",
            exchange_ts=now, local_recv_ts=now,
            question_idx=909002, named_outcome_idxs=[0, 1],
            keys=["class", "underlying", "expiry", "series_slug",
                  "yes_token_id", "no_token_id", "strike_ref_ts_ns"],
            values=["priceBinary", "BTC", expiry_str, "btc-up-or-down-daily",
                    "YES_TOKEN", "NO_TOKEN", str(now - 90 * 1_000_000_000)],
        )
        await asyncio.sleep(2.5)


class _PMUpDownSpotMarkStubAdapter(VenueAdapter):
    """Like _PMUpDownStubAdapter but also emits a Binance SPOT mark with
    symbol='BTCUSDT' (remapped to BTCUSDT_SPOT by _remap_reference_symbol) so
    market_state.last_mark('BTCUSDT_SPOT') returns 73_500.  The klines_fetcher
    returns 73_644.92, giving ~19.5 bps divergence (above the 10 bps threshold)
    — this exercises the PMStrikeMismatch alert path."""

    venue = "polymarket"

    def supports(self, *_a, **_k) -> bool:
        return True

    async def stream(self, _subs) -> AsyncIterator:
        now = time.time_ns()
        expiry_str = datetime.fromtimestamp(
            (now + 12 * 3600 * 1_000_000_000) / 1e9, tz=timezone.utc,
        ).strftime("%Y%m%d-%H%M")
        # Emit a spot BBO mark under symbol="BTCUSDT" — _remap_reference_symbol
        # in _ingest_loop renames this to BTCUSDT_SPOT so last_mark("BTCUSDT_SPOT")
        # returns 73_500.  (symbol="BTC" would not be remapped; "BTCUSDT" is the
        # canonical Binance SPOT symbol the remap function expects.)
        yield MarkEvent(
            venue="binance", product_type=ProductType.SPOT,
            mechanism=Mechanism.CLOB, symbol="BTCUSDT",
            exchange_ts=now, local_recv_ts=now, mark_px=73_500.0,
        )
        # strike_ref_ts_ns is 90s in the past → candle already closed.
        yield QuestionMetaEvent(
            venue="polymarket", product_type=ProductType.PREDICTION_BINARY,
            mechanism=Mechanism.CLOB, symbol="YES_TOKEN",
            exchange_ts=now, local_recv_ts=now,
            question_idx=909005, named_outcome_idxs=[0, 1],
            keys=["class", "underlying", "expiry", "series_slug",
                  "yes_token_id", "no_token_id", "strike_ref_ts_ns"],
            values=["priceBinary", "BTC", expiry_str, "btc-up-or-down-daily",
                    "YES_TOKEN", "NO_TOKEN", str(now - 90 * 1_000_000_000)],
        )
        await asyncio.sleep(2.5)


class _PMUpDownRestartStubAdapter(VenueAdapter):
    """A PM up/down market whose strike_ref_ts is in the past (≪ now). Used by
    both the reload test (persisted strike reused) and the fresh-engine test
    (strike captured from klines when no persisted value exists)."""

    venue = "polymarket"

    def supports(self, *_a, **_k) -> bool:
        return True

    async def stream(self, _subs) -> AsyncIterator:
        now = time.time_ns()
        past_open = now - 6 * 3600 * 1_000_000_000  # 6h ago → well past the 60s candle-close gate
        expiry_str = datetime.fromtimestamp(
            (now + 6 * 3600 * 1_000_000_000) / 1e9, tz=timezone.utc,
        ).strftime("%Y%m%d-%H%M")
        yield MarkEvent(
            venue="binance", product_type=ProductType.SPOT,
            mechanism=Mechanism.CLOB, symbol="BTC",
            exchange_ts=now, local_recv_ts=now, mark_px=80_000.0,
        )
        yield QuestionMetaEvent(
            venue="polymarket", product_type=ProductType.PREDICTION_BINARY,
            mechanism=Mechanism.CLOB, symbol="YES_TOKEN",
            exchange_ts=now, local_recv_ts=now,
            question_idx=909003, named_outcome_idxs=[0, 1],
            keys=["class", "underlying", "expiry", "series_slug",
                  "yes_token_id", "no_token_id", "strike_ref_ts_ns"],
            values=["priceBinary", "BTC", expiry_str, "btc-up-or-down-daily",
                    "YES_TOKEN", "NO_TOKEN", str(past_open)],
        )
        await asyncio.sleep(2.5)


class _FakeTelegram:
    def __init__(self) -> None:
        self.messages: list[str] = []

    async def send(self, text: str, *, markdown: bool = True) -> bool:
        self.messages.append(text)
        return True


@pytest.fixture
def cfgs(tmp_path):
    # Mirrors the v31_pm slot from config/strategy.yaml (PM-tuned theta
    # harvester) but with the risk-gate floors relaxed for a stub-data
    # integration test (no trades, so recent_volume is 0).
    entry = AllowlistEntry(
        match={"class": "priceBinary", "underlying": "BTC"},
        max_position_usd=200, stop_loss_pct=None,
        tte_min_seconds=0, tte_max_seconds=86400,
        price_extreme_threshold=0.0, price_extreme_max=1.0,
        distance_from_strike_usd_min=0, vol_max=100,
        entry_cooldown_seconds=60,
    )
    strategy = StrategyConfig(
        name="theta_harvester",
        account_alias="v31_pm",
        strategy_type="theta_harvester",
        paper_mode=True,
        allowlist=[entry],
        blocklist_question_idxs=[],
        defaults=entry,
        theta=ThetaParams(
            vol_lookback_seconds=3600, vol_sampling_dt_seconds=60,
            vol_clip_min=0.0, vol_clip_max=5.0,
            edge_buffer=0.03, fee_taker=0.0, half_spread_assumption=0.005,
            drift_lookback_seconds=3600, drift_blend=0.0,
            favorite_threshold=0.90, edge_max=None,
            exit_edge_threshold=0.0, time_stop_seconds=0,
            exit_take_profit_mode=True, exit_fee=0.0007,
            min_distance_pct=None, min_bid_notional_usd=10.0,
            topup_enabled=False,
            fee_model="pm_binary", fee_rate=0.07,
        ),
        **{"global": GlobalRiskConfig(
            max_total_inventory_usd=1000, max_concurrent_positions=5,
            daily_loss_cap_usd=100, max_strike_distance_pct=50,
            # Relaxed from 100 → 0: stub emits no TradeEvents, so the
            # last-hour-notional gauge is 0. We want to verify the scanner
            # produces a decision; the recent-volume risk gate is exercised
            # separately by the unit tests for RiskGate.
            min_recent_volume_usd=0,
            stale_data_halt_seconds=30, reconcile_interval_seconds=60,
            daily_window_start_hour_utc=6,
        )},
    )

    pm_account: AccountConfig = PolymarketAccount(
        clob_host="https://clob.polymarket.com",
        chain_id=137,
        private_key="0xstub",
        clob_api_key="stub", clob_api_secret="stub",
        clob_api_passphrase="stub",
    )
    deploy = DeployConfig(
        env="dev",
        accounts={"v31_pm": pm_account},
        alerts=AlertsConfig(telegram=TelegramConfig(bot_token="x", chat_id="y")),
        state_db_path=str(tmp_path / "state.db"),
        kill_switch_path=str(tmp_path / "halt"),
    )
    return strategy, deploy


@pytest.mark.asyncio
async def test_pm_paper_slot_emits_decision(cfgs):
    strategy_cfg, deploy_cfg = cfgs
    fake_tg = _FakeTelegram()

    def _exec_factory(
        _alias: str, _acct: AccountConfig, paper: bool,
    ) -> ExecutionClient:
        # Always paper for the test; the slot's paper_mode flows through here.
        return PMClient(paper_mode=paper)

    runtime = EngineRuntime(
        strategies=[strategy_cfg],
        deploy_cfg=deploy_cfg,
        adapter_factory=_PMStubAdapter,
        subscriptions=[],
        exec_client_factory=_exec_factory,
        telegram_factory=lambda _http: fake_tg,
    )
    runtime_task = asyncio.create_task(runtime.run())
    # Give the ingest loop time to drain the stub and the scan loop time to
    # fire at least once (scan_loop sleeps 1s between ticks).
    await asyncio.sleep(4.0)
    runtime.stop_event.set()
    await asyncio.wait_for(runtime_task, timeout=5.0)

    assert runtime.slots, "runtime built no slots"
    [slot] = runtime.slots
    assert slot.alias == "v31_pm"
    assert isinstance(slot.exec_client, PMClient)
    assert slot.exec_client.paper_mode is True
    # The scanner must have ticked and the strategy must have produced at
    # least one ENTER decision under the favourite-leg setup.
    assert slot.scans_completed >= 1, (
        f"scanner never ticked: {slot.scans_completed=}"
    )
    assert slot.decisions_emitted >= 1, (
        f"strategy emitted no non-HOLD decisions: {slot.decisions_emitted=}"
    )


@pytest.mark.asyncio
async def test_pm_updown_strike_captured_from_spot_close(cfgs):
    # An up/down PM market carries no static strike. The reference candle is
    # already closed (strike_ref_ts_ns 90s in the past), so the engine must
    # fetch the Binance spot 1m CLOSE via klines_fetcher — NOT use the live
    # reference mark of 73_500.
    strategy_cfg, deploy_cfg = cfgs
    fake_tg = _FakeTelegram()

    runtime = EngineRuntime(
        strategies=[strategy_cfg],
        deploy_cfg=deploy_cfg,
        adapter_factory=_PMUpDownStubAdapter,
        subscriptions=[],
        exec_client_factory=lambda _a, _c, paper: PMClient(paper_mode=paper),
        telegram_factory=lambda _http: fake_tg,
        klines_fetcher=lambda _ts_ns: 73_644.92,
    )
    runtime_task = asyncio.create_task(runtime.run())
    await asyncio.sleep(3.0)
    runtime.stop_event.set()
    await asyncio.wait_for(runtime_task, timeout=5.0)

    q = runtime.market_state.question(909002)
    assert q is not None
    assert q.strike == 73_644.92          # spot 1m CLOSE, NOT the live mark 73_500
    [slot] = runtime.slots
    assert slot.dal.get_pm_strike(909002) == 73_644.92


@pytest.mark.asyncio
async def test_pm_updown_strike_reloaded_from_db_after_restart(cfgs):
    # Open was missed (strike_ref_ts 6h ago) so live capture can't fire; the
    # engine must reload the previously-persisted strike instead of skipping.
    from hlanalysis.engine.state import StateDAL

    strategy_cfg, deploy_cfg = cfgs
    # Pre-seed the slot's state DB as if a prior run had captured the open.
    db_path = Path(deploy_cfg.state_db_path_for("v31_pm"))
    seed = StateDAL(db_path)
    seed.run_migrations()
    seed.set_pm_strike(909003, 71_000.0)

    runtime = EngineRuntime(
        strategies=[strategy_cfg],
        deploy_cfg=deploy_cfg,
        adapter_factory=_PMUpDownRestartStubAdapter,
        subscriptions=[],
        exec_client_factory=lambda _a, _c, paper: PMClient(paper_mode=paper),
        telegram_factory=lambda _http: _FakeTelegram(),
    )
    runtime_task = asyncio.create_task(runtime.run())
    await asyncio.sleep(3.0)
    runtime.stop_event.set()
    await asyncio.wait_for(runtime_task, timeout=5.0)

    q = runtime.market_state.question(909003)
    assert q is not None
    # Strike came from the persisted value, not the (stale) live mark of 80_000.
    assert q.strike == 71_000.0


@pytest.mark.asyncio
async def test_pm_updown_strike_captured_when_open_already_closed(cfgs):
    # Open was missed AND nothing persisted (fresh engine that started after the
    # market's open). The engine must capture the strike from the Binance spot
    # 1m close (open already past) so the market is tradeable instead of skipped.
    strategy_cfg, deploy_cfg = cfgs

    runtime = EngineRuntime(
        strategies=[strategy_cfg],
        deploy_cfg=deploy_cfg,
        adapter_factory=_PMUpDownRestartStubAdapter,  # open 6h ago, qidx 909003
        subscriptions=[],
        exec_client_factory=lambda _a, _c, paper: PMClient(paper_mode=paper),
        telegram_factory=lambda _http: _FakeTelegram(),
        # Stub the historical fetch (no network); returns a known 1m close.
        klines_fetcher=lambda _ts_ns: 70_500.0,
    )
    runtime_task = asyncio.create_task(runtime.run())
    await asyncio.sleep(3.0)
    runtime.stop_event.set()
    await asyncio.wait_for(runtime_task, timeout=5.0)

    q = runtime.market_state.question(909003)
    assert q is not None
    assert q.strike == 70_500.0
    [slot] = runtime.slots
    assert slot.dal.get_pm_strike(909003) == 70_500.0


@pytest.mark.asyncio
async def test_pm_strike_capture_loop_registered_and_captures(cfgs):
    """The periodic _pm_strike_capture_loop must be registered as a task in
    run() and must independently capture the strike even if first-sight could
    not.  We use _PMUpDownRestartStubAdapter (ref_ts 6h ago, qidx 909003) and
    assert capture happened — the integration proves the loop is wired up, not
    just defined.  (The existing test_pm_updown_strike_captured_when_open_already_closed
    exercises the same happy-path; this test complements it by also asserting
    the loop method exists and is scheduled.)"""
    strategy_cfg, deploy_cfg = cfgs

    runtime = EngineRuntime(
        strategies=[strategy_cfg],
        deploy_cfg=deploy_cfg,
        adapter_factory=_PMUpDownRestartStubAdapter,  # qidx 909003, ref_ts 6h ago
        subscriptions=[],
        exec_client_factory=lambda _a, _c, paper: PMClient(paper_mode=paper),
        telegram_factory=lambda _http: _FakeTelegram(),
        klines_fetcher=lambda _ts_ns: 70_500.0,
    )

    # The loop must exist as a callable method.
    assert callable(getattr(runtime, "_pm_strike_capture_loop", None)), (
        "_pm_strike_capture_loop not defined on EngineRuntime"
    )

    runtime_task = asyncio.create_task(runtime.run())
    await asyncio.sleep(3.0)
    runtime.stop_event.set()
    await asyncio.wait_for(runtime_task, timeout=5.0)

    q = runtime.market_state.question(909003)
    assert q is not None and q.strike == 70_500.0, (
        f"strike not captured: {q.strike if q else 'question missing'}"
    )
    [slot] = runtime.slots
    assert slot.dal.get_pm_strike(909003) == 70_500.0


@pytest.mark.asyncio
async def test_pm_strike_capture_loop_direct(cfgs):
    """Drive _pm_strike_capture_loop directly: seed a PM question into
    market_state (bypass ingest so first-sight never fires), then let the loop
    run for ~1.5s, and confirm the strike was captured."""
    import math
    from hlanalysis.events import Mechanism, ProductType, QuestionMetaEvent

    strategy_cfg, deploy_cfg = cfgs

    runtime = EngineRuntime(
        strategies=[strategy_cfg],
        deploy_cfg=deploy_cfg,
        # Use a stub that emits nothing — we seed market_state manually.
        adapter_factory=_PMUpDownRestartStubAdapter,
        subscriptions=[],
        exec_client_factory=lambda _a, _c, paper: PMClient(paper_mode=paper),
        telegram_factory=lambda _http: _FakeTelegram(),
        klines_fetcher=lambda _ts_ns: 88_888.0,
    )

    # Build slots so _pm_strike_capture_loop has something to work with.
    runtime.slots = [runtime._build_slot(strategy_cfg)]
    slots = runtime.slots

    now_ns = time.time_ns()
    past_ref_ns = now_ns - 6 * 3600 * 1_000_000_000  # 6h ago → candle long closed
    expiry_str = datetime.fromtimestamp(
        (now_ns + 6 * 3600 * 1_000_000_000) / 1e9, tz=timezone.utc,
    ).strftime("%Y%m%d-%H%M")

    # Seed the question directly into market_state (no ingest loop).
    meta_ev = QuestionMetaEvent(
        venue="polymarket",
        product_type=ProductType.PREDICTION_BINARY,
        mechanism=Mechanism.CLOB,
        symbol="YES_TOKEN",
        exchange_ts=now_ns,
        local_recv_ts=now_ns,
        question_idx=909004,
        named_outcome_idxs=[0, 1],
        keys=["class", "underlying", "expiry", "series_slug",
              "yes_token_id", "no_token_id", "strike_ref_ts_ns"],
        values=["priceBinary", "BTC", expiry_str, "btc-up-or-down-daily",
                "YES_TOKEN", "NO_TOKEN", str(past_ref_ns)],
    )
    runtime.market_state.apply(meta_ev)
    qv = runtime.market_state.question(909004)
    assert qv is not None
    assert math.isnan(qv.strike), "strike should be NaN before capture"

    # Run the loop as a task for ~2.5s, then stop.
    loop_task = asyncio.create_task(runtime._pm_strike_capture_loop(slots))
    await asyncio.sleep(2.5)
    runtime.stop_event.set()
    await asyncio.wait_for(loop_task, timeout=3.0)

    q = runtime.market_state.question(909004)
    assert q is not None and q.strike == 88_888.0, (
        f"loop did not capture strike; got {q.strike if q else 'question missing'}"
    )
    [slot] = slots
    assert slot.dal.get_pm_strike(909004) == 88_888.0


@pytest.mark.asyncio
async def test_pm_strike_divergence_alerts_but_still_trades(cfgs):
    """Strike/spot-mark divergence fires a PMStrikeMismatch Telegram alert
    but the strike is still persisted (alert-only, no skip).

    The stub adapter emits a Binance SPOT MarkEvent with symbol='BTCUSDT'
    (remapped to BTCUSDT_SPOT by _remap_reference_symbol in _ingest_loop), so
    market_state.last_mark('BTCUSDT_SPOT') returns 73_500.  klines_fetcher
    returns 73_644.92 — that is ≈19.5 bps apart, above the 10 bps threshold —
    so the mismatch alert must fire."""
    strategy_cfg, deploy_cfg = cfgs
    tg = _FakeTelegram()

    runtime = EngineRuntime(
        strategies=[strategy_cfg],
        deploy_cfg=deploy_cfg,
        adapter_factory=_PMUpDownSpotMarkStubAdapter,
        subscriptions=[],
        exec_client_factory=lambda _a, _c, paper: PMClient(paper_mode=paper),
        telegram_factory=lambda _http: tg,
        klines_fetcher=lambda _ts_ns: 73_644.92,
    )
    runtime_task = asyncio.create_task(runtime.run())
    await asyncio.sleep(3.0)
    runtime.stop_event.set()
    await asyncio.wait_for(runtime_task, timeout=5.0)

    # Strike must still be captured (alert-only — no skip).
    q = runtime.market_state.question(909005)
    assert q is not None, "question 909005 not found in market_state"
    assert q.strike == 73_644.92, (
        f"strike should be 73644.92 (klines value), got {q.strike}"
    )
    [slot] = runtime.slots
    assert slot.dal.get_pm_strike(909005) == 73_644.92

    # A PMStrikeMismatch Telegram alert must have fired.
    assert any("divergence" in m.lower() for m in tg.messages), (
        f"no divergence alert in tg.messages: {tg.messages}"
    )
