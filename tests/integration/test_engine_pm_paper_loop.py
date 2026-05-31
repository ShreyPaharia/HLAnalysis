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
    candle ts (strike_ref_ts_ns). The engine must stamp the strike from the
    live reference mark at the open."""

    venue = "polymarket"

    def supports(self, *_a, **_k) -> bool:
        return True

    async def stream(self, _subs) -> AsyncIterator:
        now = time.time_ns()
        expiry_str = datetime.fromtimestamp(
            (now + 12 * 3600 * 1_000_000_000) / 1e9, tz=timezone.utc,
        ).strftime("%Y%m%d-%H%M")
        # Reference mark first — in production the bbo reference feed runs
        # continuously, so a mark is always present when a new market lists.
        # This is the value the engine should capture as the strike.
        yield MarkEvent(
            venue="binance", product_type=ProductType.SPOT,
            mechanism=Mechanism.CLOB, symbol="BTC",
            exchange_ts=now, local_recv_ts=now, mark_px=73_500.0,
        )
        # strike_ref_ts_ns ≈ now → within the engine's capture tolerance.
        yield QuestionMetaEvent(
            venue="polymarket", product_type=ProductType.PREDICTION_BINARY,
            mechanism=Mechanism.CLOB, symbol="YES_TOKEN",
            exchange_ts=now, local_recv_ts=now,
            question_idx=909002, named_outcome_idxs=[0, 1],
            keys=["class", "underlying", "expiry", "series_slug",
                  "yes_token_id", "no_token_id", "strike_ref_ts_ns"],
            values=["priceBinary", "BTC", expiry_str, "btc-up-or-down-daily",
                    "YES_TOKEN", "NO_TOKEN", str(now)],
        )
        await asyncio.sleep(2.5)


class _PMUpDownRestartStubAdapter(VenueAdapter):
    """A PM up/down market whose open is in the past (strike_ref_ts ≪ now) —
    simulates the post-restart case where the engine can no longer observe the
    open and must reload a persisted strike."""

    venue = "polymarket"

    def supports(self, *_a, **_k) -> bool:
        return True

    async def stream(self, _subs) -> AsyncIterator:
        now = time.time_ns()
        past_open = now - 6 * 3600 * 1_000_000_000  # 6h ago → outside tolerance
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
async def test_pm_updown_strike_captured_from_reference_at_open(cfgs):
    # An up/down PM market carries no static strike. The engine must stamp it
    # from the live reference mark at the open so the strategy can price it.
    strategy_cfg, deploy_cfg = cfgs
    fake_tg = _FakeTelegram()

    runtime = EngineRuntime(
        strategies=[strategy_cfg],
        deploy_cfg=deploy_cfg,
        adapter_factory=_PMUpDownStubAdapter,
        subscriptions=[],
        exec_client_factory=lambda _a, _c, paper: PMClient(paper_mode=paper),
        telegram_factory=lambda _http: fake_tg,
    )
    runtime_task = asyncio.create_task(runtime.run())
    await asyncio.sleep(3.0)
    runtime.stop_event.set()
    await asyncio.wait_for(runtime_task, timeout=5.0)

    q = runtime.market_state.question(909002)
    assert q is not None
    # Strike stamped from the reference mark (73_500), not left NaN.
    assert q.strike == 73_500.0
    # …and persisted for restart reuse.
    [slot] = runtime.slots
    assert slot.dal.get_pm_strike(909002) == 73_500.0


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
async def test_pm_updown_strike_backfilled_when_open_missed(cfgs):
    # Open was missed AND nothing persisted (fresh engine that started after the
    # market's open). The engine must backfill the strike from the historical
    # Binance close so the market is tradeable instead of skipped.
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
