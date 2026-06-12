"""SHR-41: blocking ExecutionClient REST calls must run off the event loop.

The engine runs a single asyncio event loop shared by WS market-data ingest,
the heartbeat, the stop-loss enforcer, reconcile, and every account slot. The
ExecutionClient methods (`place`, `open_orders`, `clearinghouse_state`,
`user_fills`, `realized_pnl_since`, `cancel`) are synchronous, `requests`-backed
SDK calls. Invoking them directly from a coroutine parks the WHOLE loop for the
duration of the (retry-wrapped) network round-trip — exactly when the venue is
congested.

These tests assert the offload structurally: a recording client captures the
thread each call runs on, and we assert it is NOT the event-loop thread. With a
direct synchronous call the method runs on the loop thread (idents equal) and
the test fails; wrapping in `asyncio.to_thread(...)` runs it on a worker thread
(idents differ) and the test passes. Thread-identity is deterministic — no
sleeps, no timing flakiness.
"""
from __future__ import annotations

import threading

import pytest

from hlanalysis.engine.config import (
    AllowlistEntry, GlobalRiskConfig, StrategyConfig,
)
from hlanalysis.engine.event_bus import EventBus
from hlanalysis.engine.exec_types import (
    ClearinghouseState, OpenOrderRow, OrderAck, PlaceRequest, UserFillRow,
)
from hlanalysis.engine.risk import RiskGate, RiskInputs
from hlanalysis.engine.router import Router
from hlanalysis.engine.state import StateDAL
from hlanalysis.strategy.types import (
    Action, BookState, Decision, OrderIntent, QuestionView,
)


class RecordingExecClient:
    """Implements the ExecutionClient Protocol. Records the OS thread ident of
    each blocking call so tests can assert the engine offloaded it off the
    event-loop thread. `paper_mode=False` so the engine takes the real
    (would-be-blocking) code path, not any paper shortcut."""

    paper_mode = False

    def __init__(self) -> None:
        self.place_thread: int | None = None
        self.cancel_thread: int | None = None
        self.open_orders_thread: int | None = None
        self.state_thread: int | None = None
        self.fills_thread: int | None = None
        self.pnl_thread: int | None = None

    def place(self, req: PlaceRequest) -> OrderAck:
        self.place_thread = threading.get_ident()
        return OrderAck(
            cloid=req.cloid, venue_oid="v1", status="filled",
            fill_price=req.price, fill_size=req.size,
        )

    def cancel(self, *, cloid: str, symbol: str) -> bool:
        self.cancel_thread = threading.get_ident()
        return True

    def open_orders(self) -> list[OpenOrderRow]:
        self.open_orders_thread = threading.get_ident()
        return []

    def clearinghouse_state(self) -> ClearinghouseState:
        self.state_thread = threading.get_ident()
        return ClearinghouseState(positions=(), account_value_usd=0.0)

    def user_fills(self, *, since_ts_ns: int = 0) -> list[UserFillRow]:
        self.fills_thread = threading.get_ident()
        return []

    def realized_pnl_since(self, since_ts_ns: int, *,
                           outcome_only: bool = False) -> float:
        self.pnl_thread = threading.get_ident()
        self.pnl_outcome_only = outcome_only
        return 0.0


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
            min_recent_volume_usd=1000, stale_data_halt_seconds=5,
            reconcile_interval_seconds=60,
        )},
    )


def _q() -> QuestionView:
    return QuestionView(
        question_idx=42, yes_symbol="@30", no_symbol="@31",
        strike=80_000.0, expiry_ns=10_000_000_000_000_001 + 600_000_000_000,
        underlying="BTC", klass="priceBinary", period="1h",
    )


def _decision_enter() -> Decision:
    return Decision(
        action=Action.ENTER,
        intents=(OrderIntent(
            question_idx=42, symbol="@30", side="buy", size=10.0,
            limit_price=0.95, cloid="hla-router-1", time_in_force="ioc",
        ),),
    )


def _approval_inputs() -> RiskInputs:
    return RiskInputs(
        question=_q(),
        question_fields={"class": "priceBinary", "underlying": "BTC", "period": "1h"},
        reference_price=80_300.0,
        book=BookState(symbol="@30", bid_px=0.94, bid_sz=10.0, ask_px=0.95,
                       ask_sz=10.0, last_trade_ts_ns=10_000_000_000_000_000,
                       last_l2_ts_ns=10_000_000_000_000_000),
        recent_volume_usd=5_000.0,
        positions=[],
        live_orders_total_notional=0.0,
        realized_pnl_today=0.0,
        kill_switch_active=False,
        last_reconcile_ns=10_000_000_000_000_000,
        now_ns=10_000_000_000_000_001,
    )


def _build_runtime_with_recording(tmp_path):
    """Build a real EngineRuntime whose single HL slot uses a RecordingExecClient,
    so the runtime's venue-IO helpers can be asserted to run off the loop thread.
    Returns (runtime, slot)."""
    from hlanalysis.engine.config import (
        AlertsConfig, DeployConfig, HyperliquidAccount, TelegramConfig,
    )
    from hlanalysis.engine.runtime import EngineRuntime

    entry = AllowlistEntry(
        match={"class": "priceBinary", "underlying": "BTC"},
        max_position_usd=100, stop_loss_pct=None, tte_min_seconds=0,
        tte_max_seconds=7200, price_extreme_threshold=0.85,
        distance_from_strike_usd_min=0, vol_max=100,
        vol_lookback_seconds=3600, vol_sampling_dt_seconds=60,
    )
    strat = StrategyConfig(
        name="late_resolution", account_alias="v1", paper_mode=True,
        strategy_type="late_resolution", reference_symbol="BTC",
        allowlist=[entry], blocklist_question_idxs=[], defaults=entry,
        **{"global": GlobalRiskConfig(
            max_total_inventory_usd=500, max_concurrent_positions=5,
            daily_loss_cap_usd=200, max_strike_distance_pct=50,
            min_recent_volume_usd=100, stale_data_halt_seconds=30,
            reconcile_interval_seconds=60,
        )},
    )
    deploy = DeployConfig(
        env="dev",
        accounts={"v1": HyperliquidAccount(
            account_address="0x0", api_secret_key="0x0",
            base_url="https://api.hyperliquid.xyz",
        )},
        alerts=AlertsConfig(telegram=TelegramConfig(bot_token="x", chat_id="y")),
        state_db_path=str(tmp_path / "state.db"),
        kill_switch_path=str(tmp_path / "halt"),
    )
    rt = EngineRuntime(
        strategies=[strat], deploy_cfg=deploy,
        adapter_factory=lambda: None, subscriptions=[],
        exec_client_factory=lambda _a, _c, _p: RecordingExecClient(),
    )
    rt.slots = [rt._build_slot(strat)]
    return rt, rt.slots[0]


@pytest.mark.asyncio
async def test_venue_snapshot_runs_off_event_loop_thread(tmp_path):
    rt, slot = _build_runtime_with_recording(tmp_path)
    client = slot.exec_client
    loop_thread = threading.get_ident()

    await rt._venue_snapshot(slot)

    for name, tid in (
        ("open_orders", client.open_orders_thread),
        ("clearinghouse_state", client.state_thread),
        ("user_fills", client.fills_thread),
    ):
        assert tid is not None, f"{name}() was never called"
        assert tid != loop_thread, (
            f"exec_client.{name}() ran on the event-loop thread — must be "
            f"offloaded via asyncio.to_thread()."
        )


@pytest.mark.asyncio
async def test_realized_pnl_today_runs_off_event_loop_thread(tmp_path):
    rt, slot = _build_runtime_with_recording(tmp_path)
    client = slot.exec_client
    loop_thread = threading.get_ident()

    pnl = await rt._realized_pnl_today(slot, now_ns=10_000_000_000_000_000)

    assert pnl == 0.0
    assert client.pnl_thread is not None, "realized_pnl_since() was never called"
    assert client.pnl_thread != loop_thread, (
        "exec_client.realized_pnl_since() ran on the event-loop thread — must "
        "be offloaded via asyncio.to_thread()."
    )


@pytest.mark.asyncio
async def test_daily_loss_gate_reads_outcome_only_pnl(tmp_path):
    """The daily-loss gate must source realized PnL with outcome_only=True so the
    operator's NON-strategy manual trading on the same HL account (perps, ordinary
    spot — e.g. a −$394 HYPE day) cannot false-halt (or mask losses in) the
    strategy. The strategy only trades HIP-4 binary outcome markets (coin "#N")."""
    rt, slot = _build_runtime_with_recording(tmp_path)
    client = slot.exec_client

    await rt._realized_pnl_today(slot, now_ns=10_000_000_000_000_000)

    assert client.pnl_outcome_only is True, (
        "daily-loss gate summed ALL coins (incl. manual perp/spot), not just the "
        "strategy's '#' outcome markets"
    )


@pytest.mark.asyncio
async def test_router_place_runs_off_event_loop_thread(tmp_path):
    dal = StateDAL(tmp_path / "state.db")
    dal.run_migrations()
    bus = EventBus()
    client = RecordingExecClient()
    cfg = _strategy_cfg()
    router = Router(dal=dal, gate=RiskGate(cfg), bus=bus, exec_client=client,
                    strategy_cfg=cfg)

    loop_thread = threading.get_ident()
    await router.handle(_decision_enter(), inputs=_approval_inputs(), now_ns=2)

    assert client.place_thread is not None, "place() was never called"
    assert client.place_thread != loop_thread, (
        "exec_client.place() ran on the event-loop thread — a slow/flaky REST "
        "call here parks the whole engine. It must be offloaded via "
        "asyncio.to_thread()."
    )
