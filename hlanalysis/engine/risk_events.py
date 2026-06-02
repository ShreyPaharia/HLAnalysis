from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    model_config = ConfigDict(frozen=True)
    ts_ns: int
    # Account/strategy slot that emitted this event. Empty when the publisher is
    # cross-slot (e.g. NewQuestion fires once globally on first sight). Alerts
    # prefix this onto the Telegram message so v1 / v31 are visually distinct.
    account_alias: str = ""


class RiskVeto(_Base):
    kind: Literal["risk_veto"] = "risk_veto"
    reason: str
    question_idx: int | None = None
    detail: dict[str, str] = Field(default_factory=dict)


class RiskHalt(_Base):
    kind: Literal["risk_halt"] = "risk_halt"
    reason: str


class StopLossTriggered(_Base):
    kind: Literal["stop_loss_triggered"] = "stop_loss_triggered"
    question_idx: int
    symbol: str
    qty: float
    trigger_px: float


class DailyLossHalt(_Base):
    kind: Literal["daily_loss_halt"] = "daily_loss_halt"
    realized_pnl: float
    cap: float


class StaleDataHalt(_Base):
    kind: Literal["stale_data_halt"] = "stale_data_halt"
    symbol: str
    age_seconds: float


class KillSwitchActivated(_Base):
    kind: Literal["kill_switch_activated"] = "kill_switch_activated"
    path: str


class ReconcileDrift(_Base):
    kind: Literal["reconcile_drift"] = "reconcile_drift"
    case: Literal["local_ghost", "venue_orphan", "state_mismatch", "position_mismatch"]
    cloid: str | None = None
    question_idx: int | None = None
    detail: dict[str, str] = Field(default_factory=dict)


class Entry(_Base):
    kind: Literal["entry"] = "entry"
    cloid: str
    question_idx: int
    symbol: str
    side: Literal["buy", "sell"]
    size: float
    price: float
    question_description: str = ""
    outcome_description: str = ""


class Exit(_Base):
    kind: Literal["exit"] = "exit"
    question_idx: int
    symbol: str
    qty: float
    realized_pnl: float
    # Free-form reason from the strategy (exit_safety_d, exit_edge, exit_time_stop,
    # exit_stop_loss, settlement, manual, ...). Telegram renderer maps known
    # prefixes to emojis and falls back to a generic close icon.
    reason: str
    question_description: str = ""
    outcome_description: str = ""


class NewQuestion(_Base):
    """Fired the first time the engine sees a question_idx in a QuestionMetaEvent.
    Used for rollover/new-market alerts on Telegram."""
    kind: Literal["new_question"] = "new_question"
    question_idx: int
    klass: str = ""
    description: str = ""
    expiry_ns: int = 0
    leg_count: int = 0


class OrderRejected(_Base):
    """Fired when HL responds with status=rejected on a place call. The risk
    gate's hard veto is already covered by RiskVeto; this captures venue-side
    rejections (min notional, invalid size, insufficient balance, etc.) that
    bypass the gate."""
    kind: Literal["order_rejected"] = "order_rejected"
    cloid: str
    question_idx: int
    symbol: str
    side: Literal["buy", "sell"]
    size: float
    price: float
    error: str = ""


class OrderUnconfirmed(_Base):
    """PM-specific: a place() returned status=open but the order has sat in
    flight past the unconfirmed threshold (default 30s) without a status
    change. Polymarket CLOB orders can stall under chain load; this gives
    operators an early warning so they can investigate before the position
    sizing assumption breaks down."""
    kind: Literal["order_unconfirmed"] = "order_unconfirmed"
    cloid: str
    symbol: str
    side: Literal["buy", "sell"]
    size: float
    limit_price: float
    age_seconds: float
    venue_oid: str = ""


class RedemptionTimeout(_Base):
    """PM-specific: a settled position hasn't been redeemed on-chain within
    the timeout window (default 6h after the settlement Exit fired). Without
    a Polygon RPC integration we can't actually verify USDC inflow, so this
    is a TIME-BASED watchdog only — operator verifies on PM UI and manually
    redeems if needed. expected_payout_usd is qty when realized_pnl > 0
    (winner) or 0 otherwise (loser)."""
    kind: Literal["redemption_timeout"] = "redemption_timeout"
    question_idx: int
    symbol: str
    qty: float
    settled_ts_ns: int
    age_seconds: float
    expected_payout_usd: float


class PMStrikeMismatch(_Base):
    """PM-specific: fired after strike capture when the fetched Binance spot
    1m close diverges materially from the live BTCUSDT_SPOT BBO mark. This is
    a cross-check between two independent data paths (REST klines vs. live feed).
    Alert-only — the strike is still persisted and the market still trades."""
    kind: Literal["pm_strike_mismatch"] = "pm_strike_mismatch"
    question_idx: int
    captured_strike: float
    reference_mark: float
    divergence_bps: float


class EngineHeartbeat(_Base):
    """Liveness pulse published once per heartbeat interval. Alert-silent by
    design (no Telegram) — its purpose is to give in-process sinks and any
    external uptime monitor a positive signal so the ABSENCE of heartbeats can
    be alerted on (SHR-43)."""
    kind: Literal["engine_heartbeat"] = "engine_heartbeat"
    events_ingested: int
    d_events: int
    n_questions: int


class FeedStale(_Base):
    """Published when zero market-data events were ingested over a full
    heartbeat interval while subscriptions are active — the feed/ingest is dead,
    not a calm market. Drives a Telegram alert (SHR-43)."""
    kind: Literal["feed_stale"] = "feed_stale"
    d_events: int
    interval_seconds: float


class FeedDown(_Base):
    """The market-data stream disconnected (raised or ended); the ingest loop is
    reconnecting with backoff (SHR-42). Published once per outage."""
    kind: Literal["feed_down"] = "feed_down"
    consecutive_failures: int


class FeedRecovered(_Base):
    """The market-data stream reconnected and delivered an event after a
    FeedDown (SHR-42)."""
    kind: Literal["feed_recovered"] = "feed_recovered"


BusEvent = Annotated[
    Union[
        RiskVeto, RiskHalt, StopLossTriggered, DailyLossHalt, StaleDataHalt,
        KillSwitchActivated, ReconcileDrift, Entry, Exit, NewQuestion,
        OrderRejected, OrderUnconfirmed, RedemptionTimeout, PMStrikeMismatch,
        EngineHeartbeat, FeedStale, FeedDown, FeedRecovered,
    ],
    Field(discriminator="kind"),
]
