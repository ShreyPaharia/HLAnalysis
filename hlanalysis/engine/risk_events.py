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


BusEvent = Annotated[
    Union[
        RiskVeto, RiskHalt, StopLossTriggered, DailyLossHalt, StaleDataHalt,
        KillSwitchActivated, ReconcileDrift, Entry, Exit, NewQuestion,
        OrderRejected, OrderUnconfirmed,
    ],
    Field(discriminator="kind"),
]
