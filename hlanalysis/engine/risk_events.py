from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    model_config = ConfigDict(frozen=True)
    ts_ns: int


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
    reason: Literal["settlement", "stop_loss", "manual"]
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


BusEvent = Annotated[
    Union[
        RiskVeto, RiskHalt, StopLossTriggered, DailyLossHalt, StaleDataHalt,
        KillSwitchActivated, ReconcileDrift, Entry, Exit, NewQuestion,
    ],
    Field(discriminator="kind"),
]
