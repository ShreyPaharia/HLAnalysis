from __future__ import annotations

import asyncio
import time
from typing import Protocol

from loguru import logger

from ..engine.event_bus import EventBus
from ..engine.risk_events import (
    BusEvent, DailyLossHalt, Entry, Exit, KillSwitchActivated, ReconcileDrift,
    RiskHalt, RiskVeto, StaleDataHalt, StopLossTriggered,
)


class _TelegramLike(Protocol):
    async def send(self, text: str, *, markdown: bool = True) -> bool: ...


class AlertRules:
    """Subscribes to the EventBus and forwards a curated subset to Telegram.

    Spec §8.1 message set, with dedupe windows per kind.
    """

    def __init__(
        self,
        *,
        bus: EventBus,
        telegram: _TelegramLike,
        dedupe_window_s: int = 60,
    ) -> None:
        self._bus = bus
        self._tg = telegram
        self._dedupe_window_s = dedupe_window_s
        self._last_sent: dict[str, float] = {}

    async def run(self, sub: asyncio.Queue[BusEvent]) -> None:
        while True:
            ev = await sub.get()
            try:
                msg = self._format(ev)
                if msg is None:
                    continue
                key, text = msg
                if key is not None and self._is_deduped(key):
                    continue
                ok = await self._tg.send(text)
                if ok and key is not None:
                    self._last_sent[key] = time.monotonic()
            except Exception as e:
                logger.exception("alert rule error: {}", e)

    def _is_deduped(self, key: str) -> bool:
        last = self._last_sent.get(key)
        if last is None:
            return False
        return (time.monotonic() - last) < self._dedupe_window_s

    def _format(self, ev: BusEvent) -> tuple[str | None, str] | None:
        match ev:
            case KillSwitchActivated():
                return None, f"*KILL SWITCH ACTIVATED*\nPath: `{ev.path}`"
            case DailyLossHalt():
                return None, (
                    f"*DAILY LOSS HALT*\n"
                    f"Realized: ${ev.realized_pnl:.2f} / Cap: ${ev.cap:.2f}"
                )
            case StaleDataHalt():
                return f"stale:{ev.symbol}", (
                    f"*STALE DATA HALT* `{ev.symbol}` "
                    f"({ev.age_seconds:.1f}s)"
                )
            case RiskHalt():
                return f"halt:{ev.reason}", f"*RISK HALT* {ev.reason}"
            case RiskVeto():
                return f"veto:{ev.reason}", (
                    f"_risk veto_ {ev.reason} "
                    f"q={ev.question_idx} {ev.detail}"
                )
            case StopLossTriggered():
                return None, (
                    f"*STOP-LOSS* q={ev.question_idx} `{ev.symbol}` "
                    f"qty={ev.qty} trigger=${ev.trigger_px:.4f}"
                )
            case ReconcileDrift():
                return None, (
                    f"*DRIFT* {ev.case} cloid={ev.cloid} "
                    f"q={ev.question_idx} {ev.detail}"
                )
            case Entry():
                return None, (
                    f"*ENTRY* q={ev.question_idx} `{ev.symbol}` {ev.side} "
                    f"sz={ev.size} @ {ev.price:.4f}"
                )
            case Exit():
                return None, (
                    f"*EXIT* q={ev.question_idx} `{ev.symbol}` qty={ev.qty} "
                    f"PnL=${ev.realized_pnl:.2f} ({ev.reason})"
                )
            case _:
                return None
