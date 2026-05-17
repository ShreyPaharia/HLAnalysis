from __future__ import annotations

import asyncio
import html
import time
from typing import Protocol


def _e(s: str) -> str:
    """Escape an untrusted string for HTML-mode Telegram messages."""
    return html.escape(s, quote=False)

from loguru import logger

from ..engine.event_bus import EventBus
from ..engine.risk_events import (
    BusEvent, DailyLossHalt, Entry, Exit, KillSwitchActivated, NewQuestion,
    OrderRejected, ReconcileDrift, RiskHalt, RiskVeto, StaleDataHalt,
    StopLossTriggered,
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
                return None, f"<b>KILL SWITCH ACTIVATED</b>\nPath: <code>{_e(ev.path)}</code>"
            case DailyLossHalt():
                return None, (
                    f"<b>DAILY LOSS HALT</b>\n"
                    f"Realized: ${ev.realized_pnl:.2f} / Cap: ${ev.cap:.2f}"
                )
            case StaleDataHalt():
                return f"stale:{ev.symbol}", (
                    f"<b>STALE DATA HALT</b> <code>{_e(ev.symbol)}</code> "
                    f"({ev.age_seconds:.1f}s)"
                )
            case RiskHalt():
                return f"halt:{ev.reason}", f"<b>RISK HALT</b> {_e(ev.reason)}"
            case RiskVeto():
                # Include question_idx in the dedupe key so identical reasons
                # on different questions don't collapse into one alert.
                return f"veto:{ev.reason}:{ev.question_idx}", (
                    f"<i>risk veto</i> {_e(ev.reason)} "
                    f"q={ev.question_idx} {_e(str(ev.detail))}"
                )
            case StopLossTriggered():
                return None, (
                    f"<b>STOP-LOSS</b> q={ev.question_idx} <code>{_e(ev.symbol)}</code> "
                    f"qty={ev.qty} trigger=${ev.trigger_px:.4f}"
                )
            case ReconcileDrift():
                return None, (
                    f"<b>DRIFT</b> {_e(ev.case)} cloid={_e(ev.cloid)} "
                    f"q={ev.question_idx} {_e(str(ev.detail))}"
                )
            case Entry():
                notional = ev.size * ev.price
                lines = ["🟢 <b>ENTRY</b>"]
                if ev.question_description:
                    lines.append(f"<i>{_e(ev.question_description)}</i>")
                if ev.outcome_description:
                    lines.append(f"<b>{_e(ev.outcome_description)}</b>")
                lines.append(
                    f"{ev.side.upper()} {ev.size:g} @ ${ev.price:.4f}  "
                    f"(notional ${notional:,.2f})"
                )
                lines.append(f"<code>q={ev.question_idx}</code> <code>{_e(ev.symbol)}</code>")
                # Dedup by cloid so reconciler-driven double-Entry collapses.
                return f"entry:{ev.cloid}", "\n".join(lines)
            case Exit():
                emoji = {"settlement": "🏁", "stop_loss": "🛑", "manual": "↩️"}.get(ev.reason, "🔚")
                pnl_sign = "+" if ev.realized_pnl >= 0 else ""
                lines = [f"{emoji} <b>EXIT</b> ({_e(ev.reason)})"]
                if ev.question_description:
                    lines.append(f"<i>{_e(ev.question_description)}</i>")
                if ev.outcome_description:
                    lines.append(f"<b>{_e(ev.outcome_description)}</b>")
                lines.append(f"qty={ev.qty:g}  PnL={pnl_sign}${ev.realized_pnl:.2f}")
                lines.append(f"<code>q={ev.question_idx}</code> <code>{_e(ev.symbol)}</code>")
                return f"exit:{ev.question_idx}:{ev.reason}", "\n".join(lines)
            case OrderRejected():
                notional = ev.size * ev.price
                lines = ["❌ <b>ORDER REJECTED</b>"]
                lines.append(
                    f"{ev.side.upper()} {ev.size:g} @ ${ev.price:.4f}  "
                    f"(notional ${notional:,.2f})"
                )
                lines.append(f"<code>q={ev.question_idx}</code> <code>{_e(ev.symbol)}</code>")
                lines.append(f"<i>{_e(ev.error) or 'no_error_field'}</i>")
                # Dedupe by error string + symbol — repeated identical rejects
                # on the same leg (e.g. min-notional retries) collapse, but a
                # new error or new symbol fires a fresh alert.
                return f"rej:{ev.symbol}:{ev.error}", "\n".join(lines)
            case NewQuestion():
                lines = ["📣 <b>NEW MARKET</b>"]
                if ev.description:
                    lines.append(f"<i>{_e(ev.description)}</i>")
                if ev.klass:
                    lines.append(f"class={_e(ev.klass)}  legs={ev.leg_count}")
                lines.append(f"<code>q={ev.question_idx}</code>")
                return f"newq:{ev.question_idx}", "\n".join(lines)
            case _:
                return None
