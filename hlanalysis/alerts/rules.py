from __future__ import annotations

import asyncio
import html
import time
from typing import Mapping, Protocol


def _e(s) -> str:
    """Escape an untrusted value for HTML-mode Telegram messages.

    Tolerates None (e.g. ReconcileDrift.cloid is None for position-mismatch
    events that aren't tied to a single order). Without the None guard, every
    position-mismatch drift event would AttributeError its way out of the
    alert worker and silently drop from Telegram.
    """
    if s is None:
        return ""
    return html.escape(str(s), quote=False)

from loguru import logger

from ..engine.event_bus import EventBus
from ..engine.risk_events import (
    BusEvent, DailyLossHalt, Entry, Exit, FeedDown, FeedRecovered, FeedStale,
    KillSwitchActivated, NewQuestion, OrderRejected, OrderUnconfirmed,
    PMStrikeMismatch, ReconcileDrift, RedemptionTimeout, RiskHalt, RiskVeto,
    StaleDataHalt, StopLossTriggered,
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
        venue_by_alias: Mapping[str, str] | None = None,
    ) -> None:
        self._bus = bus
        self._tg = telegram
        self._dedupe_window_s = dedupe_window_s
        self._last_sent: dict[str, float] = {}
        # alias → short venue tag rendered in the alert prefix
        # (e.g. {"v31": "HL", "v31_pm": "PM"}). Empty mapping reverts to
        # the legacy alias-only prefix so single-venue deployments stay
        # visually unchanged.
        self._venue_by_alias = dict(venue_by_alias or {})

    async def run(self, sub: asyncio.Queue[BusEvent]) -> None:
        while True:
            ev = await sub.get()
            try:
                msg = self._format(ev)
                if msg is None:
                    continue
                key, text = msg
                # Prefix the Telegram message with the originating account
                # alias so v1 and v31 are visually distinct on the chat. When
                # a venue tag is configured for the alias we render
                # `[HL:v31]` so operators can tell HL and PM apart at a
                # glance. Cross-slot events (NewQuestion) carry an empty
                # alias and render unprefixed — preserves legacy behavior on
                # single-account deployments.
                alias = getattr(ev, "account_alias", "") or ""
                if alias:
                    venue = self._venue_by_alias.get(alias, "")
                    label = f"{venue}:{alias}" if venue else alias
                    text = f"<b>[{_e(label)}]</b> {text}"
                # Scope the dedupe key per account so two slots emitting
                # identically-shaped events (same RiskVeto reason on the same
                # symbol, etc.) each get their own alert rather than collapsing.
                if key is not None and alias:
                    key = f"{alias}|{key}"
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
            case FeedStale():
                return "feed_stale", (
                    f"🛑 <b>FEED STALE</b> — 0 market-data events in "
                    f"{ev.interval_seconds:.0f}s. Feed/ingest may be dead; "
                    f"open positions are unmanaged against a frozen feed."
                )
            case FeedDown():
                return "feed_down", (
                    f"🔌 <b>FEED DOWN</b> — market-data stream dropped "
                    f"(reconnecting; attempt {ev.consecutive_failures})."
                )
            case FeedRecovered():
                return "feed_recovered", "✅ <b>FEED RECOVERED</b> — ingest resumed."
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
                # Dedupe identical drift events (same case + question + cloid)
                # within the window. Without a key, every reconcile cycle
                # re-fires the same alert.
                key = f"drift:{ev.case}:{ev.question_idx}:{ev.cloid or ''}"
                return key, (
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
                # Map known reasons to emojis. Fall back to a generic close icon
                # for strategy-specific reasons (exit_safety_d_5m, exit_edge,
                # exit_time_stop, exit_take_profit, ...). Keeps Telegram
                # parseable even as strategies grow new exit branches.
                _EXIT_EMOJI = {
                    "settlement": "🏁",
                    "stop_loss": "🛑",
                    "exit_stop_loss": "🛑",
                    "manual": "↩️",
                    "exit_safety_d": "⚠️",
                    "exit_safety_d_5m": "⚠️",
                    "exit_edge": "📉",
                    "exit_time_stop": "⏱️",
                    "exit_take_profit": "💰",
                }
                emoji = _EXIT_EMOJI.get(ev.reason, "🔚")
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
            case OrderUnconfirmed():
                # Dedupe per cloid — same stalled order fires once per dedupe
                # window. The detection loop also tracks alerted cloids in-
                # memory so re-emission doesn't happen until status flips.
                return f"unconf:{ev.cloid}", (
                    f"⚠️ <b>ORDER UNCONFIRMED</b> cloid=<code>{_e(ev.cloid)}</code> "
                    f"{ev.side} {ev.size:g}@{ev.limit_price:.4f} "
                    f"age={ev.age_seconds:.0f}s"
                )
            case RedemptionTimeout():
                age_hours = ev.age_seconds / 3600.0
                sym_short = ev.symbol[:8]
                return f"redempt:{ev.question_idx}", (
                    f"⏰ <b>REDEMPTION TIMEOUT</b> q={ev.question_idx} "
                    f"sym=<code>{_e(sym_short)}...</code> qty={ev.qty:g} "
                    f"expected=${ev.expected_payout_usd:.2f} "
                    f"settled={age_hours:.1f}h ago"
                )
            case NewQuestion():
                lines = ["📣 <b>NEW MARKET</b>"]
                if ev.description:
                    lines.append(f"<i>{_e(ev.description)}</i>")
                if ev.klass:
                    lines.append(f"class={_e(ev.klass)}  legs={ev.leg_count}")
                lines.append(f"<code>q={ev.question_idx}</code>")
                return f"newq:{ev.question_idx}", "\n".join(lines)
            case PMStrikeMismatch():
                lines = [
                    f"⚠️ PM strike vs spot-mark divergence q={ev.question_idx}",
                    f"captured={ev.captured_strike:.2f}  mark={ev.reference_mark:.2f}  "
                    f"Δ={ev.divergence_bps:.1f}bps",
                    "(alert only — strike still trades)",
                ]
                return (ev.account_alias or None, "\n".join(lines))
            case _:
                return None
