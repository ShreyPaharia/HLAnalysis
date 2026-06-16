from __future__ import annotations

import asyncio
import html
import time
from collections.abc import Mapping
from typing import Protocol


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
    BusEvent,
    DailyLossHalt,
    Entry,
    Exit,
    FeedDown,
    FeedRecovered,
    FeedStale,
    KillSwitchActivated,
    MemoryHalt,
    NewQuestion,
    OrderRejected,
    OrderUnconfirmed,
    PMStrikeMismatch,
    ReconcileDrift,
    RedemptionTimeout,
    RiskHalt,
    RiskVeto,
    StaleDataHalt,
    StopLossTriggered,
)


class _TelegramLike(Protocol):
    async def send(self, text: str, *, markdown: bool = True) -> bool: ...


# RiskVeto reasons that reflect ordinary *market conditions* — the gate is
# correctly declining to trade an illiquid/quiet/mispriced market — rather than
# an engine or risk-state problem. They re-fire on every scan when a slot is
# pointed at a thin market (the v31_pm_eth_ms ETH-multistrike incident
# 2026-06-12 emitted ≈240/h), require no operator action, and self-heal when the
# market moves. They are still logged (`logger.info` in the router) and journaled
# (SHR-83), so diagnostics keep full visibility — only the Telegram page is
# suppressed. Mirrors the OrderRejected FAK no-match suppression below.
#
# Reasons NOT in this set page by default (fail-safe): engine/risk-state vetoes
# (daily_loss_cap, kill_switch_active, max_*_usd / max_concurrent_positions,
# stale_reconcile, stale_reference, opposite_leg_held, size_invalid) and any
# newly-added reason stay visible until explicitly classified as benign.
_BENIGN_VETO_REASONS = frozenset(
    {
        "low_volume",  # market notional below floor (thin book)
        "stale_data",  # the trading leg's book has gone quiet (PM favorites do)
        "strike_distance",  # favorite too far from the reference price
        "tte_out_of_window",  # outside the slot's time-to-expiry window
        "depth_walk_no_fill",  # no level marketable at our limit (book ticked away)
        "depth_walk_slip",  # at-limit fill would exceed the slippage cap
        "order_below_min_notional",  # effective size clamped below the min-notional floor
        "post_exit_cooldown",  # churn guard — re-entry too soon after an exit
        # Slot at its concurrent-position cap while the strategy keeps proposing
        # entries every scan — the cap is the gate working as designed and
        # re-fires every tick with no operator action possible (incident
        # 2026-06-16 v31_pm_eth_ms q=2026151396). An untracked position eating a
        # slot still surfaces via the venue_orphan/qty_mismatch drift alerts, so
        # suppressing this page loses no visibility.
        "max_concurrent_positions",
        "settled",  # market already cash-settled
        "allowlist_no_match",  # question not in this slot's allowlist
        "blocklist",  # question explicitly blocked
    }
)


# PM reconcile drift resolutions that are informational and PERSISTENT: the
# condition (a venue position we don't track, a local position the data-api
# can't see, a qty that disagrees with the laggy indexer) re-fires on every
# reconcile cycle — every 15s in prod — for the entire lifetime of the position,
# because PM is alert-only by design (we never auto-adopt/vanish from the
# flapping data-api live; see reconcile.py). The default 60s dedupe window only
# collapses a minute of repeats, so a position that stays diverged for hours
# pages ~60×/hour indefinitely (incident 2026-06-12 v1_pm q=700064348:
# venue_orphan_alert_only on a 51-share 0.98 favorite). These get the long
# `persistent_dedupe_window_s` so the operator is paged once, then reminded only
# periodically — kept visible (not silenced) because an untracked real position
# still needs eventual manual resolution or settlement. Mirrors the benign
# RiskVeto / throttled stale-halt suppression.
_PERSISTENT_DRIFT_RESOLUTIONS = frozenset(
    {
        "venue_orphan_alert_only",
        "venue_absent_alert_only",
        "qty_mismatch_alert_only",
        # A venue position whose symbol can't be mapped to a question_idx yet
        # (no QuestionMeta ingested) — the reconciler can't invent a PK to adopt
        # it, so it re-fires every cycle for the orphan's whole lifetime, exactly
        # like venue_orphan_alert_only (incident 2026-06-16 v31_pm_eth_ms, a
        # 60.02-share CTF token). Kept visible (still needs eventual manual
        # resolution / settlement) but on the long window, not every cycle.
        "venue_orphan_unattributed",
    }
)


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
        persistent_dedupe_window_s: int = 3600,
        venue_by_alias: Mapping[str, str] | None = None,
    ) -> None:
        self._bus = bus
        self._tg = telegram
        self._dedupe_window_s = dedupe_window_s
        # Long window for persistent, no-immediate-action PM alert-only drift
        # (see _PERSISTENT_DRIFT_RESOLUTIONS). Defaults to 1h: one page when the
        # divergence appears, then an hourly heartbeat instead of every cycle.
        self._persistent_dedupe_window_s = persistent_dedupe_window_s
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

    def _window_for_key(self, key: str) -> float:
        # Persistent PM alert-only drift carries its resolution as the key's
        # final `:`-segment; those get the long window so they don't flood.
        if any(key.endswith(f":{r}") for r in _PERSISTENT_DRIFT_RESOLUTIONS):
            return self._persistent_dedupe_window_s
        return self._dedupe_window_s

    def _is_deduped(self, key: str) -> bool:
        last = self._last_sent.get(key)
        if last is None:
            return False
        return (time.monotonic() - last) < self._window_for_key(key)

    def _format(self, ev: BusEvent) -> tuple[str | None, str] | None:
        match ev:
            case KillSwitchActivated():
                return None, f"<b>KILL SWITCH ACTIVATED</b>\nPath: <code>{_e(ev.path)}</code>"
            case DailyLossHalt():
                return None, (f"<b>DAILY LOSS HALT</b>\nRealized: ${ev.realized_pnl:.2f} / Cap: ${ev.cap:.2f}")
            case MemoryHalt():
                rss_mb = ev.rss_kb / 1024
                ceil_mb = ev.ceiling_kb / 1024
                return None, (
                    f"🧠 <b>MEMORY SELF-HALT</b> (W1.9)\n"
                    f"RSS {rss_mb:.0f} MB exceeds ceiling {ceil_mb:.0f} MB — "
                    f"all slots halted to prevent OOM kill mid-position.\n"
                    f"Remove kill-switch flags and restart engine to resume."
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
                    f"<b>STALE DATA HALT</b> <code>{_e(ev.symbol)}</code> ({ev.age_seconds:.1f}s)"
                )
            case RiskHalt():
                return f"halt:{ev.reason}", f"<b>RISK HALT</b> {_e(ev.reason)}"
            case RiskVeto():
                # Market-condition vetoes are the gate working as designed on a
                # thin/quiet market — they re-fire every scan and require no
                # operator action. Suppress from Telegram (still logged +
                # journaled in the router) so they don't flood the channel.
                if ev.reason in _BENIGN_VETO_REASONS:
                    logger.debug(
                        "suppressing benign risk veto reason={} q={} detail={}",
                        ev.reason,
                        ev.question_idx,
                        ev.detail,
                    )
                    return None
                # Include question_idx in the dedupe key so identical reasons
                # on different questions don't collapse into one alert.
                return f"veto:{ev.reason}:{ev.question_idx}", (
                    f"<i>risk veto</i> {_e(ev.reason)} q={ev.question_idx} {_e(str(ev.detail))}"
                )
            case StopLossTriggered():
                return None, (
                    f"<b>STOP-LOSS</b> q={ev.question_idx} <code>{_e(ev.symbol)}</code> "
                    f"qty={ev.qty} trigger=${ev.trigger_px:.4f}"
                )
            case ReconcileDrift():
                # Dedupe identical drift events (same case + question + cloid +
                # resolution) within the window. Without a key, every reconcile
                # cycle re-fires the same alert. The resolution is the key's
                # final segment so _window_for_key can route persistent PM
                # alert-only drift to the long window (kills the every-cycle
                # flood) while loud/actionable drift keeps the short window.
                resolution = (ev.detail or {}).get("resolution", "")
                key = f"drift:{ev.case}:{ev.question_idx}:{ev.cloid or ''}:{resolution}"
                return key, (
                    f"<b>DRIFT</b> {_e(ev.case)} cloid={_e(ev.cloid)} q={ev.question_idx} {_e(str(ev.detail))}"
                )
            case Entry():
                notional = ev.size * ev.price
                lines = ["🟢 <b>ENTRY</b>"]
                if ev.question_description:
                    lines.append(f"<i>{_e(ev.question_description)}</i>")
                if ev.outcome_description:
                    lines.append(f"<b>{_e(ev.outcome_description)}</b>")
                lines.append(f"{ev.side.upper()} {ev.size:g} @ ${ev.price:.4f}  (notional ${notional:,.2f})")
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
                # A marketable IOC order that finds no resting match is killed by
                # the venue:
                #   PM → "no orders found to match with FAK order. FAK orders are
                #         partially filled or killed if no match is found."
                #   HL → "Order could not immediately match against any resting
                #         orders." (incident 2026-06-16 v1/v31 IOC through a thin
                #         book)
                # Both are expected microstructure — the book ticked past our
                # limit between read and send — and self-heal: the next scan
                # re-prices and the entry fills. Don't page on either. A genuine
                # pathological case (price never marketable) trips the
                # per-(question, side) reject circuit-breaker (SHR-45), which
                # alerts separately, so suppressing the single reject is safe.
                _err = (ev.error or "").lower()
                if "no orders found to match" in _err or "could not immediately match" in _err:
                    logger.debug(
                        "suppressing self-healing IOC no-match reject q={} sym={}",
                        ev.question_idx,
                        _e(ev.symbol),
                    )
                    return None
                notional = ev.size * ev.price
                lines = ["❌ <b>ORDER REJECTED</b>"]
                lines.append(f"{ev.side.upper()} {ev.size:g} @ ${ev.price:.4f}  (notional ${notional:,.2f})")
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
                    f"captured={ev.captured_strike:.2f}  mark={ev.reference_mark:.2f}  Δ={ev.divergence_bps:.1f}bps",
                    "(alert only — strike still trades)",
                ]
                return (ev.account_alias or None, "\n".join(lines))
            case _:
                return None
