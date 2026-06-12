"""SHR-85 — sim state/halt replay + daily-loss-cap + inventory caps.

The backtest used to trade through periods the *live* engine was halted, so it
over-traded versus live (a prime suspect for "sim trades, live didn't" — e.g. a
market where live did zero fills because a feed was stale all day). This module
makes the sim stop entering when live would have stopped.

Two cleanly-separated pieces, both pure and injection-testable (no hftbacktest,
no live trade journal):

1. The **entry gate** (:func:`entry_veto`) — mirrors the entry-only caps of the
   live ``engine.risk.RiskGate``: halt-window suppression, the ``daily_loss_cap``
   slot-halt over the engine's daily window, and the ``max_total_inventory_usd`` /
   ``max_concurrent_positions`` inventory caps. Exits are never passed through it
   (the live gate also exempts exits), so a held position can always be closed.

2. The **loader** (:func:`load_halt_windows`) — turns a stream of engine
   halt/event-log records (anything exposing ``ts_ns`` + ``kind``) into
   ``HaltWindow``s. It is decoupled from the engine's pydantic event classes and
   from the Spec-3 trade journal so the suppression core stays testable with
   injected windows; the runner consumes a plain ``list[HaltWindow]``.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from hlanalysis.risk.caps import (
    concurrent_cap_exceeded as _concurrent_cap_exceeded,
)
from hlanalysis.risk.caps import (
    daily_loss_exceeded as _daily_loss_exceeded,
)
from hlanalysis.risk.caps import (
    daily_window_start_ns as _shared_daily_window_start_ns,
)
from hlanalysis.risk.caps import (
    inventory_cap_exceeded as _inventory_cap_exceeded,
)

# ---------------------------------------------------------------------------
# Halt windows
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HaltWindow:
    """A half-open ``[start_ns, end_ns)`` interval during which live was halted.

    ``reason`` carries the originating halt kind (``stale_data_halt``,
    ``daily_loss_halt``, ``memory_halt``, …) for diagnostics.
    """

    start_ns: int
    end_ns: int
    reason: str

    def contains(self, ts_ns: int) -> bool:
        return self.start_ns <= ts_ns < self.end_ns


def in_halt_window(
    windows: Sequence[HaltWindow], now_ns: int
) -> HaltWindow | None:
    """Return the first window containing ``now_ns``, or ``None``."""
    for w in windows:
        if w.contains(now_ns):
            return w
    return None


# ---------------------------------------------------------------------------
# Daily window boundary (mirror of Scanner._daily_window_start_ns)
# ---------------------------------------------------------------------------


def daily_window_start_ns(now_ns: int, *, hour: int) -> int:
    """Most-recent ``HH:00:00`` UTC boundary at-or-before ``now_ns``.

    Re-exports ``hlanalysis.risk.caps.daily_window_start_ns`` so callers that
    import from this module continue to work unchanged.  The canonical
    implementation now lives in the shared ``hlanalysis.risk`` package so the
    engine scanner, the live risk gate, and this sim module all share one copy.
    """
    return _shared_daily_window_start_ns(now_ns, hour=hour)


# ---------------------------------------------------------------------------
# Entry gate — mirrors engine.risk.RiskGate entry-only caps
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SimRiskCaps:
    """Entry-only risk caps replayed in the sim. ``None`` disables a cap.

    Field names mirror ``engine.config.GlobalRiskConfig`` so the live config can
    be lifted into the sim directly.
    """

    daily_loss_cap_usd: float | None = None
    daily_window_start_hour_utc: int = 0
    max_total_inventory_usd: float | None = None
    max_concurrent_positions: int | None = None


@dataclass(frozen=True, slots=True)
class EntryGateInputs:
    """Per-entry snapshot the gate reads. ``realized_pnl_window`` is the realized
    PnL accumulated since the current daily window's start (the runner passes the
    window *floor* so a recovered-then-redipped slot stays latched-halted, the way
    the live kill-switch latch does)."""

    now_ns: int
    intent_notional: float          # size * limit_price of the proposed entry
    held_inventory_usd: float       # sum |qty|*avg_entry across held positions
    n_held_positions: int
    is_topup: bool                  # entry targets an already-held position
    realized_pnl_window: float


def entry_veto(
    caps: SimRiskCaps,
    halt_windows: Sequence[HaltWindow],
    inp: EntryGateInputs,
) -> str | None:
    """Return a veto reason if this entry would be blocked, else ``None``.

    Ordering mirrors the live ``RiskGate.check_pre_trade`` entry path: engine
    health (halt windows) → daily-loss cap → global inventory cap → concurrent
    cap. The size/allowlist/TTE/volume checks are the runner's concern; this gate
    only adds the state/halt-replay caps SHR-85 introduces.
    """
    # Engine health: replayed halt windows (stale_data, daily_loss, reject
    # breaker, dust-block, restart_blocked, OOM gaps).
    w = in_halt_window(halt_windows, inp.now_ns)
    if w is not None:
        return f"halt_window:{w.reason}"

    # Daily loss cap (entry-only; exits are never gated).
    if _daily_loss_exceeded(inp.realized_pnl_window, caps.daily_loss_cap_usd):
        return "daily_loss_cap"

    # Global inventory cap: held notional + this entry must stay under the cap.
    if _inventory_cap_exceeded(
        inp.held_inventory_usd, inp.intent_notional, caps.max_total_inventory_usd
    ):
        return "max_total_inventory"

    # Concurrent-positions cap: a NEW position past the cap is blocked; a top-up
    # to an already-held slot is allowed (matches the live gate).
    if _concurrent_cap_exceeded(
        inp.n_held_positions, inp.is_topup, caps.max_concurrent_positions
    ):
        return "max_concurrent_positions"

    return None


# ---------------------------------------------------------------------------
# Loader — engine halt/event log → halt windows
# ---------------------------------------------------------------------------


# Halt-start kinds whose clear is an explicit feed-recovery event.
_FEED_HALT_KINDS = frozenset({"stale_data_halt", "feed_stale", "feed_down"})
_FEED_CLEAR_KIND = "feed_recovered"
# Halt-start kinds that suppress entries to the end of the current daily window.
_DAILY_HALT_KINDS = frozenset({"daily_loss_halt"})
# Halt-start kinds with no explicit clear event in the log — reject circuit
# breaker, dust-block, restart_blocked, OOM restart gaps — use a fixed fallback
# duration so the sim still sits out a plausible window.
_FALLBACK_HALT_KINDS = frozenset(
    {"memory_halt", "risk_halt", "kill_switch_activated"}
)


def _ev_field(ev: object, name: str):
    if isinstance(ev, Mapping):
        return ev.get(name)
    return getattr(ev, name, None)


def load_halt_windows(
    events: Iterable[object],
    *,
    fallback_duration_ns: int,
    daily_window_start_hour_utc: int = 0,
) -> list[HaltWindow]:
    """Build halt windows from engine halt/event-log records.

    ``events`` is any iterable of records exposing ``ts_ns`` and ``kind`` (dicts
    or the engine's pydantic event objects both work). Recognised halt-start
    kinds become windows:

    - feed halts (``stale_data_halt`` / ``feed_stale`` / ``feed_down``) pair with
      the next ``feed_recovered``; if none follows, the fallback duration is used.
    - ``daily_loss_halt`` suppresses to the end of its daily window
      (``daily_window_start_hour_utc``).
    - reject-breaker / dust-block / restart / OOM halts (``memory_halt`` /
      ``risk_halt`` / ``kill_switch_activated``) use ``fallback_duration_ns``.

    Non-halt kinds (entries, heartbeats, …) are ignored. Windows are returned in
    ``start_ns`` order.
    """
    rows = sorted(
        (
            (int(_ev_field(ev, "ts_ns")), str(_ev_field(ev, "kind")))
            for ev in events
            if _ev_field(ev, "ts_ns") is not None
            and _ev_field(ev, "kind") is not None
        ),
        key=lambda r: r[0],
    )

    windows: list[HaltWindow] = []
    for i, (ts_ns, kind) in enumerate(rows):
        if kind in _FEED_HALT_KINDS:
            end_ns = ts_ns + fallback_duration_ns
            for ts2, kind2 in rows[i + 1:]:
                if kind2 == _FEED_CLEAR_KIND:
                    end_ns = ts2
                    break
            windows.append(HaltWindow(ts_ns, end_ns, kind))
        elif kind in _DAILY_HALT_KINDS:
            window_start = daily_window_start_ns(
                ts_ns, hour=daily_window_start_hour_utc
            )
            next_boundary = window_start + 86_400 * 1_000_000_000
            windows.append(HaltWindow(ts_ns, next_boundary, kind))
        elif kind in _FALLBACK_HALT_KINDS:
            windows.append(
                HaltWindow(ts_ns, ts_ns + fallback_duration_ns, kind)
            )
        # else: non-halt event — ignored.

    windows.sort(key=lambda w: w.start_ns)
    return windows


__all__ = [
    "HaltWindow",
    "in_halt_window",
    "daily_window_start_ns",
    "SimRiskCaps",
    "EntryGateInputs",
    "entry_veto",
    "load_halt_windows",
]
