"""Durable live-engine trade journal (SHR-83 / Spec 3).

ONE journal row per emitted decision/order, progressively populated across the
decision → send → fill → (reject) lifecycle. It captures, per order:

* ``cloid`` / ``question_idx`` / ``decision_ts``;
* the book-at-decision (top-N levels) and the ``evaluate()`` inputs (σ,
  recent_returns summary, recent_volume_usd, reference price);
* the emitted ``Decision`` (action + diagnostics);
* ``send_ts``, venue ``fill_ts`` + fill px/sz, and the reject/veto reason; and
* the slot halt-state at decision time (stale_data_halt / daily_loss_cap /
  reject_breaker / restart_blocked).

The single journal serves three consumers at once — decision-parity (sim vs
live), execution-latency calibration, and halt replay.

**Off the hot path.** Every write is best-effort: the engine hooks (scanner /
router / runtime / reconcile) call these methods inline, so a journal write must
NEVER raise or block order submission. All persistence is wrapped here and a
failure is logged and swallowed. The journal observes; it must not influence
trading.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass

from loguru import logger

from ..strategy.types import BookState, Diagnostic
from .state import StateDAL, TradeJournalRow

# Diagnostic field names a strategy uses to surface the σ it computed, in
# preference order. theta_harvester / late_resolution tag entry diagnostics with
# "vol"; this lets the journal record the strategy's own σ rather than only a
# returns-derived proxy.
_SIGMA_FIELD_KEYS = ("sigma", "vol", "vol_sigma")

# Default number of L2 levels per side snapshotted into book_json.
_DEFAULT_TOP_N = 5


@dataclass(frozen=True, slots=True)
class HaltSnapshot:
    """The slot halt-state at decision time (SHR-83). Assembled by the runtime
    (restart_blocked / daily_loss_halted / realized_pnl_today / cap) and
    augmented by the router (reject_breaker_tripped / stale_reference) so the
    journal row records exactly which gates were live when the decision was
    made — the input the sim's halt-replay needs."""

    restart_blocked: bool = False
    daily_loss_halted: bool = False
    realized_pnl_today: float = 0.0
    daily_loss_cap_usd: float = 0.0
    reject_breaker_tripped: bool = False
    stale_reference: bool = False


def _returns_summary(recent_returns) -> dict | None:
    """Compact summary of the recent_returns window (we don't persist the raw
    array — it can be hundreds of floats at sub-minute cadence)."""
    rs = list(recent_returns or ())
    if not rs:
        return None
    return {
        "n": len(rs),
        "mean": statistics.fmean(rs),
        "std": statistics.pstdev(rs) if len(rs) > 1 else 0.0,
        "last": rs[-1],
    }


def _sigma_from(diagnostics, recent_returns) -> float | None:
    """The annualised σ the strategy reasoned about.

    Only the evaluate() diagnostics carry an annualised vol (tagged under one of
    the keys in ``_SIGMA_FIELD_KEYS``).  Entry decisions surface it in the
    "edge" diagnostic block; exit decisions typically do NOT — their diagnostics
    name the gate that fired (e.g. "exit_safety_d_below_min") and carry no vol
    field.

    The function returns ``None`` when no annualised σ is available.  The
    former fallback to ``statistics.pstdev(recent_returns)`` produced a
    raw-returns stdev (≈ 0.0002) on exit rows while entry rows stored an
    annualised value (≈ 0.40), making the column incomparable across row types
    and silently corrupting any consumer that compared them."""
    for d in diagnostics or ():
        for k, v in d.fields:
            if k in _SIGMA_FIELD_KEYS:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    continue
    return None


def _book_snapshot(book: BookState | None, top_n: int) -> dict | None:
    if book is None:
        return None
    return {
        "symbol": book.symbol,
        "bid_px": book.bid_px,
        "bid_sz": book.bid_sz,
        "ask_px": book.ask_px,
        "ask_sz": book.ask_sz,
        "last_l2_ts_ns": book.last_l2_ts_ns,
        "bid_levels": [list(lvl) for lvl in book.bid_levels[:top_n]],
        "ask_levels": [list(lvl) for lvl in book.ask_levels[:top_n]],
    }


def _diagnostics_snapshot(diagnostics) -> list | None:
    out = [{"level": d.level, "message": d.message, "fields": [list(f) for f in d.fields]} for d in diagnostics or ()]
    return out or None


class TradeJournal:
    """Best-effort writer for the per-order trade journal.

    Wraps a slot's :class:`StateDAL`; every method swallows persistence errors
    so the journal can never break the engine's order path."""

    def __init__(
        self,
        dal: StateDAL,
        *,
        suppress_veto_reasons: frozenset[str] = frozenset(),
    ) -> None:
        self.dal = dal
        # Veto reasons whose decisions we do NOT retain. The router journals the
        # decision the instant the cloid is final (so a sent/filled order always
        # has a row); when one of these high-frequency mechanical gates then
        # vetoes it pre-send, the transient row is dropped here. Without this a
        # high-fan-out slot accumulates hundreds of thousands of `low_volume`
        # veto rows/day (the 2026-06-14 eth_ms disk-fill). Empty set = retain all
        # (the historical behaviour).
        self.suppress_veto_reasons = suppress_veto_reasons

    def record_decision(
        self,
        *,
        cloid: str,
        question_idx: int,
        decision_ts_ns: int,
        action: str,
        side: str | None = None,
        symbol: str | None = None,
        intended_size: float | None = None,
        intended_price: float | None = None,
        book: BookState | None = None,
        reference_price: float | None = None,
        recent_volume_usd: float | None = None,
        recent_returns=(),
        diagnostics: tuple[Diagnostic, ...] = (),
        halt: HaltSnapshot | None = None,
        top_n: int = _DEFAULT_TOP_N,
    ) -> None:
        """Append the decision row for ``cloid`` (insert-once; a duplicate cloid
        is a no-op). Captures inputs + book + halt at decision time."""
        try:
            row = TradeJournalRow(
                cloid=cloid,
                question_idx=question_idx,
                decision_ts_ns=decision_ts_ns,
                action=action,
                side=side,
                symbol=symbol,
                intended_size=intended_size,
                intended_price=intended_price,
                reference_price=reference_price,
                recent_volume_usd=recent_volume_usd,
                sigma=_sigma_from(diagnostics, recent_returns),
                returns_summary_json=_json_or_none(_returns_summary(recent_returns)),
                book_json=_json_or_none(_book_snapshot(book, top_n)),
                diagnostics_json=_json_or_none(_diagnostics_snapshot(diagnostics)),
                halt_json=_json_or_none(asdict(halt) if halt is not None else None),
            )
            self.dal.add_journal_decision(row)
        except Exception as e:  # noqa: BLE001 — journal must never raise
            logger.warning("trade_journal record_decision failed cloid={}: {}", cloid, e)

    def record_send(self, *, cloid: str, send_ts_ns: int) -> None:
        self._update(cloid, send_ts_ns=send_ts_ns)

    def record_reject(self, *, cloid: str, reject_reason: str) -> None:
        # Drop (don't retain) decisions vetoed by a routine, high-frequency gate
        # reason — the decision row was inserted at decision time; remove it now
        # rather than update it with the reason. Best-effort like every write.
        if reject_reason in self.suppress_veto_reasons:
            try:
                self.dal.delete_journal_decision(cloid)
            except Exception as e:  # noqa: BLE001 — journal must never raise
                logger.warning("trade_journal suppress-delete failed cloid={}: {}", cloid, e)
            return
        self._update(cloid, reject_reason=reject_reason)

    def record_fill(
        self,
        *,
        cloid: str,
        fill_ts_ns: int,
        fill_px: float,
        fill_sz: float,
    ) -> None:
        self._update(
            cloid,
            fill_ts_ns=fill_ts_ns,
            fill_px=fill_px,
            fill_sz=fill_sz,
        )

    def _update(self, cloid: str, **changes) -> None:
        try:
            self.dal.update_journal(cloid, **changes)
        except Exception as e:  # noqa: BLE001 — journal must never raise
            logger.warning("trade_journal update failed cloid={}: {}", cloid, e)


def _json_or_none(obj) -> str | None:
    return None if obj is None else json.dumps(obj)
