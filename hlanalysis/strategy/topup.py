"""Shared position-topup logic for v1 (late_resolution) and v3.1 (theta).

When a held position is under-filled (its notional is below ``max_position_usd``
by at least ``topup_threshold_pct``), the strategy re-runs ALL entry gates
against the current state and, if the same leg is still the favourite, emits a
second IOC buy sized to the shortfall. This ~70-line tail was near-verbatim in
both strategies; only the two early-out branches differ:

* v1 returns ``None`` on a missing book and on the routine "fully sized" tick
  (so its caller falls through to the legacy have-position diagnostic).
* theta returns a HOLD ``topup_skip`` Decision (with a debug log) for both.

The caller supplies those two behaviours as ``on_no_book`` / ``on_not_needed``
callables and the entry evaluation as ``run_entry``; the shared tail (re-run
entry, gate-failed / leg-changed / below-min-notional skips, emit) is identical.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from loguru import logger

from .intents import make_entry_intent, round_size
from .types import Action, BookState, Decision, Diagnostic, Position, QuestionView


def run_topup(
    *,
    question: QuestionView,
    books: Mapping[str, BookState],
    position: Position,
    max_position_usd: float,
    topup_threshold_pct: float,
    topup_min_notional_usd: float,
    run_entry: Callable[[], Decision],
    on_no_book: Callable[[], Decision | None],
    on_not_needed: Callable[[float, float], Decision | None],
) -> Decision | None:
    """Evaluate a topup for an under-filled ``position``.

    Returns an ENTER Decision when a topup is warranted, a HOLD ``topup_skip``
    Decision when a check fired but was rejected, or whatever the caller's
    ``on_no_book`` / ``on_not_needed`` callables return for the two early-outs.
    """
    held = books.get(position.symbol)
    if held is None or held.ask_px is None or held.ask_px <= 0:
        return on_no_book()
    ask = held.ask_px
    current_ntl = abs(position.qty) * ask
    target_ntl = max_position_usd
    shortfall_ntl = target_ntl - current_ntl
    if shortfall_ntl < target_ntl * topup_threshold_pct:
        return on_not_needed(current_ntl, target_ntl)

    # Re-run ALL entry gates against the current state. The entry evaluator
    # already encodes every gate (TTE, near-strike, favorite, edge, edge_max,
    # bid notional, etc.) so we delegate rather than duplicate.
    entry_dec = run_entry()
    if entry_dec.action != Action.ENTER or not entry_dec.intents:
        failed_gate = entry_dec.diagnostics[0].message if entry_dec.diagnostics else "unknown"
        logger.debug(
            "topup_skip q={} sym={} reason=gate_failed:{} current_ntl=${:.2f} target_ntl=${:.2f}",
            question.question_idx,
            position.symbol,
            failed_gate,
            current_ntl,
            target_ntl,
        )
        return Decision(
            action=Action.HOLD,
            diagnostics=(Diagnostic("info", "topup_skip", (("reason", f"gate_failed:{failed_gate}"),)),),
        )
    candidate = entry_dec.intents[0]
    if candidate.symbol != position.symbol:
        logger.debug(
            "topup_skip q={} sym={} reason=leg_changed chosen={} current_ntl=${:.2f} target_ntl=${:.2f}",
            question.question_idx,
            position.symbol,
            candidate.symbol,
            current_ntl,
            target_ntl,
        )
        return Decision(
            action=Action.HOLD,
            diagnostics=(
                Diagnostic(
                    "info",
                    "topup_skip",
                    (
                        ("reason", "leg_changed"),
                        ("chosen", candidate.symbol),
                    ),
                ),
            ),
        )

    topup_size = round_size(shortfall_ntl, ask)
    topup_ntl = topup_size * ask
    if topup_ntl < topup_min_notional_usd:
        logger.debug(
            "topup_skip q={} sym={} reason=below_min_notional topup_ntl=${:.2f} min=${:.2f}",
            question.question_idx,
            position.symbol,
            topup_ntl,
            topup_min_notional_usd,
        )
        return Decision(
            action=Action.HOLD,
            diagnostics=(
                Diagnostic(
                    "info",
                    "topup_skip",
                    (
                        ("reason", "below_min_notional"),
                        ("topup_ntl", f"{topup_ntl:.2f}"),
                    ),
                ),
            ),
        )

    intent = make_entry_intent(
        question,
        symbol=position.symbol,
        size=topup_size,
        limit_price=ask,
    )
    logger.info(
        "topup_emit q={} sym={} side=buy current_ntl=${:.2f} target_ntl=${:.2f} topup_size={:.2f} ask={:.5f}",
        question.question_idx,
        position.symbol,
        current_ntl,
        target_ntl,
        topup_size,
        ask,
    )
    return Decision(
        action=Action.ENTER,
        intents=(intent,),
        diagnostics=(
            Diagnostic(
                "info",
                "topup_emit",
                (
                    ("current_ntl", f"{current_ntl:.2f}"),
                    ("target_ntl", f"{target_ntl:.2f}"),
                    ("topup_size", f"{topup_size:.2f}"),
                    ("ask", f"{ask:.5f}"),
                ),
            ),
        ),
    )
