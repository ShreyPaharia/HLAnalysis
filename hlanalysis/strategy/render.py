"""Pure-policy human-readable rendering of QuestionView + outcome leg.

Lives in strategy/ because it operates only on QuestionView (no IO). Used by
the engine's router to populate alert messages and by sim/report.
"""

from __future__ import annotations

from datetime import UTC, datetime

from ..marketdata.position_math import PositionState, open_mtm
from .regions import kv_get as _kv
from .types import QuestionView


def _expiry_str(qv: QuestionView) -> str:
    if not qv.expiry_ns:
        return ""
    dt = datetime.fromtimestamp(qv.expiry_ns / 1e9, tz=UTC)
    return dt.strftime("%Y-%m-%d %H:%M UTC")


def _fmt_usd(s: str) -> str:
    """Format a numeric string as $X,XXX with no cents (PM/HL strikes are integers)."""
    try:
        n = float(s)
        return f"${n:,.0f}"
    except (TypeError, ValueError):
        return s


def question_description(qv: QuestionView) -> str:
    """Human-readable summary of the question.

    Examples:
      priceBinary HL/PM-bucket: "BTC > $79,583 by 2026-05-09 06:00 UTC"
      priceBinary PM up/down (no static strike): "Will BTC go up or down on May 26? (2026-05-27 00:00 UTC)"
      priceBucket: "BTC bucketed by $77,991 / $81,174 by 2026-05-09 06:00 UTC"
    """
    underlying = qv.underlying or "?"
    expiry = _expiry_str(qv)
    if qv.klass == "priceBinary":
        target = _kv(qv, "targetPrice") or _kv(qv, "strike")
        if not target and qv.strike == qv.strike:  # not NaN
            target = f"{qv.strike:.0f}"
        if target:
            return f"{underlying} > {_fmt_usd(target)} by {expiry}"
        # PM daily up/down markets resolve against a per-day reference
        # candle, so no static strike is published at market creation. Fall
        # back to the human-readable question text + expiry so the alert
        # still carries enough context to identify the market.
        if qv.name:
            return f"{qv.name} ({expiry})" if expiry else qv.name
        return f"{underlying} binary ({expiry})" if expiry else f"{underlying} binary"
    if qv.klass == "priceBucket":
        thresholds = _kv(qv, "priceThresholds")
        parts = [_fmt_usd(p) for p in thresholds.split(",") if p.strip()]
        joined = " / ".join(parts) if parts else "?"
        return f"{underlying} bucketed by {joined} by {expiry}"
    # Unknown class — fallback to the raw description fields.
    return f"{qv.klass or 'question'} {qv.name} ({expiry})".strip()


def outcome_description(qv: QuestionView, symbol: str) -> str:
    """Human-readable description of the leg this symbol represents.

    Examples:
      priceBinary YES: "YES (BTC > $79,583)"
      priceBinary NO:  "NO (BTC ≤ $79,583)"
      priceBucket leg 0 YES (lowest bucket): "YES (BTC < $77,991)"
      priceBucket leg 3 NO (middle bucket):  "NO (BTC NOT in $77,991–$81,174)"
    """
    if not qv.leg_symbols or symbol not in qv.leg_symbols:
        return f"leg {symbol}"
    idx = qv.leg_symbols.index(symbol)
    side_idx = idx % 2
    outcome_pos = idx // 2  # which named outcome
    side_label = "YES" if side_idx == 0 else "NO"

    if qv.klass == "priceBinary":
        target = _kv(qv, "targetPrice") or _kv(qv, "strike") or f"{qv.strike:.0f}"
        target_fmt = _fmt_usd(target)
        cond = f"{qv.underlying} > {target_fmt}"
        if side_idx == 0:
            return f"{side_label} ({cond})"
        return f"{side_label} ({qv.underlying} ≤ {target_fmt})"

    if qv.klass == "priceBucket":
        thresholds_raw = _kv(qv, "priceThresholds")
        thr = [t.strip() for t in thresholds_raw.split(",") if t.strip()]
        # Outcome layout (HL convention with N thresholds):
        #   outcome 0  → below the lowest threshold
        #   outcome 1..N-1 → between two adjacent thresholds
        #   outcome N → above the highest threshold
        if outcome_pos == 0 and thr:
            yes_cond = f"{qv.underlying} < {_fmt_usd(thr[0])}"
            no_cond = f"{qv.underlying} ≥ {_fmt_usd(thr[0])}"
        elif outcome_pos == len(thr):
            yes_cond = f"{qv.underlying} > {_fmt_usd(thr[-1])}" if thr else f"{qv.underlying}"
            no_cond = f"{qv.underlying} ≤ {_fmt_usd(thr[-1])}" if thr else f"{qv.underlying}"
        else:
            lo, hi = thr[outcome_pos - 1], thr[outcome_pos]
            yes_cond = f"{_fmt_usd(lo)} ≤ {qv.underlying} < {_fmt_usd(hi)}"
            no_cond = f"{qv.underlying} NOT in [{_fmt_usd(lo)}, {_fmt_usd(hi)})"
        cond = yes_cond if side_idx == 0 else no_cond
        return f"{side_label} ({cond})"

    return f"{side_label} {symbol}"


def settlement_pnl_usd(
    qv: QuestionView | None, symbol: str, qty: float, avg_entry: float,
    prior_realized: float = 0.0,
) -> float:
    """Expected PnL at settlement for a held position on a binary-payout
    market (HL HIP-4 or Polymarket CLOB).

    Both venues pay 1.0 per share to the winning leg and 0.0 to the rest.
    `qv.settled_symbol` is the canonical "which leg won" signal (set by
    MarketState._mark_settled from a SettlementEvent). When that's unknown
    (e.g. position vanished from the venue before the settle event was
    polled) we fall back to `qv.settled_side` + binary leg layout, then to
    `prior_realized` so callers don't lie about realized PnL.

    `prior_realized` is the per-position realized PnL accrued from partial
    exits prior to settlement and is added on top of the settlement leg.
    """
    if qv is None or not qv.settled:
        return prior_realized
    payout: float | None = None
    if qv.settled_symbols:
        payout = 1.0 if symbol in qv.settled_symbols else 0.0
    elif qv.settled_symbol:
        payout = 1.0 if symbol == qv.settled_symbol else 0.0
    elif qv.settled_side and qv.yes_symbol and qv.no_symbol:
        winning = qv.yes_symbol if qv.settled_side == "yes" else qv.no_symbol
        payout = 1.0 if symbol == winning else 0.0
    if payout is None:
        return prior_realized
    # SHR-88: the settlement leg PnL is the open position marked to its payoff —
    # routed through the shared ``position_math.open_mtm`` so the engine's
    # (compute-path) settlement accounting is identical to the sim's, by
    # construction. The winner above comes from venue truth (settled_symbol(s) /
    # settled_side), never a re-derived YES leg.
    return prior_realized + open_mtm(PositionState(qty, avg_entry), payout)
