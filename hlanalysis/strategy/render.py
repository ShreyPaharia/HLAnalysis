"""Pure-policy human-readable rendering of QuestionView + outcome leg.

Lives in strategy/ because it operates only on QuestionView (no IO). Used by
the engine's router to populate alert messages and by sim/report.
"""

from __future__ import annotations

from datetime import datetime, timezone

from .types import QuestionView


def _kv(qv: QuestionView, key: str, default: str = "") -> str:
    for k, v in qv.kv:
        if k == key:
            return v
    return default


def _expiry_str(qv: QuestionView) -> str:
    if not qv.expiry_ns:
        return ""
    dt = datetime.fromtimestamp(qv.expiry_ns / 1e9, tz=timezone.utc)
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
      priceBinary: "BTC > $79,583 by 2026-05-09 06:00 UTC"
      priceBucket: "BTC bucketed by $77,991 / $81,174 by 2026-05-09 06:00 UTC"
    """
    underlying = qv.underlying or "?"
    expiry = _expiry_str(qv)
    if qv.klass == "priceBinary":
        target = _kv(qv, "targetPrice") or _kv(qv, "strike")
        if not target and qv.strike == qv.strike:  # not NaN
            target = f"{qv.strike:.0f}"
        return f"{underlying} > {_fmt_usd(target)} by {expiry}"
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
