"""Shared leg geometry for HIP-4 / Polymarket binary & bucket questions.

``kv_get`` and ``winning_region`` are pure functions of a ``QuestionView`` —
they read the question's class, leg layout and threshold metadata and return
the price interval in which a given leg wins. They were previously copy-pasted
(byte-for-byte) into ``late_resolution.py`` and ``theta_harvester.py`` (and a
``kv``-only variant in ``render.py``); since the geometry is identical across
strategies there is nothing strategy-specific to keep decoupled. Consolidated
here so a layout fix lands once.
"""

from __future__ import annotations

from .types import QuestionView


def kv_get(qv: QuestionView, key: str, default: str = "") -> str:
    """Return the value for ``key`` in the question's kv pairs, or ``default``."""
    for k, v in qv.kv:
        if k == key:
            return v
    return default


def winning_region(qv: QuestionView, symbol: str) -> tuple[float | None, float | None]:
    """Return (lo, hi) such that the leg ``symbol`` wins iff the underlying is in
    [lo, hi] at expiry. ``None`` denotes an unbounded side (-∞ for lo, +∞ for hi).

    Binary YES wins above strike → (strike, None); NO wins at-or-below → (None, strike).
    Bucket layout (HL convention, N thresholds yield N+1 outcomes, 2 legs each;
    YES at even leg index, NO at odd):
      outcome 0       (lowest)  → (None, thr[0])
      outcome 1..N-1  (middle)  → (thr[i-1], thr[i])
      outcome N       (highest) → (thr[-1], None)
    NO of an edge bucket inverts to the opposite half-line. NO of a middle
    bucket is the union of two half-lines (non-contiguous) and is signaled by
    returning (None, None) — callers must skip such legs as no-edge.
    """
    if qv.klass == "priceBinary":
        if symbol == qv.yes_symbol:
            return (qv.strike, None)
        if symbol == qv.no_symbol:
            return (None, qv.strike)
        return (None, None)

    if qv.klass != "priceBucket" or not qv.leg_symbols or symbol not in qv.leg_symbols:
        return (None, None)

    thresholds_raw = kv_get(qv, "priceThresholds")
    thr = [float(t) for t in thresholds_raw.split(",") if t.strip()]
    if not thr:
        return (None, None)

    idx = qv.leg_symbols.index(symbol)
    side_idx = idx % 2  # 0=YES, 1=NO

    # PM above-ladder layout: each pair (YES, NO) at index 2k/2k+1 is an
    # independent "above thr[k]" binary.  YES wins iff underlying > thr[k];
    # NO wins iff underlying <= thr[k].
    if kv_get(qv, "bucketLayout") == "above_ladder":
        k = idx // 2
        if k >= len(thr):
            return (None, None)
        if side_idx == 0:
            return (thr[k], None)  # YES: above thr[k]
        else:
            return (None, thr[k])  # NO: at-or-below thr[k]

    outcome_pos = idx // 2

    # YES region for this outcome bucket.
    if outcome_pos == 0:
        yes_lo: float | None = None
        yes_hi: float | None = thr[0]
    elif outcome_pos == len(thr):
        yes_lo, yes_hi = thr[-1], None
    elif 0 < outcome_pos < len(thr):
        yes_lo, yes_hi = thr[outcome_pos - 1], thr[outcome_pos]
    else:
        return (None, None)

    if side_idx == 0:
        return (yes_lo, yes_hi)

    # NO of a single-sided bucket inverts to the opposite half-line.
    if yes_lo is None:
        return (yes_hi, None)
    if yes_hi is None:
        return (None, yes_lo)
    # NO of a middle bucket = union of two half-lines (non-contiguous). Callers
    # treat (None, None) as "no leg-level gate" and fall back to non-safety
    # exits (stop-loss, settlement). Buying NO of a middle bucket is disallowed
    # by the YES-only entry path anyway.
    return (None, None)
