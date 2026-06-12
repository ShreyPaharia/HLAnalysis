"""Binary-leg fee model for the hftbacktest runner.

Extracted verbatim from ``hftbt_runner.py`` — no logic changes.
"""
from __future__ import annotations

from hlanalysis.strategy.fee import pm_binary_fee


def _binary_fee(px: float, qty: float, cfg: object) -> float:
    """Per-fill fee for binary tokens. Branches on cfg.fee_model.

    - "flat":      px * qty * fee_taker          (legacy; HL & synthetic)
    - "pm_binary": qty * fee_rate * p * (1-p)    (Polymarket docs formula)
    """
    if cfg.fee_model == "pm_binary":  # type: ignore[attr-defined]
        p = max(0.0, min(1.0, px))
        return pm_binary_fee(cfg.fee_rate, p, qty)  # type: ignore[attr-defined]
    return px * qty * cfg.fee_taker  # type: ignore[attr-defined]
