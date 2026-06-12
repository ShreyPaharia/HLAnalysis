"""Single per-share fee model for the v3.1 theta family (theta + nba_wp).

The fee a strategy nets out of its edge takes one of three shapes, previously
hand-inlined at ~5 sites across ``theta_harvester.py`` and ``nba_wp.py``:

* ``fee_model == "pm_binary"`` — Polymarket curve ``fee_rate · p · (1 − p)``,
  applied on BOTH the entry and exit taker legs.
* ``fee_model == "flat"`` (HL, legacy) — a fixed ``fee_taker`` per share on
  entry; on exit, ``exit_fee`` when ``exit_take_profit_mode`` is on, else
  ``fee_taker``.

``cfg`` is a ``ThetaHarvesterConfig`` (both callers share it). ``side`` selects
the entry vs exit ladder; under ``pm_binary`` both sides use the same curve.
"""
from __future__ import annotations

from typing import Literal


def pm_binary_fee(fee_rate: float, p: float, qty: float = 1.0) -> float:
    """Polymarket binary fee for ``qty`` shares: ``qty * fee_rate * p * (1 - p)``.

    Single source of truth shared by ``fee_per_share`` (strategy layer, ``qty=1``
    → per-share) and ``_binary_fee`` (backtest runner, passes the fill ``qty``).
    The multiply order ``qty * fee_rate * p * (1-p)`` is preserved exactly so both
    callers stay byte-identical to their previous inline expressions (qty=1.0 is
    an exact IEEE-754 identity, so the per-share result is unchanged). Caller
    clamps ``p`` to [0, 1] when the input may be a raw fill price.
    """
    return qty * fee_rate * p * (1.0 - p)


def fee_per_share(cfg, p: float, *, side: Literal["entry", "exit"]) -> float:
    """Per-share taker fee at model probability ``p`` for the given ``side``."""
    if cfg.fee_model == "pm_binary":
        return pm_binary_fee(cfg.fee_rate, p)
    if side == "exit" and cfg.exit_take_profit_mode:
        return cfg.exit_fee
    return cfg.fee_taker
