"""HL Outcome-Market Desk Analysis toolkit.

Public API exports — every downstream notebook/script imports from here.
"""

from hlanalysis.research.dataset import build_panel
from hlanalysis.research.metrics import (
    depth_at_n,
    implied_prob_gbm,
    leadlag_xcorr,
    realized_vol_termstructure,
    settlement_convergence_curve,
    spread_bps,
    theta_decay_curve,
    trade_markout_curve,
    yes_no_overround,
)
from hlanalysis.research.outcome_markets import load_market_reference, load_settlements
from hlanalysis.research.report import Report, fig_to_base64

__all__ = [
    "load_market_reference",
    "load_settlements",
    "build_panel",
    "spread_bps",
    "depth_at_n",
    "trade_markout_curve",
    "leadlag_xcorr",
    "yes_no_overround",
    "implied_prob_gbm",
    "theta_decay_curve",
    "settlement_convergence_curve",
    "realized_vol_termstructure",
    "Report",
    "fig_to_base64",
]
