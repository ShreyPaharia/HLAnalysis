"""Multi-source backtester driven by hftbacktest.

Replaces `hlanalysis/sim/` (Polymarket-only synthetic-L2 backtester). The
public seam is the §3 interface contracts in :mod:`hlanalysis.backtest.core`;
data sources (`data/`) and the runner (`runner/`) consume them.
"""
