"""Shared market-data primitives used by BOTH the live engine and the backtest.

This package is the single source of truth for logic the engine (live path) and
the hftbacktest runner (sim path) must keep bit-identical. Duplicated copies of
this logic drifted repeatedly and caused production incidents (sigma-sampling
regressions, position-accounting fidelity bugs); collapsing each pair to one
implementation here removes the manual-discipline coupling.
"""
