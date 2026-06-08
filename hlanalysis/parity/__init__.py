"""Sim-vs-live fidelity parity tooling (SHR-90).

The standing regression gate for the sim-fidelity program: for each settled
market it reconciles the sim's PnL/fills against live (venue truth + trade
journal) and attributes the residual to input-skew / execution / unmodeled-halt.
"""
