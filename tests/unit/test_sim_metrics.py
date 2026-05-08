from __future__ import annotations

import math

from hlanalysis.sim.metrics import (
    annualized_sharpe, hit_rate, summarise_run, RunSummary
)


def test_sharpe_with_constant_returns_is_inf_or_zero_std_handled():
    s = annualized_sharpe([1.0, 1.0, 1.0], periods_per_year=365)
    assert s == 0.0  # zero std → return 0.0 by convention


def test_sharpe_normalizes_by_period_count():
    rets = [0.01, -0.005, 0.02, 0.0, 0.015]
    s = annualized_sharpe(rets, periods_per_year=365)
    mu = sum(rets) / len(rets)
    var = sum((r - mu) ** 2 for r in rets) / (len(rets) - 1)
    expected = (mu / math.sqrt(var)) * math.sqrt(365)
    assert abs(s - expected) < 1e-9


def test_hit_rate():
    assert hit_rate([1.0, -1.0, 0.5, 0.0, -0.1]) == 0.4   # 2 strictly positive of 5
