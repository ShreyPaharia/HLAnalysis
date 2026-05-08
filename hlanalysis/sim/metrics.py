from __future__ import annotations

import math
from dataclasses import dataclass


def annualized_sharpe(per_obs_pnl: list[float], *, periods_per_year: float) -> float:
    n = len(per_obs_pnl)
    if n < 2:
        return 0.0
    mu = sum(per_obs_pnl) / n
    var = sum((r - mu) ** 2 for r in per_obs_pnl) / (n - 1)
    if var <= 0:
        return 0.0
    return (mu / math.sqrt(var)) * math.sqrt(periods_per_year)


def hit_rate(per_obs_pnl: list[float]) -> float:
    if not per_obs_pnl:
        return 0.0
    return sum(1 for r in per_obs_pnl if r > 0) / len(per_obs_pnl)


@dataclass(frozen=True, slots=True)
class RunSummary:
    n_markets: int
    n_trades: int
    total_pnl_usd: float
    sharpe: float
    hit_rate: float
    max_drawdown_usd: float


def summarise_run(per_market_pnl: list[float], n_trades: int) -> RunSummary:
    sharpe = annualized_sharpe(per_market_pnl, periods_per_year=365.0)
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for r in per_market_pnl:
        cumulative += r
        peak = max(peak, cumulative)
        max_dd = max(max_dd, peak - cumulative)
    return RunSummary(
        n_markets=len(per_market_pnl),
        n_trades=n_trades,
        total_pnl_usd=sum(per_market_pnl),
        sharpe=sharpe,
        hit_rate=hit_rate(per_market_pnl),
        max_drawdown_usd=max_dd,
    )
