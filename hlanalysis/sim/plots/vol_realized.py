"""hlanalysis/sim/plots/vol_realized.py

Scatter of entry-time σ (annualised implied vol) vs realized 24h |log-return|
of BTC for each market evaluated by the strategy.

Public API
----------
plot_vol_realized(diagnostics_dir, klines_dir, markets, out_path,
                  tte_min_seconds, tte_max_seconds) -> Path | None
    For each market: resolve the entry-time σ from the diagnostics parquet,
    pair with the realized 24h |log-return| computed from the kline cache,
    produce an interactive scatter with the σ·√τ reference line, write HTML to
    *out_path*, and return *out_path*. Returns None (writes no file) if no
    market has a valid σ (e.g. v1 run or empty diagnostics).

_compute_vol_realized_points(diagnostics_dir, klines_dir, markets,
                              tte_min_seconds, tte_max_seconds)
    Pure helper exposed for unit testing.  Returns a list of
    ``(sigma, realized_abs_logret)`` tuples — one per market that has a valid σ
    and a valid realized return.  Skips markets with no klines or all-null σ.
"""
from __future__ import annotations

import bisect
import json
import math
from pathlib import Path
from typing import Optional, Protocol

import pyarrow.parquet as pq

from hlanalysis.sim.plots._common import save_fig


# ---------------------------------------------------------------------------
# Protocol for market objects (duck-typed; accepts PMMarket or test fakes)
# ---------------------------------------------------------------------------

class _MarketLike(Protocol):
    condition_id: str
    start_ts_ns: int
    end_ts_ns: int


# ---------------------------------------------------------------------------
# Entry-window midpoint TTE (seconds) — matches v2 strategy defaults
# Defaults for tte_min_seconds / tte_max_seconds kwargs below.
# ---------------------------------------------------------------------------

_DEFAULT_TTE_MIN_SECONDS: float = 14_400.0    # 4h
_DEFAULT_TTE_MAX_SECONDS: float = 86_400.0    # 24h
_TTE_MID_SECONDS: float = (_DEFAULT_TTE_MIN_SECONDS + _DEFAULT_TTE_MAX_SECONDS) / 2.0  # 14h

_SECS_PER_YEAR: float = 365.25 * 24.0 * 3600.0
_TTE_MID_YR: float = _TTE_MID_SECONDS / _SECS_PER_YEAR

_TAU_1D: float = 1.0 / 365.25   # 1 calendar day in years


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_diagnostics(path: Path) -> dict[str, list]:
    """Load a diagnostics parquet; return column-oriented dict or {} on error."""
    if not path.exists():
        return {}
    try:
        table = pq.read_table(path)
    except Exception:
        return {}
    if table.num_rows == 0:
        return {}
    return table.to_pydict()


def _sigma_for_market(raw: dict[str, list]) -> Optional[float]:
    """Return the best σ from a diagnostics dict for one market.

    Priority:
    1. First ENTER row's σ (if any ENTER action with non-null sigma).
    2. Row whose tau_yr is closest to the entry-window midpoint TTE.
    3. None if no row has a non-null sigma.
    """
    actions = raw.get("action", [])
    sigmas = raw.get("sigma", [])
    tau_yrs = raw.get("tau_yr", [])

    n = len(sigmas)
    if n == 0:
        return None

    # Priority 1: first ENTER row with non-null sigma
    for i, action in enumerate(actions):
        if action == "enter":
            s = sigmas[i]
            if s is not None:
                return float(s)

    # Priority 2: row closest to midpoint TTE with non-null sigma
    best_sigma: Optional[float] = None
    best_dist: float = float("inf")
    for i in range(n):
        s = sigmas[i]
        if s is None:
            continue
        tau = tau_yrs[i] if i < len(tau_yrs) else None
        if tau is None:
            continue
        dist = abs(float(tau) - _TTE_MID_YR)
        if dist < best_dist:
            best_dist = dist
            best_sigma = float(s)

    return best_sigma


def _load_all_klines(klines_dir: Path) -> list[dict]:
    """Load all kline records from all JSON files in *klines_dir*, sorted by ts_ns."""
    if not klines_dir.exists():
        return []
    records: list[dict] = []
    for f in sorted(klines_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        records.extend(data)
    # Sort ascending by ts_ns
    records.sort(key=lambda k: k["ts_ns"])
    return records


def _realized_abs_logret(
    klines: list[dict],
    start_ts_ns: int,
    end_ts_ns: int,
) -> Optional[float]:
    """Compute |log(close_at_end / open_at_start)| from kline records.

    Accepts the full sorted klines list (unsifted); does bisect-based lookup
    internally so callers need not pre-filter.

    open_at_start: ``open`` of the kline whose ts_ns is the largest value ≤ start_ts_ns.
    close_at_end:  ``close`` of the kline whose ts_ns is the largest value ≤ end_ts_ns.

    Returns None if no suitable klines are found.
    """
    if not klines:
        return None

    # Build a sorted list of ts_ns values for bisect.
    # klines is already sorted by ts_ns (guaranteed by _load_all_klines).
    ts_list = [k["ts_ns"] for k in klines]

    # Find the largest kline index whose ts_ns <= start_ts_ns
    # bisect_right gives us the insertion point after all equal values, so
    # idx-1 is the last index with ts_ns <= start_ts_ns.
    idx_open = bisect.bisect_right(ts_list, start_ts_ns) - 1
    if idx_open < 0:
        return None
    open_val = float(klines[idx_open]["open"])

    # Find the largest kline index whose ts_ns <= end_ts_ns
    idx_close = bisect.bisect_right(ts_list, end_ts_ns) - 1
    if idx_close < 0:
        return None
    close_val = float(klines[idx_close]["close"])

    if open_val <= 0.0 or close_val <= 0.0:
        return None

    return abs(math.log(close_val / open_val))


# ---------------------------------------------------------------------------
# Pure computation helper (testable without plotly)
# ---------------------------------------------------------------------------

def _compute_vol_realized_points(
    diagnostics_dir: Path,
    klines_dir: Path,
    markets: list,
    tte_min_seconds: int = 14_400,
    tte_max_seconds: int = 86_400,
) -> list[tuple[float, float]]:
    """Return list of (sigma, realized_abs_logret) for each valid market.

    A market is valid if:
    - Its diagnostics parquet exists and has at least one row with non-null sigma.
    - The kline cache has at least one kline at or before both start_ts_ns and end_ts_ns.

    Markets that fail either condition are silently skipped.

    Parameters
    ----------
    tte_min_seconds:
        Minimum TTE for the entry window in seconds.
        Default 14_400 (4h) matches v2's current config.
    tte_max_seconds:
        Maximum TTE for the entry window in seconds.
        Default 86_400 (24h) matches v2's current config.
    """
    all_klines = _load_all_klines(klines_dir)

    points: list[tuple[float, float]] = []
    for m in markets:
        diag_path = diagnostics_dir / f"{m.condition_id}.parquet"
        raw = _load_diagnostics(diag_path)
        sigma = _sigma_for_market(raw)
        if sigma is None:
            continue

        # Pass the full sorted klines list; _realized_abs_logret handles
        # boundary lookup via bisect — no pre-filtering needed.
        rlr = _realized_abs_logret(all_klines, m.start_ts_ns, m.end_ts_ns)
        if rlr is None:
            continue

        points.append((sigma, rlr))

    return points


# ---------------------------------------------------------------------------
# Plot builder
# ---------------------------------------------------------------------------

def plot_vol_realized(
    diagnostics_dir: Path,
    klines_dir: Path,
    markets: list,
    out_path: Path,
    tte_min_seconds: int = 14_400,
    tte_max_seconds: int = 86_400,
) -> Optional[Path]:
    """Build and write the σ vs realized 24h |log-return| scatter.

    For each market the strategy evaluated, plots the annualised implied σ
    (from the entry decision or the closest-to-midpoint-TTE diagnostic row)
    against the realized 24h |log-return| of BTC on that market's day.

    Overlays the reference line y = σ · √(1/365.25), spanning from min(σ) to
    max(σ) with 50 sample points.

    Returns *out_path* on success, or None when no market has a valid σ (v1
    run, empty diagnostics, or no matching klines).  Never raises on missing
    files.

    Parameters
    ----------
    tte_min_seconds:
        Minimum TTE for the entry window in seconds.
        Default 14_400 (4h) matches v2's current config.
    tte_max_seconds:
        Maximum TTE for the entry window in seconds.
        Default 86_400 (24h) matches v2's current config.
    """
    points = _compute_vol_realized_points(
        diagnostics_dir, klines_dir, markets,
        tte_min_seconds=tte_min_seconds,
        tte_max_seconds=tte_max_seconds,
    )
    if not points:
        return None

    # Import plotly here to keep it out of other modules
    import plotly.graph_objects as go  # noqa: PLC0415

    sigmas = [p[0] for p in points]
    realized = [p[1] for p in points]

    scatter = go.Scatter(
        x=sigmas,
        y=realized,
        mode="markers",
        name="market",
        marker=dict(opacity=0.55, size=7, color="#5b7fad"),
        hovertemplate="σ=%{x:.4f}<br>|Δln S|=%{y:.4f}<extra></extra>",
    )

    # Reference line: y = σ · √(1/365.25) — sampled over 50 x points
    sigma_lo = min(sigmas)
    sigma_hi = max(sigmas)
    if sigma_lo == sigma_hi:
        # Only one unique σ — still draw the reference as two coincident points
        ref_xs = [sigma_lo, sigma_hi]
    else:
        step = (sigma_hi - sigma_lo) / 49
        ref_xs = [sigma_lo + i * step for i in range(50)]
    ref_ys = [x * math.sqrt(_TAU_1D) for x in ref_xs]

    ref_line = go.Scatter(
        x=ref_xs,
        y=ref_ys,
        mode="lines",
        name="σ·√(1d)",
        line=dict(color="black", dash="dash", width=1),
        hoverinfo="skip",
    )

    fig = go.Figure([scatter, ref_line])
    fig.update_layout(
        title="σ vs realized 24h |log-return|",
        xaxis_title="σ (annualised, entry-time)",
        yaxis_title="realized 24h |Δ ln S|",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="closest",
    )

    return save_fig(fig, out_path)
