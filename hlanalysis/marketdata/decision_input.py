"""Shared decision-input resolver (SHR-97).

Single config-driven path that determines HOW reference-price data is wired
into the shared MarketState for both the live engine and the backtest. The two
divergent wirings that caused SHR-92/93/96 are collapsed here.

Usage
-----
Engine (``_register_reference_cadences``)::

    from hlanalysis.marketdata.decision_input import from_engine
    dic = from_engine(slot.cfg)
    market_state.set_reference_cadence(sym, sampling_dt_seconds=dic.sampling_dt_seconds,
                                       lookback_seconds=dic.vol_lookback_seconds)
    market_state.set_reference_source(sym, dic.reference_source)

Backtest (``_source_config_from_args`` / ``cmd_run`` / ``cmd_tune``)::

    from hlanalysis.marketdata.decision_input import from_backtest_params
    dic = from_backtest_params(params, track_default_source="mark")
    source_config = SourceConfig(
        ...
        hl_ref_event=dic.reference_source,
        hl_ref_ticks=dic.reference_ticks,
        reference_resample_seconds=dic.sampling_dt_seconds,
    )
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hlanalysis.engine.config import StrategyConfig


@dataclass(frozen=True)
class DecisionInputConfig:
    """Resolved decision-input wiring: which price stream, at what cadence.

    A single struct shared by the engine and backtest so the two paths are
    provably configured identically. The four fields cover the three divergences
    that SHR-92/93/96 each fixed in isolation:

    ``reference_source``  — ``"mark"`` (venue mark price) or ``"bbo"`` (BBO mid).
                            Engine reads ``reference_sigma_source`` (default "mark");
                            backtest had a non-live default of ``"bbo"`` (SHR-92).
    ``sampling_dt_seconds`` — OHLC bucket width == ``vol_sampling_dt_seconds``.
                              Must agree between the σ formula and the bar bucketer
                              or σ is inflated by ``sqrt(dt_live / dt_backtest)``
                              (SHR-96 interaction bug).
    ``vol_lookback_seconds`` — σ window used for history sizing / warm-up.
    ``reference_ticks``   — ``"raw"`` (always for live) or ``"bars"`` (legacy override).
                            The live engine always injects raw ticks; the backtest
                            had a non-live default of ``"bars"`` (SHR-93).
    """

    reference_source: str        # "mark" | "bbo"
    sampling_dt_seconds: int     # OHLC bucket width == vol_sampling_dt_seconds
    vol_lookback_seconds: int    # σ window (history sizing / warm-up)
    reference_ticks: str         # "raw" (live) | "bars" (legacy override)


def from_engine(cfg: "StrategyConfig") -> DecisionInputConfig:
    """Build a ``DecisionInputConfig`` from a live-engine ``StrategyConfig``.

    Returns EXACTLY the values the engine uses today — this is the canonical
    live wiring. The engine stays bit-identical (no behaviour change).
    """
    from hlanalysis.engine.config_builders import (
        reference_sampling_dt_seconds,
        reference_vol_lookback_seconds,
    )

    return DecisionInputConfig(
        reference_source=cfg.reference_sigma_source,
        sampling_dt_seconds=reference_sampling_dt_seconds(cfg),
        vol_lookback_seconds=reference_vol_lookback_seconds(cfg),
        reference_ticks="raw",  # live always raw — this is what the engine does
    )


def from_backtest_params(
    params: dict,
    *,
    track_default_source: str,
) -> DecisionInputConfig:
    """Build a ``DecisionInputConfig`` from a backtest params dict.

    Priority for ``reference_source``:
      1. ``params["reference_sigma_source"]`` if present (explicit override)
      2. ``track_default_source`` (HL → "mark", PM → "mark")

    ``sampling_dt_seconds`` — ``params["vol_sampling_dt_seconds"]`` (flat) or
    the FIRST class's value in a per-class nested dict (the primary class),
    defaulting to 60.

    ``vol_lookback_seconds`` — max across all classes (flat or nested), so the
    reference buffer covers the largest window any class will request.

    ``reference_ticks`` — always ``"raw"`` (live-faithful default, SHR-93 fix).
    """
    # Source: explicit params override > track default.
    reference_source = (
        str(params["reference_sigma_source"])
        if "reference_sigma_source" in params
        else track_default_source
    )

    # Collect all vol_sampling_dt_seconds and vol_lookback_seconds values,
    # handling both flat dicts and per-class nested dicts.
    dt_values: list[int] = []
    lookback_values: list[int] = []

    for k, v in params.items():
        if isinstance(v, dict):
            # Per-class nested: e.g. {"binary": {...}, "bucket": {...}}
            if "vol_sampling_dt_seconds" in v:
                dt_values.append(int(v["vol_sampling_dt_seconds"]))
            if "vol_lookback_seconds" in v:
                lookback_values.append(int(v["vol_lookback_seconds"]))

    # Flat top-level keys.
    if "vol_sampling_dt_seconds" in params:
        val = params["vol_sampling_dt_seconds"]
        if not isinstance(val, dict):
            dt_values.insert(0, int(val))  # flat takes precedence as primary
    if "vol_lookback_seconds" in params:
        val = params["vol_lookback_seconds"]
        if not isinstance(val, dict):
            lookback_values.append(int(val))

    # Primary dt: flat key > first class in iteration order > default 60.
    sampling_dt_seconds = dt_values[0] if dt_values else 60

    # Lookback: max across all classes (largest window wins for sizing).
    vol_lookback_seconds = max(lookback_values) if lookback_values else 3600

    return DecisionInputConfig(
        reference_source=reference_source,
        sampling_dt_seconds=sampling_dt_seconds,
        vol_lookback_seconds=vol_lookback_seconds,
        reference_ticks="raw",  # live-faithful default (SHR-93 fix)
    )


__all__ = [
    "DecisionInputConfig",
    "from_engine",
    "from_backtest_params",
]
