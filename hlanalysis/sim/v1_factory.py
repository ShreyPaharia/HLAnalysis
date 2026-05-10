from __future__ import annotations

from typing import Any

from hlanalysis.strategy.late_resolution import LateResolutionConfig, LateResolutionStrategy


def build_v1_strategy_from_params(params: dict[str, Any]) -> LateResolutionStrategy:
    cfg = LateResolutionConfig(
        tte_min_seconds=int(params["tte_min_seconds"]),
        tte_max_seconds=int(params["tte_max_seconds"]),
        price_extreme_threshold=float(params["price_extreme_threshold"]),
        distance_from_strike_usd_min=float(params["distance_from_strike_usd_min"]),
        vol_max=float(params["vol_max"]),
        max_position_usd=float(params.get("max_position_usd", 100.0)),
        stop_loss_pct=float(params["stop_loss_pct"]) if params["stop_loss_pct"] is not None else 1e9,
        max_strike_distance_pct=float(params.get("max_strike_distance_pct", 50.0)),
        min_recent_volume_usd=float(params.get("min_recent_volume_usd", 0.0)),
        stale_data_halt_seconds=int(params.get("stale_data_halt_seconds", 86400)),
        price_extreme_max=float(params.get("price_extreme_max", 1.0)),
        min_safety_d=float(params.get("min_safety_d", 0.0)),
        vol_lookback_seconds=int(params.get("vol_lookback_seconds", 1800)),
        exit_safety_d=float(params.get("exit_safety_d", 0.0)),
        exit_bid_floor=float(params.get("exit_bid_floor", 0.0)),
        drift_aware_d=bool(params.get("drift_aware_d", False)),
    )
    return LateResolutionStrategy(cfg)
