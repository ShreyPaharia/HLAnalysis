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
    )
    return LateResolutionStrategy(cfg)
