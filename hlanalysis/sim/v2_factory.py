from __future__ import annotations

from typing import Any

from hlanalysis.strategy.model_edge import ModelEdgeConfig, ModelEdgeStrategy


def build_v2_strategy_from_params(params: dict[str, Any]) -> ModelEdgeStrategy:
    drift_lb = int(params.get("drift_lookback_seconds", 0))
    cfg = ModelEdgeConfig(
        vol_lookback_seconds=int(params["vol_lookback_seconds"]),
        vol_sampling_dt_seconds=int(params.get("vol_sampling_dt_seconds", 60)),
        vol_clip_min=float(params.get("vol_clip_min", 0.05)),
        vol_clip_max=float(params.get("vol_clip_max", 3.0)),
        edge_buffer=float(params["edge_buffer"]),
        # PM CLOB taker fee = 0% (since ~2024). Override per param dict for HL HIP-4
        # or other venues with non-zero taker fees.
        fee_taker=float(params.get("fee_taker", 0.0)),
        # Default 0 because the sim's synthetic L2 already adds the half_spread to
        # the ask. Subtracting it again would double-count crossing cost. Keep
        # configurable for callers that feed real (mid-quoted) book data.
        half_spread_assumption=float(params.get("half_spread_assumption", 0.0)),
        stop_loss_pct=(float(params["stop_loss_pct"]) if params["stop_loss_pct"] is not None else None),
        drift_lookback_seconds=drift_lb,
        drift_blend=float(params.get("drift_blend", 1.0 if drift_lb else 0.0)),
        max_position_usd=float(params.get("max_position_usd", 100.0)),
    )
    return ModelEdgeStrategy(cfg)
