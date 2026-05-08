from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PMMarket(BaseModel):
    condition_id: str
    yes_token_id: str
    no_token_id: str
    start_ts_ns: int
    end_ts_ns: int
    resolved_outcome: Literal["yes", "no", "unknown"]
    total_volume_usd: float
    n_trades: int


class PMTrade(BaseModel):
    ts_ns: int
    token_id: str
    side: Literal["buy", "sell"]
    price: float = Field(ge=0.0, le=1.0)
    size: float = Field(gt=0.0)
