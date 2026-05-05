from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict

from .events import Mechanism, ProductType


# Standard channel vocabulary. Each adapter translates these to native streams.
# Adapters may silently drop channels that don't apply (e.g. funding for spot).
STANDARD_CHANNELS = frozenset(
    {"trades", "book", "bbo", "mark", "funding", "oracle", "liquidations"}
)


class Subscription(BaseModel):
    model_config = ConfigDict(frozen=True)

    venue: str
    product_type: ProductType
    mechanism: Mechanism
    symbol: str
    # Channels to subscribe to for this market. Use STANDARD_CHANNELS values.
    channels: tuple[str, ...] = ("trades",)
    # Static metadata (e.g. strike, expiry, settlement source for prediction markets).
    metadata: dict[str, Any] = {}


class RecorderConfig(BaseModel):
    subscriptions: list[Subscription]


def load_config(path: Path) -> RecorderConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return RecorderConfig(**raw)
