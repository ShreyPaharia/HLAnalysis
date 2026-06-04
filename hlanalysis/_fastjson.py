"""Fast JSON decode for the inbound WS hot path.

`msgspec.json.decode` returns the same plain Python structures as
`json.loads` (dict / list / str / int / float / bool / None) but is markedly
faster on the per-frame parse that dominates ingestion. Centralised here so the
adapters share one import site and the swap is benchmarkable / reversible.

Schema-less on purpose: the adapters already validate fields downstream when
they build pydantic NormalizedEvents, so we keep decode untyped and drop-in.
"""
from __future__ import annotations

from typing import Any

import msgspec.json as _mj

_decoder = _mj.Decoder()


def decode(raw: str | bytes) -> Any:
    """Drop-in replacement for ``json.loads`` on trusted WS frames."""
    return _decoder.decode(raw)
