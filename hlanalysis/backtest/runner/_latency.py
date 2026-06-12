"""Pluggable order-latency models for the hftbacktest runner (SHR-89).

Extracted verbatim from ``hftbt_runner.py`` — no logic changes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# Matches hftbacktest's order-latency record layout (data/utils/feed_order_latency.py).
_ORDER_LATENCY_DTYPE = np.dtype(
    [("req_ts", "i8"), ("exch_ts", "i8"), ("resp_ts", "i8"), ("_padding", "i8")],
    align=True,
)


class LatencyModel:
    """Pluggable order-latency model. Subclasses configure how an order's
    round-trip latency δ is wired into the hftbacktest asset."""

    def apply(self, asset: Any, *, start_ts_ns: int, end_ts_ns: int) -> Any:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class ConstantLatency(LatencyModel):
    """Constant round-trip latency (Spec 1 behaviour). ``latency_ms`` ms entry
    AND response latency, identical for every order."""

    latency_ms: float = 50.0

    def apply(self, asset: Any, *, start_ts_ns: int, end_ts_ns: int) -> Any:
        ns = int(self.latency_ms * 1_000_000)
        return asset.constant_order_latency(ns, ns)


@dataclass(frozen=True, slots=True)
class SampledLatency(LatencyModel):
    """Draw order entry latency δ from an empirical sample of millisecond values
    (e.g. the SHR-83 journal's measured δ distribution, or any synthetic list).

    Builds an hftbacktest interpolated order-latency array spanning the question
    on a ``step_ns`` grid, sampling one δ per grid point. The engine interpolates
    each order's latency from the surrounding grid points, so an order submitted
    at ``t`` reaches the exchange around ``t + δ`` and fills on the book then.

    Deterministic given ``seed`` — sims must be reproducible (no process-salted
    randomness). ``resp_ms`` is the response latency added on top of the entry
    latency (default 0; the entry latency is what governs the fill book)."""

    samples_ms: tuple[float, ...]
    seed: int = 0
    step_ns: int = 1_000_000_000  # 1s grid
    resp_ms: float = 0.0

    def build_latency_array(self, *, start_ts_ns: int, end_ts_ns: int) -> np.ndarray:
        if not self.samples_ms:
            raise ValueError("SampledLatency requires at least one δ sample")
        span = max(0, int(end_ts_ns) - int(start_ts_ns))
        n = max(2, span // int(self.step_ns) + 2)
        rng = np.random.default_rng(self.seed)
        samples = np.asarray(self.samples_ms, dtype=float)
        draws = rng.choice(samples, size=n)
        req_ts = int(start_ts_ns) + np.arange(n, dtype=np.int64) * int(self.step_ns)
        entry_ns = np.rint(draws * 1_000_000).astype(np.int64)
        resp_ns = int(self.resp_ms * 1_000_000)
        out = np.zeros(n, dtype=_ORDER_LATENCY_DTYPE)
        out["req_ts"] = req_ts
        out["exch_ts"] = req_ts + entry_ns
        out["resp_ts"] = req_ts + entry_ns + resp_ns
        return out

    def apply(self, asset: Any, *, start_ts_ns: int, end_ts_ns: int) -> Any:
        arr = self.build_latency_array(start_ts_ns=start_ts_ns, end_ts_ns=end_ts_ns)
        return asset.intp_order_latency(arr)
