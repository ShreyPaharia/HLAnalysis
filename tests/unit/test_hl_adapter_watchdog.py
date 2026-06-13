"""Part A: the Hyperliquid adapter now routes through BaseWsAdapter, so it
gains the per-frame staleness watchdog it previously lacked (SHR-60 was
binance-only). A silent/half-open HL socket must emit `feed_stale` and
reconnect instead of blocking forever on a bare `recv()`.
"""

from __future__ import annotations

import asyncio

import pytest

from hlanalysis.adapters.hyperliquid import HyperliquidAdapter
from hlanalysis.config import Subscription
from hlanalysis.events import HealthEvent, Mechanism, ProductType


class _SilentWS:
    """A live socket that accepts the subscribe but delivers no data frames."""

    def __init__(self):
        self.connects = 0

    async def send(self, _data):
        pass

    async def recv(self):
        await asyncio.sleep(3600)

    async def __aenter__(self):
        self.connects += 1
        return self

    async def __aexit__(self, *a):
        return False


@pytest.mark.asyncio
async def test_hl_silent_socket_trips_watchdog_and_reconnects(monkeypatch):
    adapter = HyperliquidAdapter()
    adapter.stale_timeout_s = 0.01
    adapter.reconnect_backoff_initial_s = 0.001
    adapter.reconnect_backoff_max_s = 0.001

    ws = _SilentWS()
    monkeypatch.setattr(adapter, "_connect", lambda url: ws)

    # A concrete (non-wildcard) PERP sub: no outcomeMeta REST in the loop.
    sub = Subscription(
        venue="hyperliquid",
        product_type=ProductType.PERP,
        mechanism=Mechanism.CLOB,
        symbol="BTC",
        channels=("bbo",),
    )

    seen: list[HealthEvent] = []

    async def _drain():
        async for ev in adapter.stream([sub]):
            seen.append(ev)
            if sum(1 for e in seen if e.kind.startswith("feed_stale")) >= 2:
                return

    await asyncio.wait_for(_drain(), timeout=2.0)

    kinds = [e.kind for e in seen]
    assert any(k.startswith("feed_stale") for k in kinds), kinds
    assert ws.connects >= 2, f"watchdog should have forced a reconnect: {ws.connects}"
    # Health for a PERP-only run is labelled perp, not mislabelled.
    assert all(e.product_type == ProductType.PERP for e in seen), kinds


@pytest.mark.asyncio
async def test_hl_wildcard_run_labels_health_prediction_binary(monkeypatch):
    """A HIP-4 wildcard run stamps connection health as prediction_binary
    (the old hardcoded PERP was wrong for the binary use case)."""
    adapter = HyperliquidAdapter()
    adapter.stale_timeout_s = 0.01
    adapter.reconnect_backoff_initial_s = 0.001

    ws = _SilentWS()
    monkeypatch.setattr(adapter, "_connect", lambda url: ws)

    # Wildcard sync hits the HL info REST; stub the (now async, offloaded)
    # network helper to return nothing so the loop proceeds to connect with no
    # network. _expand_wildcard / _sync_wildcards then early-return.
    async def _no_meta():
        return None

    monkeypatch.setattr(adapter, "_fetch_outcome_meta", _no_meta)

    sub = Subscription(
        venue="hyperliquid",
        product_type=ProductType.PREDICTION_BINARY,
        mechanism=Mechanism.CLOB,
        symbol="*",
        channels=("bbo",),
        match={"underlying": "BTC"},
    )

    seen: list[HealthEvent] = []

    async def _drain():
        async for ev in adapter.stream([sub]):
            seen.append(ev)
            if any(e.kind.startswith("connected") for e in seen):
                return

    await asyncio.wait_for(_drain(), timeout=2.0)
    connected = [e for e in seen if e.kind.startswith("connected")]
    assert connected, [e.kind for e in seen]
    assert all(e.product_type == ProductType.PREDICTION_BINARY for e in connected)
