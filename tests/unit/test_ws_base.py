"""Part A: shared BaseWsAdapter owns connect/backoff/health/circuit-breaker AND
the per-frame staleness watchdog, so EVERY venue (binance, hyperliquid,
polymarket) gets the half-open-socket detection — not just binance (SHR-60).

These tests target the base directly via a tiny concrete subclass; the
per-venue adapter tests assert the venues actually route through it.
"""
from __future__ import annotations

import asyncio

import pytest

from hlanalysis.adapters._ws_base import BaseWsAdapter
from hlanalysis.events import HealthEvent, ProductType


class _Tiny(BaseWsAdapter):
    venue = "tiny"

    def supports(self, *a, **k):
        return True

    async def stream(self, subscriptions):  # pragma: no cover - not used here
        if False:
            yield None


class _SilentWS:
    async def recv(self):
        await asyncio.sleep(3600)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class _OneFrameThenSilentWS:
    def __init__(self, frame: str):
        self._frame = frame
        self._sent = False

    async def recv(self):
        if not self._sent:
            self._sent = True
            return self._frame
        await asyncio.sleep(3600)


async def _drain(q: asyncio.Queue) -> list[HealthEvent]:
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


@pytest.mark.asyncio
async def test_watchdog_emits_feed_stale_and_returns_on_silent_stream():
    a = _Tiny()
    a.stale_timeout_s = 0.01
    q: asyncio.Queue = asyncio.Queue()

    await asyncio.wait_for(
        a._recv_until_stale(_SilentWS(), lambda m, t: [], "perp", q), timeout=1.0,
    )
    kinds = [e.kind for e in await _drain(q)]
    assert any(k.startswith("feed_stale") for k in kinds), kinds


@pytest.mark.asyncio
async def test_watchdog_health_carries_configured_product_type():
    a = _Tiny()
    a.stale_timeout_s = 0.01
    a.health_product_type = ProductType.PREDICTION_BINARY
    q: asyncio.Queue = asyncio.Queue()

    await asyncio.wait_for(
        a._recv_until_stale(_SilentWS(), lambda m, t: [], "", q), timeout=1.0,
    )
    evs = await _drain(q)
    assert evs and all(e.product_type == ProductType.PREDICTION_BINARY for e in evs), evs


@pytest.mark.asyncio
async def test_watchdog_dispatches_handler_then_trips_on_silence():
    a = _Tiny()
    a.stale_timeout_s = 0.01
    q: asyncio.Queue = asyncio.Queue()
    seen = []

    def handle(msg, recv_ns):
        seen.append(msg)
        return []

    await asyncio.wait_for(
        a._recv_until_stale(_OneFrameThenSilentWS('{"a":1}'), handle, "x", q),
        timeout=1.0,
    )
    assert seen == [{"a": 1}]
    kinds = [e.kind for e in await _drain(q)]
    assert any(k.startswith("feed_stale") for k in kinds), kinds


@pytest.mark.asyncio
async def test_run_ws_emits_connected_then_reconnect_and_retries():
    a = _Tiny()
    a.stale_timeout_s = 0.01
    q: asyncio.Queue = asyncio.Queue()
    connects = {"n": 0}

    class _CtxSilent:
        async def __aenter__(self):
            connects["n"] += 1
            return _SilentWS()

        async def __aexit__(self, *exc):
            return False

    def connect(url):
        return _CtxSilent()

    async def subscribe(ws):
        pass

    task = asyncio.create_task(
        a._run_ws(
            url="wss://x", subscribe=subscribe, handle=lambda m, t: [], queue=q,
            label="perp", connect=connect, circuit_breaker=False,
        )
    )
    # let it connect → go stale → reconnect at least twice
    await asyncio.sleep(0.2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    kinds = [e.kind for e in await _drain(q)]
    assert any(k.startswith("connected") for k in kinds), kinds
    assert any(k.startswith("feed_stale") for k in kinds), kinds
    assert connects["n"] >= 2, f"should have reconnected; connects={connects['n']}"


@pytest.mark.asyncio
async def test_circuit_breaker_trips_after_threshold():
    a = _Tiny()
    a.stale_timeout_s = 0.001
    a.reconnect_threshold = 3
    a.reconnect_window_s = 60.0
    a.reconnect_cooldown_s = 0.05
    a.reconnect_backoff_initial_s = 0.001
    a.reconnect_backoff_max_s = 0.001
    q: asyncio.Queue = asyncio.Queue()

    class _CtxBoom:
        async def __aenter__(self):
            return self

        async def recv(self):
            raise OSError("boom")

        async def __aexit__(self, *exc):
            return False

    task = asyncio.create_task(
        a._run_ws(
            url="wss://x", subscribe=lambda ws: asyncio.sleep(0),
            handle=lambda m, t: [], queue=q, label="", connect=lambda u: _CtxBoom(),
            circuit_breaker=True,
        )
    )
    await asyncio.sleep(0.2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    kinds = [e.kind for e in await _drain(q)]
    assert any(k == "circuit-breaker" for k in kinds), kinds
