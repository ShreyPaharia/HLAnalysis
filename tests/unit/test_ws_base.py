"""Part A: shared BaseWsAdapter owns connect/backoff/health/circuit-breaker AND
the per-frame staleness watchdog, so EVERY venue (binance, hyperliquid,
polymarket) gets the half-open-socket detection — not just binance (SHR-60).

These tests target the base directly via a tiny concrete subclass; the
per-venue adapter tests assert the venues actually route through it.
"""

from __future__ import annotations

import asyncio
import logging

import pytest
import requests

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
        a._recv_until_stale(_SilentWS(), lambda m, t: [], "perp", q),
        timeout=1.0,
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
        a._recv_until_stale(_SilentWS(), lambda m, t: [], "", q),
        timeout=1.0,
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
            url="wss://x",
            subscribe=subscribe,
            handle=lambda m, t: [],
            queue=q,
            label="perp",
            connect=connect,
            circuit_breaker=False,
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
            url="wss://x",
            subscribe=lambda ws: asyncio.sleep(0),
            handle=lambda m, t: [],
            queue=q,
            label="",
            connect=lambda u: _CtxBoom(),
            circuit_breaker=True,
        )
    )
    await asyncio.sleep(0.2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    kinds = [e.kind for e in await _drain(q)]
    assert any(k == "circuit-breaker" for k in kinds), kinds


# ---------------------------------------------------------------------------
# Fix 3: JSON decode failures must increment decode_failures counter and emit
# a throttled log.warning once past the threshold — not silently swallow every
# bad frame, which would hide a venue wire-format change.
# ---------------------------------------------------------------------------


class _MalformedFramesWS:
    """Delivers N malformed (non-JSON) frames then goes silent forever."""

    def __init__(self, n: int) -> None:
        self._n = n
        self._sent = 0

    async def recv(self) -> str:
        if self._sent < self._n:
            self._sent += 1
            return "not-valid-json{{{"
        await asyncio.sleep(3600)


@pytest.mark.asyncio
async def test_decode_failure_counter_increments(caplog):
    """Each malformed frame increments decode_failures on the adapter."""
    a = _Tiny()
    a.stale_timeout_s = 0.01  # short so the test exits quickly after frames
    q: asyncio.Queue = asyncio.Queue()

    n_bad = 3
    await asyncio.wait_for(
        a._recv_until_stale(_MalformedFramesWS(n_bad), lambda m, t: [], "x", q),
        timeout=2.0,
    )
    assert a.decode_failures == n_bad, f"expected decode_failures={n_bad}, got {a.decode_failures}"


@pytest.mark.asyncio
async def test_decode_failure_warning_emitted_at_threshold(caplog):
    """A warning is emitted once decode_failures reaches _decode_fail_warn_threshold.

    We set the threshold to 2 and feed 3 bad frames; the third frame should
    NOT produce an extra warning (only at-threshold fires, not at-threshold+1).
    The warning after exactly threshold frames proves the venue wire-format
    change becomes visible without flooding every frame.
    """
    a = _Tiny()
    a.stale_timeout_s = 0.01
    a._decode_fail_warn_threshold = 2
    a._decode_fail_warn_every = 1000  # suppress the periodic re-fire for this test
    q: asyncio.Queue = asyncio.Queue()

    with caplog.at_level(logging.WARNING, logger="hlanalysis.adapters.tiny"):
        await asyncio.wait_for(
            a._recv_until_stale(_MalformedFramesWS(3), lambda m, t: [], "x", q),
            timeout=2.0,
        )

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING and "decode" in r.message.lower()]
    assert warnings, "no decode-failure warning logged at threshold"
    # Only one warning should fire for the threshold crossing (not one per frame).
    assert len(warnings) == 1, f"expected exactly 1 threshold-crossing warning, got {len(warnings)}: " + str(
        [r.message for r in warnings]
    )


@pytest.mark.asyncio
async def test_real_polymarket_adapter_counts_decode_failures():
    """Regression: the REAL PolymarketAdapter overrides __init__; a malformed
    frame must not crash with AttributeError on a missing decode_failures.

    The _Tiny stub used above inherits the base init, so it never exercised the
    super()-skipping subclass. PolymarketAdapter does — the first bad live frame
    would have raised AttributeError and killed the PM stream. decode_failures is
    now a class-attribute default + PM calls super().__init__()."""
    from hlanalysis.adapters.polymarket import PolymarketAdapter

    a = PolymarketAdapter(ws_factory=lambda url: None, gamma_client=object())
    assert a.decode_failures == 0  # attribute exists pre-increment
    a.stale_timeout_s = 0.01  # return promptly once the stream goes silent

    n_bad = 3
    await asyncio.wait_for(
        a._recv_until_stale(_MalformedFramesWS(n_bad), lambda m, t: [], "x", asyncio.Queue()),
        timeout=2.0,
    )
    assert a.decode_failures == n_bad


@pytest.mark.parametrize(
    "adapter_path",
    [
        "hlanalysis.adapters.polymarket:PolymarketAdapter",
        "hlanalysis.adapters.binance:BinanceAdapter",
        "hlanalysis.adapters.hyperliquid:HyperliquidAdapter",
    ],
)
def test_all_adapters_initialize_decode_failures(adapter_path):
    """Every concrete adapter must expose decode_failures == 0 on a fresh
    instance, regardless of whether it overrides __init__."""
    import importlib

    mod_name, cls_name = adapter_path.split(":")
    cls = getattr(importlib.import_module(mod_name), cls_name)
    # default-constructible with stubs; only the attribute presence matters
    try:
        inst = cls()
    except TypeError:
        inst = cls(ws_factory=lambda url: None, gamma_client=object())
    assert inst.decode_failures == 0


# ---------------------------------------------------------------------------
# Fix 1: hyperliquid._fetch_outcome_meta must pass a (connect, read) tuple.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hyperliquid_fetch_outcome_meta_passes_tuple_timeout(monkeypatch):
    """HyperliquidAdapter._fetch_outcome_meta must use a (connect, read) tuple.

    We monkeypatch requests.post (used inside asyncio.to_thread) to capture
    the kwargs. A scalar timeout alone can't bound the TLS handshake phase.
    """
    from hlanalysis.adapters.hyperliquid import HyperliquidAdapter

    captured: list[dict] = []

    def fake_post(url, **kwargs):
        captured.append(kwargs)

        class _R:
            def raise_for_status(self):
                pass

            def json(self):
                return {"outcomes": [], "questions": []}

        return _R()

    monkeypatch.setattr(requests, "post", fake_post)

    async def fake_to_thread(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)

    adapter = HyperliquidAdapter()
    result = await adapter._fetch_outcome_meta()
    assert result is not None, "expected a dict back from mocked _fetch_outcome_meta"

    assert captured, "requests.post was never called"
    timeout = captured[0].get("timeout")
    assert isinstance(timeout, tuple), f"expected tuple timeout, got {timeout!r}"
    assert len(timeout) == 2
    connect_t, read_t = timeout
    assert connect_t > 0 and read_t > 0
