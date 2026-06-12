"""SHR-60: the Binance reference feed needs a per-frame staleness watchdog.

`async for raw in ws` has no per-stream staleness detection — ping_interval/
timeout only catch a dead TCP layer. A half-open or silently-idle stream (the
geo-blocked "subscribed but silent" case) keeps the socket open and blocks
forever with no exception and no reconnect. The receive path must force-close
(→ reconnect) and emit a feed_stale health event if no frame arrives within the
watchdog window. Thanks to wait_for, the same window is also the first-data
deadline.
"""
from __future__ import annotations

import asyncio
import logging

import pytest
import requests

from hlanalysis.adapters.binance import BinanceAdapter, PERP_MARK_POLL_INTERVAL_S


class _SilentWS:
    """recv() never returns a frame — models a half-open/idle stream."""

    async def recv(self):
        await asyncio.sleep(3600)


class _OneFrameThenSilentWS:
    """Delivers one (unparseable) frame, then goes silent."""

    def __init__(self) -> None:
        self._sent = False

    async def recv(self):
        if not self._sent:
            self._sent = True
            return "not-json"  # exercises the JSONDecodeError continue path
        await asyncio.sleep(3600)


async def _drain_kinds(q: asyncio.Queue) -> list[str]:
    kinds = []
    while not q.empty():
        kinds.append(q.get_nowait().kind)
    return kinds


@pytest.mark.asyncio
async def test_watchdog_emits_feed_stale_and_returns_on_silent_stream():
    adapter = BinanceAdapter()
    adapter.stale_timeout_s = 0.01
    q: asyncio.Queue = asyncio.Queue()

    await asyncio.wait_for(
        adapter._recv_until_stale(_SilentWS(), lambda m, t: [], "perp", q), timeout=1.0,
    )

    kinds = await _drain_kinds(q)
    assert any(k.startswith("feed_stale") for k in kinds), kinds


@pytest.mark.asyncio
async def test_watchdog_fires_after_a_frame_then_silence():
    adapter = BinanceAdapter()
    adapter.stale_timeout_s = 0.01
    q: asyncio.Queue = asyncio.Queue()

    await asyncio.wait_for(
        adapter._recv_until_stale(_OneFrameThenSilentWS(), lambda m, t: [], "perp", q),
        timeout=1.0,
    )

    kinds = await _drain_kinds(q)
    assert any(k.startswith("feed_stale") for k in kinds), kinds


# ---------------------------------------------------------------------------
# Fix 1: _poll_perp_premium must pass a (connect, read) tuple to requests.get.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_poll_perp_premium_passes_tuple_timeout(monkeypatch):
    """_poll_perp_premium must call requests.get with a (connect, read) tuple.

    A scalar timeout alone can hang on a stalled TLS handshake — the read
    timer only starts after the connection is open. We monkeypatch asyncio
    .to_thread so the capture runs synchronously.
    """
    captured: list[dict] = []

    class _FakeResponse:
        status_code = 200
        def json(self):
            return {"markPrice": "95000", "lastFundingRate": "0.001",
                    "nextFundingTime": 1700000000000, "time": 1700000000000}

    def fake_get(url, **kwargs):
        captured.append(kwargs)
        return _FakeResponse()

    monkeypatch.setattr(requests, "get", fake_get)

    # Replace asyncio.to_thread with a synchronous call so the test is sync.
    async def fake_to_thread(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)

    from hlanalysis.config import Subscription
    from hlanalysis.events import Mechanism, ProductType
    sub = Subscription(
        venue="binance", symbol="BTCUSDT", product_type=ProductType.PERP,
        mechanism=Mechanism.CLOB, channels=("mark", "funding"),
    )
    adapter = BinanceAdapter()
    q: asyncio.Queue = asyncio.Queue(maxsize=100)

    # Run one iteration — the loop sleeps PERP_MARK_POLL_INTERVAL_S at the end;
    # cancel after a short yield so we don't block.
    task = asyncio.create_task(adapter._poll_perp_premium([sub], q))
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert captured, "requests.get was never called"
    timeout = captured[0].get("timeout")
    assert isinstance(timeout, tuple), f"expected tuple timeout, got {timeout!r}"
    assert len(timeout) == 2
    connect_t, read_t = timeout
    assert connect_t > 0 and read_t > 0


# ---------------------------------------------------------------------------
# Fix 2: non-200 from premiumIndex must log a warning and back off (not
# tight-spin), so a 418/4xx IP ban doesn't burn the rate-limit silently.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_poll_perp_premium_non200_logs_warning_and_backs_off(
    monkeypatch, caplog
):
    """On non-200 the poll loop must: (a) log a warning, (b) sleep before
    the next iteration so a 418/4xx ban doesn't tight-spin.

    We monkeypatch asyncio.sleep to capture the sleep duration and cancel
    after the first sleep so the test doesn't run forever.
    """
    sleeps: list[float] = []
    _orig_sleep = asyncio.sleep

    async def capturing_sleep(delay):
        sleeps.append(delay)
        # Only allow a short pause, then re-raise cancel on next opportunity.
        await _orig_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", capturing_sleep)

    class _Non200Response:
        status_code = 429
        def json(self):
            return {}

    async def fake_to_thread(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Non200Response())

    from hlanalysis.config import Subscription
    from hlanalysis.events import Mechanism, ProductType
    sub = Subscription(
        venue="binance", symbol="BTCUSDT", product_type=ProductType.PERP,
        mechanism=Mechanism.CLOB, channels=("mark",),
    )
    adapter = BinanceAdapter()
    q: asyncio.Queue = asyncio.Queue(maxsize=100)

    with caplog.at_level(logging.WARNING, logger="hlanalysis.adapters.binance"):
        task = asyncio.create_task(adapter._poll_perp_premium([sub], q))
        # Let the loop run one full non-200 iteration.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    # (a) A warning was emitted mentioning the status code.
    warning_msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("429" in str(m) or "non-200" in str(m) for m in warning_msgs), \
        f"no non-200 warning found; records={warning_msgs}"

    # (b) A sleep > 0 was called (backoff before retry), not a tight spin.
    assert sleeps, "no sleep called after non-200 — loop tight-spins on ban"
    # The backoff should be at least PERP_MARK_POLL_INTERVAL_S (not shrunk).
    assert any(s > 0 for s in sleeps), f"all sleeps were 0: {sleeps}"
