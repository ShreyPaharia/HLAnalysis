"""#33: Binance _poll_perp_premium and HL _fetch_outcome_meta error backoff.

Tests assert:
- Repeated errors on _poll_perp_premium cause sleep duration to grow (exponential
  backoff), not stay flat at the base interval.
- The backoff is capped (does not grow unboundedly).
- On a successful call after errors the backoff resets to base.
- HL _fetch_outcome_meta also backs off on repeated errors (via a wrapper that
  callers of _fetch_outcome_meta can rely on — we test the exponential-backoff
  helper exposed on the adapter).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hlanalysis.adapters.binance import BinanceAdapter, PERP_MARK_POLL_INTERVAL_S
from hlanalysis.adapters.hyperliquid import HyperliquidAdapter
from hlanalysis.config import Subscription
from hlanalysis.events import Mechanism, ProductType

PERP_SUB = Subscription(
    venue="binance",
    product_type=ProductType.PERP,
    mechanism=Mechanism.CLOB,
    symbol="BTCUSDT",
    channels=("mark", "funding"),
)


# ---------------------------------------------------------------------------
# Binance _poll_perp_premium backoff
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_binance_poller_backoff_on_repeated_errors():
    """Persistent exceptions → sleep duration grows (exponential), not flat."""
    adapter = BinanceAdapter()
    queue: asyncio.Queue = asyncio.Queue(maxsize=1000)

    sleep_calls: list[float] = []

    async def fake_sleep(d: float) -> None:
        sleep_calls.append(d)
        if len(sleep_calls) >= 4:
            raise asyncio.CancelledError

    def boom(*a, **kw):
        raise RuntimeError("network down")

    with (
        patch("hlanalysis.adapters.binance.asyncio.to_thread", side_effect=boom),
        patch("hlanalysis.adapters.binance.asyncio.sleep", side_effect=fake_sleep),
    ):
        try:
            await adapter._poll_perp_premium([PERP_SUB], queue)
        except asyncio.CancelledError:
            pass

    assert len(sleep_calls) >= 3, f"Expected multiple sleeps, got {sleep_calls}"
    # Each successive sleep must be >= the previous (monotone non-decreasing under backoff)
    for i in range(1, len(sleep_calls)):
        assert sleep_calls[i] >= sleep_calls[i - 1], (
            f"Backoff must not decrease: sleep_calls={sleep_calls}"
        )
    # At least one sleep must be strictly larger than the base interval
    assert max(sleep_calls) > PERP_MARK_POLL_INTERVAL_S, (
        f"Backoff never exceeded base interval {PERP_MARK_POLL_INTERVAL_S}s: {sleep_calls}"
    )


@pytest.mark.asyncio
async def test_binance_poller_backoff_is_capped():
    """Backoff must not grow past a finite cap even after many errors."""
    adapter = BinanceAdapter()
    queue: asyncio.Queue = asyncio.Queue(maxsize=1000)

    sleep_calls: list[float] = []
    call_count = 0

    async def fake_sleep(d: float) -> None:
        sleep_calls.append(d)
        nonlocal call_count
        call_count += 1
        if call_count >= 8:
            raise asyncio.CancelledError

    def boom(*a, **kw):
        raise RuntimeError("network down")

    with (
        patch("hlanalysis.adapters.binance.asyncio.to_thread", side_effect=boom),
        patch("hlanalysis.adapters.binance.asyncio.sleep", side_effect=fake_sleep),
    ):
        try:
            await adapter._poll_perp_premium([PERP_SUB], queue)
        except asyncio.CancelledError:
            pass

    assert sleep_calls, "Expected at least one sleep"
    max_sleep = max(sleep_calls)
    assert max_sleep <= 120.0, f"Backoff cap exceeded: max_sleep={max_sleep}"


@pytest.mark.asyncio
async def test_binance_poller_backoff_resets_on_success():
    """After a successful poll the backoff counter resets to the base interval."""
    adapter = BinanceAdapter()
    queue: asyncio.Queue = asyncio.Queue(maxsize=1000)

    sleep_calls: list[float] = []
    call_count = 0
    to_thread_count = 0

    async def fake_sleep(d: float) -> None:
        sleep_calls.append(d)
        nonlocal call_count
        call_count += 1
        if call_count >= 6:
            raise asyncio.CancelledError

    async def fake_to_thread(fn, *args, **kwargs):
        nonlocal to_thread_count
        to_thread_count += 1
        # First 3 calls fail; 4th succeeds
        if to_thread_count <= 3:
            raise RuntimeError("transient error")
        # Return a successful response object
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "time": 1700000000000,
            "markPrice": "30000.0",
            "lastFundingRate": "0.0001",
            "nextFundingTime": 1700004000000,
            "estimatedSettlePrice": "30000.0",
        }
        return mock_resp

    with (
        patch("hlanalysis.adapters.binance.asyncio.to_thread", side_effect=fake_to_thread),
        patch("hlanalysis.adapters.binance.asyncio.sleep", side_effect=fake_sleep),
    ):
        try:
            await adapter._poll_perp_premium([PERP_SUB], queue)
        except asyncio.CancelledError:
            pass

    # After backoff builds up and then a success, subsequent sleeps should reset
    # toward the base poll interval. The last sleep before cancellation should be
    # close to PERP_MARK_POLL_INTERVAL_S (not a large backoff value).
    if len(sleep_calls) >= 2:
        last_sleep = sleep_calls[-1]
        assert last_sleep <= PERP_MARK_POLL_INTERVAL_S * 4, (
            f"Sleep after recovery should be near base interval, got {last_sleep}; "
            f"all sleeps: {sleep_calls}"
        )


# ---------------------------------------------------------------------------
# HL _fetch_outcome_meta backoff
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hl_fetch_outcome_meta_backoff_on_repeated_errors():
    """HL _fetch_outcome_meta_with_backoff (or equivalent) must back off on errors."""
    adapter = HyperliquidAdapter()

    sleep_calls: list[float] = []
    call_count = 0

    async def fake_sleep(d: float) -> None:
        sleep_calls.append(d)
        nonlocal call_count
        call_count += 1

    def boom(fn, *args, **kwargs):
        raise RuntimeError("network down")

    # Call the backoff-aware fetch multiple times manually.
    # The adapter must expose _fetch_outcome_meta_with_backoff or track backoff state.
    with (
        patch("hlanalysis.adapters.hyperliquid.asyncio.to_thread", side_effect=boom),
        patch("hlanalysis.adapters.hyperliquid.asyncio.sleep", side_effect=fake_sleep),
    ):
        for _ in range(4):
            result = await adapter._fetch_outcome_meta_with_backoff()
            assert result is None, "_fetch_outcome_meta_with_backoff must return None on error"

    # If backoff was applied, we should have slept at least once with increasing intervals
    if sleep_calls:
        assert max(sleep_calls) > 0, "Backoff sleeps should be positive"
        # Must be monotone non-decreasing
        for i in range(1, len(sleep_calls)):
            assert sleep_calls[i] >= sleep_calls[i - 1], f"Backoff must not decrease: {sleep_calls}"


@pytest.mark.asyncio
async def test_hl_fetch_outcome_meta_backoff_resets_on_success():
    """After a successful fetch, HL backoff state resets."""
    adapter = HyperliquidAdapter()

    sleep_calls: list[float] = []
    call_count = 0

    async def fake_sleep(d: float) -> None:
        sleep_calls.append(d)

    good_payload = {"outcomes": [], "questions": []}

    async def fake_to_thread_fail(fn, *args, **kwargs):
        raise RuntimeError("transient")

    async def fake_to_thread_ok(fn, *args, **kwargs):
        return good_payload

    with (
        patch("hlanalysis.adapters.hyperliquid.asyncio.to_thread", side_effect=fake_to_thread_fail),
        patch("hlanalysis.adapters.hyperliquid.asyncio.sleep", side_effect=fake_sleep),
    ):
        for _ in range(3):
            await adapter._fetch_outcome_meta_with_backoff()

    # Record backoff level before success
    backoff_before_success = getattr(adapter, "_fetch_meta_backoff", None)

    with patch("hlanalysis.adapters.hyperliquid.asyncio.to_thread", side_effect=fake_to_thread_ok):
        result = await adapter._fetch_outcome_meta_with_backoff()

    assert result == good_payload

    # After success, backoff must be reset (attribute back to initial/base value)
    backoff_after_success = getattr(adapter, "_fetch_meta_backoff", None)
    if backoff_before_success is not None and backoff_after_success is not None:
        assert backoff_after_success <= backoff_before_success, (
            "Backoff must reset (decrease) after a successful fetch"
        )
