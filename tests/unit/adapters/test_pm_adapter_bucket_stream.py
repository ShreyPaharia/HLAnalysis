"""Tests: live PM adapter emits priceBucket meta (one per multi-strike event)
and leaves the priceBinary path unchanged (regression).

Harness mirrors tests/unit/test_pm_adapter.py:
- GammaClient(http_get=...) is the network-injection point.
- A _FakeWS([]) that raises ConnectionClosedOK immediately terminates the
  WS loop so the test only exercises the Gamma-poll path and collects the
  QuestionMetaEvents that land on the queue before the WS exits.
- asyncio.wait_for / pytest-asyncio drives the async test.
"""
from __future__ import annotations

import asyncio
import json

import pytest
import websockets.exceptions

from hlanalysis.adapters.polymarket import PolymarketAdapter
from hlanalysis.adapters.polymarket_gamma import GammaClient
from hlanalysis.config import Subscription
from hlanalysis.events import Mechanism, ProductType, QuestionMetaEvent


# ---------------------------------------------------------------------------
# Shared fixture data
# ---------------------------------------------------------------------------


def _bucket_event():
    """One multi-strike event with two sub-markets (out of order by strike)."""
    def mk(s, y, n, c):
        return {
            "conditionId": c,
            "groupItemTitle": f"${s:,.0f}",
            "clobTokenIds": json.dumps([y, n]),
            "outcomePrices": json.dumps(["0", "0"]),
            "endDate": "2026-06-12T16:00:00Z",
        }
    return [
        {
            "slug": "btc-multi-strikes-weekly-x",
            "startDate": "2026-06-08T16:00:00Z",
            "endDate": "2026-06-12T16:00:00Z",
            "markets": [
                mk(80000, "y80", "n80", "c80"),
                mk(90000, "y90", "n90", "c90"),
            ],
        }
    ]


# ---------------------------------------------------------------------------
# Minimal fake WS that closes immediately (no frames needed for gamma-only path)
# ---------------------------------------------------------------------------


class _FakeWS:
    """Async context manager that raises ConnectionClosedOK on first recv().

    This causes the adapter's WS loop to exit cleanly so the test can collect
    all events that were enqueued during the preceding Gamma poll.
    """

    def __init__(self):
        self.sent: list[str] = []

    async def send(self, data: str) -> None:
        self.sent.append(data)

    async def recv(self) -> str:
        await asyncio.sleep(0)
        raise websockets.exceptions.ConnectionClosedOK(None, None)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


# ---------------------------------------------------------------------------
# Test harness: run one Gamma poll and collect QuestionMetaEvents
# ---------------------------------------------------------------------------


def _run_one_poll(gamma: GammaClient, *, klass: str, series: str, underlying: str) -> list[QuestionMetaEvent]:
    """Construct an adapter with the given gamma client and subscription, drive
    one Gamma poll (via stream() which calls _gamma_poll_once on startup),
    collect all QuestionMetaEvents emitted before the WS closes, and return them.
    """
    sub = Subscription(
        venue="polymarket",
        product_type=ProductType.PREDICTION_BINARY,
        mechanism=Mechanism.CLOB,
        symbol="*",
        channels=("trades", "book"),
        match={"class": klass, "series_slug": series, "underlying": underlying},
    )
    adapter = PolymarketAdapter(
        ws_factory=lambda url: _FakeWS(),
        gamma_client=gamma,
    )
    metas: list[QuestionMetaEvent] = []

    async def _drain() -> None:
        try:
            async for ev in adapter.stream([sub]):
                if isinstance(ev, QuestionMetaEvent):
                    metas.append(ev)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass

    asyncio.run(asyncio.wait_for(_drain(), timeout=5.0))
    return metas


# ---------------------------------------------------------------------------
# Task 3 tests
# ---------------------------------------------------------------------------


def test_bucket_series_emits_one_pricebucket_meta_with_all_legs():
    """A priceBucket subscription should produce exactly one QuestionMetaEvent
    with class=priceBucket, carrying all leg token IDs in ascending-strike order
    and bucketLayout=above_ladder."""
    gamma = GammaClient(http_get=lambda url, params: _bucket_event())
    metas = _run_one_poll(gamma, klass="priceBucket",
                          series="btc-multi-strikes-weekly", underlying="BTC")
    bucket_metas = [m for m in metas if dict(zip(m.keys, m.values)).get("class") == "priceBucket"]
    assert len(bucket_metas) == 1, f"expected 1 priceBucket meta, got {len(bucket_metas)}"
    kv = dict(zip(bucket_metas[0].keys, bucket_metas[0].values))
    assert kv["leg_token_ids"] == "y80,n80,y90,n90"
    assert kv["bucketLayout"] == "above_ladder"


def test_updown_series_still_emits_per_market_binaries():
    """A priceBinary subscription (legacy up/down path) must still emit one
    QuestionMetaEvent per sub-market — regression guard for the existing path."""
    gamma = GammaClient(http_get=lambda url, params: _bucket_event())
    metas = _run_one_poll(gamma, klass="priceBinary",
                          series="btc-up-or-down-daily", underlying="BTC")
    classes = [dict(zip(m.keys, m.values)).get("class") for m in metas]
    assert classes == ["priceBinary", "priceBinary"], (
        f"expected 2 priceBinary metas (one per sub-market), got: {classes}"
    )
