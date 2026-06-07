"""Tests for Task 5: ETHUSDT_SPOT reference feed alongside BTCUSDT_SPOT.

Verifies that:
- binance_spot_reference_subscription(symbol) is parameterized (accepts BTCUSDT
  and ETHUSDT).
- build_engine_subscriptions emits both BTC and ETH spot reference subs.
- _remap_reference_symbol remaps both ETHUSDT and BTCUSDT SPOT events to their
  internal _SPOT suffixed symbols.

The _make_spot_bbo helper constructs a real BboEvent (same event type used by
the existing BTC remap test in tests/unit/test_engine_ingest_remap.py).
"""
from __future__ import annotations

from hlanalysis.engine.main import binance_spot_reference_subscription, build_engine_subscriptions
from hlanalysis.engine.runtime import _remap_reference_symbol
from hlanalysis.events import BboEvent, Mechanism, ProductType


def _make_spot_bbo(symbol: str) -> BboEvent:
    """Construct a real Binance SPOT BboEvent for the given symbol.

    Mirrors the _btcusdt_spot_bbo helper in test_engine_reference_feed.py and
    the events used in test_engine_ingest_remap.py — same event type and fields,
    parameterized by symbol so both BTC and ETH can be tested.
    """
    return BboEvent(
        venue="binance",
        product_type=ProductType.SPOT,
        mechanism=Mechanism.CLOB,
        symbol=symbol,
        exchange_ts=1,
        local_recv_ts=1,
        bid_px=1000.0,
        bid_sz=1.0,
        ask_px=1001.0,
        ask_sz=1.0,
    )


def test_binance_spot_reference_subscription_accepts_btcusdt():
    sub = binance_spot_reference_subscription("BTCUSDT")
    assert sub.venue == "binance"
    assert sub.product_type == ProductType.SPOT
    assert sub.symbol == "BTCUSDT"
    assert sub.channels == ("bbo",)


def test_binance_spot_reference_subscription_accepts_ethusdt():
    sub = binance_spot_reference_subscription("ETHUSDT")
    assert sub.venue == "binance"
    assert sub.product_type == ProductType.SPOT
    assert sub.symbol == "ETHUSDT"
    assert sub.channels == ("bbo",)


def test_both_spot_reference_feeds_present():
    subs = [
        binance_spot_reference_subscription("BTCUSDT"),
        binance_spot_reference_subscription("ETHUSDT"),
    ]
    syms = {s.symbol for s in subs}
    assert syms == {"BTCUSDT", "ETHUSDT"}


def test_build_engine_subscriptions_includes_both_spot_references():
    from hlanalysis.config import RecorderConfig
    cfg = RecorderConfig(subscriptions=[])
    subs = build_engine_subscriptions(cfg)
    binance = [(s.product_type, s.symbol) for s in subs if s.venue == "binance"]
    assert (ProductType.SPOT, "BTCUSDT") in binance
    assert (ProductType.SPOT, "ETHUSDT") in binance
    assert (ProductType.PERP, "BTCUSDT") not in binance


def test_eth_spot_remapped_to_ethusdt_spot():
    out = _remap_reference_symbol(_make_spot_bbo("ETHUSDT"))
    assert out.symbol == "ETHUSDT_SPOT"


def test_btc_spot_still_remapped_to_btcusdt_spot():
    out = _remap_reference_symbol(_make_spot_bbo("BTCUSDT"))
    assert out.symbol == "BTCUSDT_SPOT"


def test_remap_eth_returns_new_object_not_mutation():
    ev = _make_spot_bbo("ETHUSDT")
    out = _remap_reference_symbol(ev)
    assert out is not ev
    assert ev.symbol == "ETHUSDT"  # original untouched


def test_unknown_spot_symbol_not_remapped():
    """Spot symbols NOT in the known map (SOLUSDT, etc.) are left untouched."""
    ev = _make_spot_bbo("SOLUSDT")
    out = _remap_reference_symbol(ev)
    assert out.symbol == "SOLUSDT"


def test_non_binance_eth_spot_not_remapped():
    """Only binance-venue spot events are remapped."""
    ev = BboEvent(
        venue="other_venue",
        product_type=ProductType.SPOT,
        mechanism=Mechanism.CLOB,
        symbol="ETHUSDT",
        exchange_ts=1,
        local_recv_ts=1,
        bid_px=1000.0,
        bid_sz=1.0,
        ask_px=1001.0,
        ask_sz=1.0,
    )
    out = _remap_reference_symbol(ev)
    assert out.symbol == "ETHUSDT"
