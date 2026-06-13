"""Tests for _remap_reference_symbol: Binance SPOT BTCUSDT events must be
remapped to BTCUSDT_SPOT on ingest so they don't overwrite the perp BTCUSDT
book key in MarketState."""

from __future__ import annotations

from hlanalysis.engine.runtime import _remap_reference_symbol
from hlanalysis.events import BboEvent, Mechanism, ProductType


def test_spot_btcusdt_bbo_is_remapped_to_btcusdt_spot():
    ev = BboEvent(
        venue="binance",
        product_type=ProductType.SPOT,
        mechanism=Mechanism.CLOB,
        symbol="BTCUSDT",
        exchange_ts=1,
        local_recv_ts=1,
        bid_px=73644.0,
        bid_sz=1.0,
        ask_px=73645.0,
        ask_sz=1.0,
    )
    out = _remap_reference_symbol(ev)
    assert out.symbol == "BTCUSDT_SPOT"


def test_perp_btcusdt_bbo_is_left_unchanged():
    ev = BboEvent(
        venue="binance",
        product_type=ProductType.PERP,
        mechanism=Mechanism.CLOB,
        symbol="BTCUSDT",
        exchange_ts=1,
        local_recv_ts=1,
        bid_px=73609.0,
        bid_sz=1.0,
        ask_px=73610.0,
        ask_sz=1.0,
    )
    out = _remap_reference_symbol(ev)
    assert out.symbol == "BTCUSDT"


def test_remap_returns_new_object_not_mutation():
    """Pydantic frozen model: remap must return a copy, not mutate in-place."""
    ev = BboEvent(
        venue="binance",
        product_type=ProductType.SPOT,
        mechanism=Mechanism.CLOB,
        symbol="BTCUSDT",
        exchange_ts=1,
        local_recv_ts=1,
        bid_px=73644.0,
        bid_sz=1.0,
        ask_px=73645.0,
        ask_sz=1.0,
    )
    out = _remap_reference_symbol(ev)
    assert out is not ev
    assert ev.symbol == "BTCUSDT"  # original untouched


def test_known_eth_spot_is_remapped_to_ethusdt_spot():
    """ETHUSDT spot is now also remapped (ETH reference feed added in Task 5)."""
    ev = BboEvent(
        venue="binance",
        product_type=ProductType.SPOT,
        mechanism=Mechanism.CLOB,
        symbol="ETHUSDT",
        exchange_ts=1,
        local_recv_ts=1,
        bid_px=3000.0,
        bid_sz=1.0,
        ask_px=3001.0,
        ask_sz=1.0,
    )
    out = _remap_reference_symbol(ev)
    assert out.symbol == "ETHUSDT_SPOT"


def test_unknown_spot_symbol_is_left_unchanged():
    """Only known reference symbols (BTC, ETH) are remapped; others are left alone."""
    ev = BboEvent(
        venue="binance",
        product_type=ProductType.SPOT,
        mechanism=Mechanism.CLOB,
        symbol="SOLUSDT",
        exchange_ts=1,
        local_recv_ts=1,
        bid_px=150.0,
        bid_sz=1.0,
        ask_px=150.1,
        ask_sz=1.0,
    )
    out = _remap_reference_symbol(ev)
    assert out.symbol == "SOLUSDT"


def test_non_binance_spot_btcusdt_is_left_unchanged():
    """Only binance venue spot BTCUSDT is remapped; other venues' spot is not."""
    ev = BboEvent(
        venue="other_venue",
        product_type=ProductType.SPOT,
        mechanism=Mechanism.CLOB,
        symbol="BTCUSDT",
        exchange_ts=1,
        local_recv_ts=1,
        bid_px=73644.0,
        bid_sz=1.0,
        ask_px=73645.0,
        ask_sz=1.0,
    )
    out = _remap_reference_symbol(ev)
    assert out.symbol == "BTCUSDT"
