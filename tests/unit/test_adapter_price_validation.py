"""SHR-61 (#22): Binance BBO/partial-depth price validation + sequence-gap protection.

Tests assert:
- A crossed BBO frame (bid >= ask) is dropped (returns empty list).
- A zero/negative price BBO frame is dropped.
- A frame whose `seq` regresses below the last seen seq for that symbol is dropped.
- A valid frame passes through bit-identical.
- Drop counters increment; a log warning is emitted.
"""

from __future__ import annotations

import logging
import time

import pytest

from hlanalysis.adapters.binance import BinanceAdapter
from hlanalysis.config import Subscription
from hlanalysis.events import BboEvent, BookSnapshotEvent, Mechanism, ProductType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SPOT_SUB = Subscription(
    venue="binance",
    product_type=ProductType.SPOT,
    mechanism=Mechanism.CLOB,
    symbol="BTCUSDT",
    channels=("bbo",),
)
PERP_SUB = Subscription(
    venue="binance",
    product_type=ProductType.PERP,
    mechanism=Mechanism.CLOB,
    symbol="BTCUSDT",
    channels=("bbo",),
)


def _adapter() -> BinanceAdapter:
    return BinanceAdapter()


def _recv_ns() -> int:
    return time.time_ns()


def _spot_bbo_msg(bid: float, ask: float, seq: int = 1) -> dict:
    """Spot @bookTicker: no 'e' field."""
    return {"u": seq, "s": "BTCUSDT", "b": str(bid), "B": "1.0", "a": str(ask), "A": "1.0"}


def _perp_bbo_msg(bid: float, ask: float, seq: int = 1) -> dict:
    """Perp @bookTicker: has 'e' == 'bookTicker'."""
    return {
        "e": "bookTicker",
        "u": seq,
        "s": "BTCUSDT",
        "b": str(bid),
        "B": "1.0",
        "a": str(ask),
        "A": "1.0",
        "T": 1700000000,
    }


def _spot_depth_msg(bids: list[list[str]], asks: list[list[str]], seq: int = 1) -> dict:
    """Spot partial-depth: no 'e', has 'lastUpdateId'."""
    return {"lastUpdateId": seq, "bids": bids, "asks": asks}


# ---------------------------------------------------------------------------
# SHR-61: crossed / non-positive price → drop
# ---------------------------------------------------------------------------


class TestSpotBboPriceValidation:
    """Spot @bookTicker path."""

    def _handle(self, adapter: BinanceAdapter, msg: dict) -> list:
        sym_to_sub = {"BTCUSDT": SPOT_SUB}
        return adapter._handle(msg, _recv_ns(), sym_to_sub, "spot", ProductType.SPOT)

    def test_valid_frame_passes_unchanged(self):
        adapter = _adapter()
        out = self._handle(adapter, _spot_bbo_msg(bid=30000.0, ask=30001.0, seq=5))
        assert len(out) == 1
        ev = out[0]
        assert isinstance(ev, BboEvent)
        assert ev.bid_px == 30000.0
        assert ev.ask_px == 30001.0
        assert ev.seq == 5

    def test_crossed_frame_dropped(self):
        """bid >= ask → invalid, must be silently skipped."""
        adapter = _adapter()
        out = self._handle(adapter, _spot_bbo_msg(bid=30001.0, ask=30000.0))
        assert out == [], "crossed BBO must be dropped"

    def test_equal_bid_ask_dropped(self):
        """bid == ask is also crossed (zero spread is suspicious for BBO)."""
        adapter = _adapter()
        out = self._handle(adapter, _spot_bbo_msg(bid=30000.0, ask=30000.0))
        assert out == [], "zero-spread BBO must be dropped"

    def test_zero_bid_dropped(self):
        adapter = _adapter()
        out = self._handle(adapter, _spot_bbo_msg(bid=0.0, ask=30001.0))
        assert out == [], "zero bid must be dropped"

    def test_negative_ask_dropped(self):
        adapter = _adapter()
        out = self._handle(adapter, _spot_bbo_msg(bid=30000.0, ask=-1.0))
        assert out == [], "negative ask must be dropped"

    def test_seq_regression_dropped(self):
        """Sending a lower seq after a higher seq → stale frame, must drop."""
        adapter = _adapter()
        sym_to_sub = {"BTCUSDT": SPOT_SUB}
        # First frame: seq=10 — accepted
        out1 = adapter._handle(_spot_bbo_msg(30000.0, 30001.0, seq=10), _recv_ns(), sym_to_sub, "spot", ProductType.SPOT)
        assert len(out1) == 1, "first valid frame must pass"
        # Second frame: seq=5 < 10 — stale, must be dropped
        out2 = adapter._handle(_spot_bbo_msg(30000.0, 30001.0, seq=5), _recv_ns(), sym_to_sub, "spot", ProductType.SPOT)
        assert out2 == [], "out-of-order (lower seq) frame must be dropped"

    def test_seq_equal_to_last_dropped(self):
        """Exact-same seq on reconnect replay → duplicate, must drop."""
        adapter = _adapter()
        sym_to_sub = {"BTCUSDT": SPOT_SUB}
        adapter._handle(_spot_bbo_msg(30000.0, 30001.0, seq=10), _recv_ns(), sym_to_sub, "spot", ProductType.SPOT)
        out = adapter._handle(_spot_bbo_msg(30000.0, 30001.0, seq=10), _recv_ns(), sym_to_sub, "spot", ProductType.SPOT)
        assert out == [], "repeated seq must be dropped"

    def test_seq_advance_accepted(self):
        """seq > last → normal progression, must pass."""
        adapter = _adapter()
        sym_to_sub = {"BTCUSDT": SPOT_SUB}
        adapter._handle(_spot_bbo_msg(30000.0, 30001.0, seq=10), _recv_ns(), sym_to_sub, "spot", ProductType.SPOT)
        out = adapter._handle(_spot_bbo_msg(30000.0, 30001.0, seq=11), _recv_ns(), sym_to_sub, "spot", ProductType.SPOT)
        assert len(out) == 1

    def test_drop_counter_increments(self):
        adapter = _adapter()
        assert adapter.bbo_drops == 0
        self._handle(adapter, _spot_bbo_msg(bid=30001.0, ask=30000.0))  # crossed
        self._handle(adapter, _spot_bbo_msg(bid=0.0, ask=30001.0))  # zero
        assert adapter.bbo_drops == 2

    def test_seq_regression_drop_counter_increments(self):
        adapter = _adapter()
        sym_to_sub = {"BTCUSDT": SPOT_SUB}
        adapter._handle(_spot_bbo_msg(30000.0, 30001.0, seq=10), _recv_ns(), sym_to_sub, "spot", ProductType.SPOT)
        before = adapter.bbo_drops
        adapter._handle(_spot_bbo_msg(30000.0, 30001.0, seq=5), _recv_ns(), sym_to_sub, "spot", ProductType.SPOT)
        assert adapter.bbo_drops == before + 1


class TestPerpBboPriceValidation:
    """Perp @bookTicker path (has 'e' == 'bookTicker')."""

    def _handle(self, adapter: BinanceAdapter, msg: dict) -> list:
        sym_to_sub = {"BTCUSDT": PERP_SUB}
        return adapter._handle(msg, _recv_ns(), sym_to_sub, "perp", ProductType.PERP)

    def test_valid_frame_passes_unchanged(self):
        adapter = _adapter()
        out = self._handle(adapter, _perp_bbo_msg(30000.0, 30001.0, seq=1))
        assert len(out) == 1
        assert isinstance(out[0], BboEvent)
        assert out[0].bid_px == 30000.0

    def test_crossed_frame_dropped(self):
        adapter = _adapter()
        out = self._handle(adapter, _perp_bbo_msg(bid=30001.0, ask=30000.0))
        assert out == []

    def test_zero_price_dropped(self):
        adapter = _adapter()
        out = self._handle(adapter, _perp_bbo_msg(bid=0.0, ask=30001.0))
        assert out == []

    def test_seq_regression_dropped(self):
        adapter = _adapter()
        sym_to_sub = {"BTCUSDT": PERP_SUB}
        adapter._handle(_perp_bbo_msg(30000.0, 30001.0, seq=20), _recv_ns(), sym_to_sub, "perp", ProductType.PERP)
        out = adapter._handle(_perp_bbo_msg(30000.0, 30001.0, seq=10), _recv_ns(), sym_to_sub, "perp", ProductType.PERP)
        assert out == []


class TestSpotDepthPriceValidation:
    """Spot partial-depth path."""

    def _handle(self, adapter: BinanceAdapter, msg: dict) -> list:
        sym_to_sub = {"BTCUSDT": SPOT_SUB}
        return adapter._handle(msg, _recv_ns(), sym_to_sub, "spot", ProductType.SPOT)

    def test_valid_frame_passes(self):
        adapter = _adapter()
        msg = _spot_depth_msg(bids=[["30000.0", "1.0"]], asks=[["30001.0", "1.0"]], seq=1)
        out = self._handle(adapter, msg)
        assert len(out) == 1
        assert isinstance(out[0], BookSnapshotEvent)

    def test_seq_regression_dropped(self):
        adapter = _adapter()
        sym_to_sub = {"BTCUSDT": SPOT_SUB}
        msg_hi = _spot_depth_msg([["30000.0", "1.0"]], [["30001.0", "1.0"]], seq=100)
        adapter._handle(msg_hi, _recv_ns(), sym_to_sub, "spot", ProductType.SPOT)
        msg_lo = _spot_depth_msg([["30000.0", "1.0"]], [["30001.0", "1.0"]], seq=50)
        out = adapter._handle(msg_lo, _recv_ns(), sym_to_sub, "spot", ProductType.SPOT)
        assert out == []

    def test_zero_price_in_bids_dropped(self):
        """Any non-positive price in the order book → drop whole frame."""
        adapter = _adapter()
        msg = _spot_depth_msg(bids=[["0.0", "1.0"]], asks=[["30001.0", "1.0"]], seq=1)
        out = self._handle(adapter, msg)
        assert out == []

    def test_drop_counter_increments_on_depth_drop(self):
        adapter = _adapter()
        sym_to_sub = {"BTCUSDT": SPOT_SUB}
        msg_hi = _spot_depth_msg([["30000.0", "1.0"]], [["30001.0", "1.0"]], seq=100)
        adapter._handle(msg_hi, _recv_ns(), sym_to_sub, "spot", ProductType.SPOT)
        before = adapter.bbo_drops
        msg_lo = _spot_depth_msg([["30000.0", "1.0"]], [["30001.0", "1.0"]], seq=50)
        adapter._handle(msg_lo, _recv_ns(), sym_to_sub, "spot", ProductType.SPOT)
        assert adapter.bbo_drops == before + 1


class TestDropLogging:
    """A drop must emit at least one log warning (not crash)."""

    def test_crossed_frame_logs_warning(self, caplog):
        adapter = _adapter()
        sym_to_sub = {"BTCUSDT": SPOT_SUB}
        with caplog.at_level(logging.WARNING, logger="hlanalysis.adapters.binance"):
            adapter._handle(_spot_bbo_msg(30001.0, 30000.0), _recv_ns(), sym_to_sub, "spot", ProductType.SPOT)
        assert any(
            "drop" in r.message.lower() or "cross" in r.message.lower() or "invalid" in r.message.lower()
            for r in caplog.records
        ), "Expected a warning log on BBO drop"
