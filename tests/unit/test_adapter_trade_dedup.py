"""#39: HL trade de-duplication on reconnect.

Tests assert:
- A trade with a previously-seen `tid` is not emitted a second time.
- The same-tid trade delivered on two separate calls → emitted exactly once.
- The bound on the seen-set is enforced: once the deque/set fills past the cap,
  the oldest id is evicted and can be re-emitted (this proves the cap is finite
  and old tids don't accumulate forever).
- Valid (unseen) trades always pass through.
"""

from __future__ import annotations

import time

import pytest

from hlanalysis.adapters.hyperliquid import HyperliquidAdapter
from hlanalysis.config import Subscription
from hlanalysis.events import Mechanism, ProductType, TradeEvent

PERP_SUB = Subscription(
    venue="hyperliquid",
    product_type=ProductType.PERP,
    mechanism=Mechanism.CLOB,
    symbol="BTC",
    channels=("trades",),
)


def _adapter() -> HyperliquidAdapter:
    return HyperliquidAdapter()


def _recv_ns() -> int:
    return time.time_ns()


def _trades_msg(trades: list[dict]) -> dict:
    return {"channel": "trades", "data": trades}


def _trade(tid: int, coin: str = "BTC") -> dict:
    return {
        "coin": coin,
        "px": "30000.0",
        "sz": "0.1",
        "side": "B",
        "time": 1700000000,
        "tid": tid,
        "users": ["0xbuyer", "0xseller"],
        "hash": f"0xhash{tid}",
    }


class TestHlTradeDedup:
    def _handle(self, adapter: HyperliquidAdapter, msg: dict) -> list[TradeEvent]:
        sym_to_sub = {"BTC": PERP_SUB}
        return [e for e in adapter._handle(msg, _recv_ns(), sym_to_sub) if isinstance(e, TradeEvent)]

    def test_new_trade_passes(self):
        adapter = _adapter()
        out = self._handle(adapter, _trades_msg([_trade(tid=1)]))
        assert len(out) == 1
        assert out[0].trade_id == "1"

    def test_duplicate_tid_dropped_second_call(self):
        """Same tid delivered on reconnect → emitted only on first call."""
        adapter = _adapter()
        out1 = self._handle(adapter, _trades_msg([_trade(tid=42)]))
        assert len(out1) == 1, "first delivery must pass"
        out2 = self._handle(adapter, _trades_msg([_trade(tid=42)]))
        assert out2 == [], "second delivery of same tid must be dropped"

    def test_duplicate_in_same_batch_dropped(self):
        """HL batches multiple trades in one message; duplicates within the batch must deduplicate."""
        adapter = _adapter()
        # tid=7 appears twice in the same WS message (shouldn't happen normally but be safe)
        out = self._handle(adapter, _trades_msg([_trade(tid=7), _trade(tid=7)]))
        assert len(out) == 1, "within-batch dup must be dropped"

    def test_different_tids_all_pass(self):
        adapter = _adapter()
        out = self._handle(adapter, _trades_msg([_trade(tid=1), _trade(tid=2), _trade(tid=3)]))
        assert len(out) == 3

    def test_seen_set_bound_enforced(self):
        """After the cap is reached, the oldest tid should be evicted so it can be re-accepted."""
        adapter = _adapter()
        sym_to_sub = {"BTC": PERP_SUB}

        # Insert enough tids to overflow the cap
        cap = adapter._trade_dedup_cap
        assert cap > 0, "_trade_dedup_cap must be a positive integer"

        # Fill the seen-set to exactly cap entries
        for i in range(cap):
            adapter._handle(_trades_msg([_trade(tid=i)]), _recv_ns(), sym_to_sub)

        # Verify tid=0 is in the seen-set (not yet evicted)
        out_before = [
            e
            for e in adapter._handle(_trades_msg([_trade(tid=0)]), _recv_ns(), sym_to_sub)
            if isinstance(e, TradeEvent)
        ]
        assert out_before == [], "tid=0 must still be in seen-set (not evicted yet)"

        # Push one more entry → tid=0 must be evicted (FIFO / LRU)
        adapter._handle(_trades_msg([_trade(tid=cap)]), _recv_ns(), sym_to_sub)

        # Now tid=0 should be evicted and re-accepted
        out_after = [
            e
            for e in adapter._handle(_trades_msg([_trade(tid=0)]), _recv_ns(), sym_to_sub)
            if isinstance(e, TradeEvent)
        ]
        assert len(out_after) == 1, "evicted tid must be re-accepted after cap overflow"

    def test_bound_size_reasonable(self):
        """Cap should be big enough to cover a reconnect burst but not unbounded."""
        adapter = _adapter()
        cap = adapter._trade_dedup_cap
        # Must be at least a few hundred (realistic reconnect burst) but not absurdly large
        assert 100 <= cap <= 100_000, f"cap={cap} is unreasonable"
