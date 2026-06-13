"""TDD for the stateful PmBook class (SHR-62).

price_change deltas must be MERGED into the full book before emission —
MarketState.apply treats BookSnapshotEvent as a full replace, so emitting
only the changed levels would corrupt the book to 1-2 phantom levels.
"""

from hlanalysis.adapters.polymarket_normalize import PmBook  # new


def test_price_change_delta_merges_into_full_book():
    bk = PmBook()
    # Full book snapshot: two bid levels, two ask levels.
    snap = bk.apply_book(
        {
            "asset_id": "A",
            "timestamp": 1,
            "bids": [{"price": "0.40", "size": "100"}, {"price": "0.39", "size": "50"}],
            "asks": [{"price": "0.60", "size": "80"}, {"price": "0.61", "size": "40"}],
        }
    )
    assert snap.bid_px[0] == 0.40 and len(snap.bid_px) == 2

    # price_change touches only the 0.39 level (resize) — must NOT wipe 0.40/asks.
    out = bk.apply_price_change(
        {
            "asset_id": "A",
            "timestamp": 2,
            "changes": [{"price": "0.39", "size": "10", "side": "BUY"}],
        }
    )
    assert set(out.bid_px) == {0.40, 0.39}  # full book preserved
    assert dict(zip(out.bid_px, out.bid_sz))[0.39] == 10.0  # delta applied
    assert len(out.ask_px) == 2  # asks untouched


def test_price_change_zero_size_removes_level():
    bk = PmBook()
    bk.apply_book(
        {
            "asset_id": "A",
            "timestamp": 1,
            "bids": [{"price": "0.40", "size": "100"}, {"price": "0.39", "size": "50"}],
            "asks": [],
        }
    )
    out = bk.apply_price_change(
        {"asset_id": "A", "timestamp": 2, "changes": [{"price": "0.39", "size": "0", "side": "BUY"}]}
    )
    assert set(out.bid_px) == {0.40}  # zero-size removed the level


def test_apply_book_replaces_state_completely():
    """A second apply_book should fully replace — no stale levels from first."""
    bk = PmBook()
    bk.apply_book(
        {
            "asset_id": "A",
            "timestamp": 1,
            "bids": [{"price": "0.70", "size": "50"}],
            "asks": [{"price": "0.80", "size": "30"}],
        }
    )
    snap2 = bk.apply_book({"asset_id": "A", "timestamp": 2, "bids": [{"price": "0.60", "size": "20"}], "asks": []})
    assert snap2.bid_px == [0.60]
    assert snap2.ask_px == []


def test_apply_price_change_returns_none_when_no_changes():
    bk = PmBook()
    bk.apply_book({"asset_id": "A", "timestamp": 1, "bids": [{"price": "0.50", "size": "10"}], "asks": []})
    out = bk.apply_price_change({"asset_id": "A", "timestamp": 2, "changes": []})
    assert out is None


def test_price_change_ask_side_updates_correctly():
    """SELL-side changes must update the asks dict, not the bids."""
    bk = PmBook()
    bk.apply_book(
        {
            "asset_id": "A",
            "timestamp": 1,
            "bids": [{"price": "0.45", "size": "100"}],
            "asks": [{"price": "0.55", "size": "80"}, {"price": "0.60", "size": "40"}],
        }
    )
    out = bk.apply_price_change(
        {"asset_id": "A", "timestamp": 2, "changes": [{"price": "0.55", "size": "50", "side": "SELL"}]}
    )
    assert set(out.ask_px) == {0.55, 0.60}
    assert dict(zip(out.ask_px, out.ask_sz))[0.55] == 50.0
    # bids unchanged
    assert out.bid_px == [0.45]


def test_emit_ordering_bids_high_to_low_asks_low_to_high():
    """Emitted snapshot must always be best-first regardless of insertion order."""
    bk = PmBook()
    # Feed levels in reverse-best order.
    snap = bk.apply_book(
        {
            "asset_id": "A",
            "timestamp": 1,
            "bids": [{"price": "0.30", "size": "1"}, {"price": "0.50", "size": "2"}, {"price": "0.40", "size": "3"}],
            "asks": [{"price": "0.70", "size": "4"}, {"price": "0.55", "size": "5"}, {"price": "0.65", "size": "6"}],
        }
    )
    # bids: highest first
    assert snap.bid_px == [0.50, 0.40, 0.30]
    # asks: lowest first
    assert snap.ask_px == [0.55, 0.65, 0.70]
