"""Unit tests for hlanalysis/analysis/book.py.

Fixtures are small synthetic parquet files written to tmp_path, laid out in
the same hive-partition structure that the recorder uses:

    data/venue=*/product_type=*/mechanism=*/event=*/symbol=*/date=*/hour=*/*.parquet

All timestamps are nanoseconds since epoch.  We use values in the range
1_000_000_000_000_000_000–1_000_000_001_000_000_000 (around 2001-09-09) so
they're clearly synthetic.
"""
from __future__ import annotations

import math
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

# We patch helpers.DATA_ROOT so book.py picks up our tmp fixtures.
import hlanalysis.analysis.helpers as _helpers


# ---------------------------------------------------------------------------
# Fixtures helpers
# ---------------------------------------------------------------------------

BASE_NS = 1_000_000_000_000_000_000  # arbitrary epoch base


def _write_parquet(path: Path, table: pa.Table) -> None:
    """Write a PyArrow table to a parquet file, creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(path))


def _partition_path(
    root: Path,
    venue: str,
    product_type: str,
    event: str,
    symbol: str,
) -> Path:
    """Return the parquet directory for a given partition set."""
    return (
        root
        / f"venue={venue}"
        / f"product_type={product_type}"
        / "mechanism=clob"
        / f"event={event}"
        / f"symbol={symbol}"
        / "date=2001-09-09"
        / "hour=01"
    )


@pytest.fixture()
def data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Patch helpers.DATA_ROOT to a tmp directory and return it."""
    monkeypatch.setattr(_helpers, "DATA_ROOT", tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# BBO fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def bbo_root(data_root: Path) -> Path:
    """Write 3 BBO rows for hyperliquid/perp/BTC to tmp parquet."""
    rows = [
        # (local_recv_ts, exchange_ts, bid_px, bid_sz, ask_px, ask_sz)
        (BASE_NS + 100_000_000, BASE_NS + 99_000_000, 50_000.0, 1.0, 50_010.0, 1.0),
        (BASE_NS + 200_000_000, BASE_NS + 199_000_000, 50_100.0, 2.0, 50_110.0, 2.0),
        (BASE_NS + 300_000_000, BASE_NS + 299_000_000, 50_200.0, 3.0, 50_210.0, 3.0),
    ]
    table = pa.table(
        {
            "venue": ["hyperliquid"] * 3,
            "product_type": ["perp"] * 3,
            "mechanism": ["clob"] * 3,
            "symbol": ["BTC"] * 3,
            "event_type": ["bbo"] * 3,
            "exchange_ts": [r[1] for r in rows],
            "local_recv_ts": [r[0] for r in rows],
            "seq": [None, None, None],
            "bid_px": [r[2] for r in rows],
            "bid_sz": [r[3] for r in rows],
            "ask_px": [r[4] for r in rows],
            "ask_sz": [r[5] for r in rows],
        }
    )
    part = _partition_path(data_root, "hyperliquid", "perp", "bbo", "BTC")
    _write_parquet(part / "part0.parquet", table)
    return data_root


# ---------------------------------------------------------------------------
# Book-snapshot fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def book_root(data_root: Path) -> Path:
    """Write 1 book_snapshot row for hyperliquid/perp/BTC."""
    # 5 bid levels, 5 ask levels
    bid_pxs = [50_000.0, 49_990.0, 49_980.0, 49_970.0, 49_960.0]
    bid_szs = [1.0, 2.0, 3.0, 4.0, 5.0]
    ask_pxs = [50_010.0, 50_020.0, 50_030.0, 50_040.0, 50_050.0]
    ask_szs = [1.0, 2.0, 3.0, 4.0, 5.0]

    table = pa.table(
        {
            "venue": ["hyperliquid"],
            "product_type": ["perp"],
            "mechanism": ["clob"],
            "symbol": ["BTC"],
            "event_type": ["book_snapshot"],
            "exchange_ts": [BASE_NS + 99_000_000],
            "local_recv_ts": [BASE_NS + 100_000_000],
            "seq": [None],
            "bid_px": pa.array([bid_pxs], type=pa.list_(pa.float64())),
            "bid_sz": pa.array([bid_szs], type=pa.list_(pa.float64())),
            "ask_px": pa.array([ask_pxs], type=pa.list_(pa.float64())),
            "ask_sz": pa.array([ask_szs], type=pa.list_(pa.float64())),
        }
    )
    part = _partition_path(data_root, "hyperliquid", "perp", "book_snapshot", "BTC")
    _write_parquet(part / "part0.parquet", table)
    return data_root


# ---------------------------------------------------------------------------
# Trade fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def trade_root(data_root: Path) -> Path:
    """Write 4 trade rows (mix of buy/sell/unknown) for hyperliquid/perp/BTC."""
    rows = [
        # (local_recv_ts, exchange_ts, price, size, side)
        (BASE_NS + 150_000_000, BASE_NS + 149_000_000, 50_005.0, 0.1, "buy"),
        (BASE_NS + 160_000_000, BASE_NS + 159_000_000, 50_004.0, 0.2, "sell"),
        (BASE_NS + 170_000_000, BASE_NS + 169_000_000, 50_003.0, 0.3, "unknown"),
        (BASE_NS + 250_000_000, BASE_NS + 249_000_000, 50_105.0, 0.4, "buy"),
    ]
    table = pa.table(
        {
            "venue": ["hyperliquid"] * 4,
            "product_type": ["perp"] * 4,
            "mechanism": ["clob"] * 4,
            "symbol": ["BTC"] * 4,
            "event_type": ["trade"] * 4,
            "exchange_ts": [r[1] for r in rows],
            "local_recv_ts": [r[0] for r in rows],
            "seq": [None, None, None, None],
            "price": [r[2] for r in rows],
            "size": [r[3] for r in rows],
            "side": [r[4] for r in rows],
            "trade_id": [None, None, None, None],
            "block_ts": [None, None, None, None],
            "buyer": [None, None, None, None],
            "seller": [None, None, None, None],
            "block_hash": [None, None, None, None],
        }
    )
    part = _partition_path(data_root, "hyperliquid", "perp", "trade", "BTC")
    _write_parquet(part / "part0.parquet", table)
    return data_root


# ---------------------------------------------------------------------------
# DuckDB connection fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def con() -> duckdb.DuckDBPyConnection:
    c = duckdb.connect()
    c.execute("SET enable_progress_bar=false;")
    yield c
    c.close()


# ---------------------------------------------------------------------------
# Tests: best_quotes_at
# ---------------------------------------------------------------------------

from hlanalysis.analysis.book import (
    best_quotes_at,
    mid_path,
    trades_in_window,
    depth_at_offset_bps,
)


class TestBestQuotesAt:
    def test_returns_correct_tuple_at_known_ts(self, con, bbo_root):
        """At ts_ns exactly equal to the second row's local_recv_ts, the
        second row's bid/ask should be returned."""
        ts = BASE_NS + 200_000_000
        bid, ask = best_quotes_at(
            con, venue="hyperliquid", product_type="perp", symbol="BTC", ts_ns=ts
        )
        assert bid == pytest.approx(50_100.0)
        assert ask == pytest.approx(50_110.0)

    def test_returns_most_recent_before_ts(self, con, bbo_root):
        """At ts_ns between row 2 and row 3, the second row should be
        returned (last seen)."""
        ts = BASE_NS + 250_000_000  # between 200ms and 300ms marks
        bid, ask = best_quotes_at(
            con, venue="hyperliquid", product_type="perp", symbol="BTC", ts_ns=ts
        )
        assert bid == pytest.approx(50_100.0)
        assert ask == pytest.approx(50_110.0)

    def test_returns_first_row_exactly(self, con, bbo_root):
        """Requesting ts_ns == first row's local_recv_ts returns first row."""
        ts = BASE_NS + 100_000_000
        bid, ask = best_quotes_at(
            con, venue="hyperliquid", product_type="perp", symbol="BTC", ts_ns=ts
        )
        assert bid == pytest.approx(50_000.0)
        assert ask == pytest.approx(50_010.0)

    def test_raises_when_no_bbo_before_ts(self, con, bbo_root):
        """ts_ns before any recorded BBO should raise ValueError."""
        ts = BASE_NS + 50_000_000  # before all rows (first is at +100ms)
        with pytest.raises(ValueError, match="No BBO row found"):
            best_quotes_at(
                con, venue="hyperliquid", product_type="perp", symbol="BTC", ts_ns=ts
            )


# ---------------------------------------------------------------------------
# Tests: mid_path
# ---------------------------------------------------------------------------

class TestMidPath:
    def test_shape_and_resample(self, con, bbo_root):
        """resample_ms=100 over a 300ms window should produce 4 grid points
        (start, start+100ms, start+200ms, start+300ms)."""
        start = BASE_NS
        end = BASE_NS + 300_000_000
        df = mid_path(
            con,
            venue="hyperliquid",
            product_type="perp",
            symbol="BTC",
            start_ns=start,
            end_ns=end,
            resample_ms=100,
        )
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["ts_ns", "mid"]
        assert len(df) == 4  # start, +100ms, +200ms, +300ms

    def test_first_grid_point_before_any_bbo_is_null(self, con, bbo_root):
        """Grid point at start_ns=BASE_NS is before first BBO (at BASE_NS+100ms),
        so mid should be null."""
        start = BASE_NS
        end = BASE_NS + 300_000_000
        df = mid_path(
            con,
            venue="hyperliquid",
            product_type="perp",
            symbol="BTC",
            start_ns=start,
            end_ns=end,
            resample_ms=100,
        )
        # Row 0: ts_ns = BASE_NS, before first BBO.
        assert pd.isna(df.loc[0, "mid"])

    def test_mid_values_correct(self, con, bbo_root):
        """Rows 1–3 should carry the mid from the corresponding BBO rows."""
        start = BASE_NS
        end = BASE_NS + 300_000_000
        df = mid_path(
            con,
            venue="hyperliquid",
            product_type="perp",
            symbol="BTC",
            start_ns=start,
            end_ns=end,
            resample_ms=100,
        )
        # Row 1: ts_ns = BASE_NS + 100ms → BBO row 1: (50000+50010)/2 = 50005.
        assert df.loc[1, "mid"] == pytest.approx(50_005.0)
        # Row 2: ts_ns = BASE_NS + 200ms → BBO row 2: (50100+50110)/2 = 50105.
        assert df.loc[2, "mid"] == pytest.approx(50_105.0)
        # Row 3: ts_ns = BASE_NS + 300ms → BBO row 3: (50200+50210)/2 = 50205.
        assert df.loc[3, "mid"] == pytest.approx(50_205.0)

    def test_ts_ns_dtype(self, con, bbo_root):
        start = BASE_NS
        end = BASE_NS + 200_000_000
        df = mid_path(
            con,
            venue="hyperliquid",
            product_type="perp",
            symbol="BTC",
            start_ns=start,
            end_ns=end,
            resample_ms=100,
        )
        assert df["ts_ns"].dtype == "int64"

    def test_returns_empty_grid_when_no_data(self, con, data_root):
        """If no BBO data at all, all mids should be null."""
        start = BASE_NS
        end = BASE_NS + 200_000_000
        df = mid_path(
            con,
            venue="hyperliquid",
            product_type="perp",
            symbol="BTC",
            start_ns=start,
            end_ns=end,
            resample_ms=100,
        )
        assert len(df) == 3
        assert df["mid"].isna().all()


# ---------------------------------------------------------------------------
# Tests: trades_in_window
# ---------------------------------------------------------------------------

class TestTradesInWindow:
    def test_returns_all_aggressed_trades(self, con, trade_root):
        """only_aggressed=True (default) should exclude 'unknown' side rows."""
        start = BASE_NS + 100_000_000
        end = BASE_NS + 300_000_000
        df = trades_in_window(
            con,
            venue="hyperliquid",
            product_type="perp",
            symbol="BTC",
            start_ns=start,
            end_ns=end,
        )
        # Rows at +150ms (buy), +160ms (sell), +250ms (buy).  +170ms (unknown) excluded.
        assert len(df) == 3
        assert set(df["aggressor_side"].unique()) <= {"buy", "sell"}

    def test_returns_all_trades_when_only_aggressed_false(self, con, trade_root):
        """only_aggressed=False should include unknown-side rows."""
        start = BASE_NS + 100_000_000
        end = BASE_NS + 300_000_000
        df = trades_in_window(
            con,
            venue="hyperliquid",
            product_type="perp",
            symbol="BTC",
            start_ns=start,
            end_ns=end,
            only_aggressed=False,
        )
        assert len(df) == 4
        assert "unknown" in df["aggressor_side"].values

    def test_window_filtering(self, con, trade_root):
        """Only trades within [start_ns, end_ns] inclusive should appear."""
        # Only the first trade at +150ms should appear.
        start = BASE_NS + 140_000_000
        end = BASE_NS + 155_000_000
        df = trades_in_window(
            con,
            venue="hyperliquid",
            product_type="perp",
            symbol="BTC",
            start_ns=start,
            end_ns=end,
        )
        assert len(df) == 1
        assert df.iloc[0]["price"] == pytest.approx(50_005.0)
        assert df.iloc[0]["aggressor_side"] == "buy"

    def test_columns_present(self, con, trade_root):
        df = trades_in_window(
            con,
            venue="hyperliquid",
            product_type="perp",
            symbol="BTC",
            start_ns=BASE_NS,
            end_ns=BASE_NS + 300_000_000,
        )
        assert set(df.columns) >= {"ts_ns", "price", "size", "aggressor_side"}

    def test_dtypes(self, con, trade_root):
        df = trades_in_window(
            con,
            venue="hyperliquid",
            product_type="perp",
            symbol="BTC",
            start_ns=BASE_NS,
            end_ns=BASE_NS + 300_000_000,
        )
        assert df["ts_ns"].dtype == "int64"
        assert df["price"].dtype == "float64"
        assert df["size"].dtype == "float64"


# ---------------------------------------------------------------------------
# Tests: depth_at_offset_bps
# ---------------------------------------------------------------------------

class TestDepthAtOffsetBps:
    def test_bid_depth_within_offset(self, con, book_root):
        """At 2 bps offset from best bid (50000), levels within threshold
        50000 * (1 - 2/10000) = 50000 * 0.9998 = 49990 should be included.
        Levels at 50000 (1.0) and 49990 (2.0) qualify → total = 3.0."""
        ts = BASE_NS + 200_000_000  # well after the snapshot at +100ms
        result = depth_at_offset_bps(
            con,
            venue="hyperliquid",
            product_type="perp",
            symbol="BTC",
            ts_ns=ts,
            side="bid",
            offset_bps=2.0,
        )
        assert result == pytest.approx(3.0)

    def test_ask_depth_within_offset(self, con, book_root):
        """At 2 bps offset from best ask (50010), threshold =
        50010 * (1 + 2/10000) = 50010 * 1.0002 = 50020.002.
        Levels at 50010 (1.0) and 50020 (2.0) qualify → total = 3.0."""
        ts = BASE_NS + 200_000_000
        result = depth_at_offset_bps(
            con,
            venue="hyperliquid",
            product_type="perp",
            symbol="BTC",
            ts_ns=ts,
            side="ask",
            offset_bps=2.0,
        )
        assert result == pytest.approx(3.0)

    def test_bid_depth_zero_offset(self, con, book_root):
        """At 0 bps, only the best bid level (50000, size=1.0) qualifies."""
        ts = BASE_NS + 200_000_000
        result = depth_at_offset_bps(
            con,
            venue="hyperliquid",
            product_type="perp",
            symbol="BTC",
            ts_ns=ts,
            side="bid",
            offset_bps=0.0,
        )
        assert result == pytest.approx(1.0)

    def test_ask_depth_zero_offset(self, con, book_root):
        """At 0 bps, only best ask level (50010, size=1.0) qualifies."""
        ts = BASE_NS + 200_000_000
        result = depth_at_offset_bps(
            con,
            venue="hyperliquid",
            product_type="perp",
            symbol="BTC",
            ts_ns=ts,
            side="ask",
            offset_bps=0.0,
        )
        assert result == pytest.approx(1.0)

    def test_returns_zero_when_no_snapshot(self, con, data_root):
        """No snapshot data at all → 0.0 returned, no exception."""
        result = depth_at_offset_bps(
            con,
            venue="hyperliquid",
            product_type="perp",
            symbol="BTC",
            ts_ns=BASE_NS + 200_000_000,
            side="bid",
            offset_bps=5.0,
        )
        assert result == 0.0

    def test_returns_zero_when_ts_before_any_snapshot(self, con, book_root):
        """ts_ns before any snapshot row → 0.0 returned."""
        result = depth_at_offset_bps(
            con,
            venue="hyperliquid",
            product_type="perp",
            symbol="BTC",
            ts_ns=BASE_NS + 50_000_000,  # before the snapshot at +100ms
            side="bid",
            offset_bps=5.0,
        )
        assert result == 0.0

    def test_large_offset_sums_all_levels(self, con, book_root):
        """10000 bps (100%) offset should include all 5 bid levels.
        Sizes: 1+2+3+4+5 = 15."""
        ts = BASE_NS + 200_000_000
        result = depth_at_offset_bps(
            con,
            venue="hyperliquid",
            product_type="perp",
            symbol="BTC",
            ts_ns=ts,
            side="bid",
            offset_bps=10_000.0,
        )
        assert result == pytest.approx(15.0)
