"""Unit tests for hlanalysis/analysis/microstructure.py.

Fixtures are small synthetic parquet files written to tmp_path, laid out in
the hive-partition structure produced by the recorder:

    data/venue=*/product_type=*/mechanism=*/event=*/symbol=*/date=*/hour=*/*.parquet

All timestamps are nanoseconds since epoch.  BASE_NS is around 2001-09-09.
"""
from __future__ import annotations

import math
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import hlanalysis.analysis.helpers as _helpers
from hlanalysis.analysis.microstructure import (
    book_imbalance,
    cross_correlation,
    quoted_spread_bps,
    returns_resampled,
    tob_churn,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_NS = 1_000_000_000_000_000_000  # ~2001-09-09
NS_PER_S = 1_000_000_000
NS_PER_MS = 1_000_000

# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------


def _write_parquet(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(path))


def _partition_path(root: Path, venue: str, product_type: str, event: str, symbol: str) -> Path:
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
    monkeypatch.setattr(_helpers, "DATA_ROOT", tmp_path)
    return tmp_path


@pytest.fixture()
def con() -> duckdb.DuckDBPyConnection:
    c = duckdb.connect()
    c.execute("SET enable_progress_bar=false;")
    yield c
    c.close()


# ---------------------------------------------------------------------------
# Parquet writers
# ---------------------------------------------------------------------------


def _write_bbo(
    root: Path, venue: str, product_type: str, symbol: str, rows: list[tuple]
) -> None:
    """rows: list of (local_recv_ts, bid_px, ask_px)"""
    table = pa.table(
        {
            "venue": [venue] * len(rows),
            "product_type": [product_type] * len(rows),
            "mechanism": ["clob"] * len(rows),
            "symbol": [symbol] * len(rows),
            "event_type": ["bbo"] * len(rows),
            "exchange_ts": [r[0] for r in rows],
            "local_recv_ts": [r[0] for r in rows],
            "seq": pa.array([None] * len(rows), type=pa.int64()),
            "bid_px": [float(r[1]) for r in rows],
            "bid_sz": [1.0] * len(rows),
            "ask_px": [float(r[2]) for r in rows],
            "ask_sz": [1.0] * len(rows),
        }
    )
    part = _partition_path(root, venue, product_type, "bbo", symbol)
    _write_parquet(part / "part0.parquet", table)


def _write_book_snapshot(
    root: Path,
    venue: str,
    product_type: str,
    symbol: str,
    rows: list[tuple],
) -> None:
    """rows: list of (local_recv_ts, bid_pxs, bid_szs, ask_pxs, ask_szs)"""
    table = pa.table(
        {
            "venue": [venue] * len(rows),
            "product_type": [product_type] * len(rows),
            "mechanism": ["clob"] * len(rows),
            "symbol": [symbol] * len(rows),
            "event_type": ["book_snapshot"] * len(rows),
            "exchange_ts": [r[0] for r in rows],
            "local_recv_ts": [r[0] for r in rows],
            "seq": pa.array([None] * len(rows), type=pa.int64()),
            "bid_px": pa.array([list(r[1]) for r in rows], type=pa.list_(pa.float64())),
            "bid_sz": pa.array([list(r[2]) for r in rows], type=pa.list_(pa.float64())),
            "ask_px": pa.array([list(r[3]) for r in rows], type=pa.list_(pa.float64())),
            "ask_sz": pa.array([list(r[4]) for r in rows], type=pa.list_(pa.float64())),
        }
    )
    part = _partition_path(root, venue, product_type, "book_snapshot", symbol)
    _write_parquet(part / "part0.parquet", table)


# ===========================================================================
# Tests: quoted_spread_bps
# ===========================================================================


class TestQuotedSpreadBps:
    """AC: spread_bps = (ask - bid) / mid * 1e4; resampled at resample_s cadence."""

    def test_basic_spread_values(self, con, data_root):
        """Three BBO rows with known spreads; resample=1s should carry each forward."""
        # Row 0 at t=0s: bid=99, ask=101  → spread=2, mid=100 → 200 bps
        # Row 1 at t=1s: bid=100, ask=102 → spread=2, mid=101 → ~198.02 bps
        # Row 2 at t=2s: bid=200, ask=204 → spread=4, mid=202 → ~198.02 bps
        bbo_rows = [
            (BASE_NS + 0 * NS_PER_S, 99.0, 101.0),
            (BASE_NS + 1 * NS_PER_S, 100.0, 102.0),
            (BASE_NS + 2 * NS_PER_S, 200.0, 204.0),
        ]
        _write_bbo(data_root, "hyperliquid", "perp", "BTC", bbo_rows)

        df = quoted_spread_bps(
            con,
            venue="hyperliquid",
            product_type="perp",
            symbol="BTC",
            start_ns=BASE_NS,
            end_ns=BASE_NS + 2 * NS_PER_S,
            resample_s=1,
        )

        assert list(df.columns) == ["ts_ns", "spread_bps"]
        assert len(df) == 3  # grid: t=0,1,2

        # t=0: mid=100, spread=2 → 200 bps
        assert df.iloc[0]["spread_bps"] == pytest.approx(200.0, rel=1e-6)
        # t=1: mid=101, spread=2 → 2/101 * 1e4 ≈ 198.0198
        assert df.iloc[1]["spread_bps"] == pytest.approx(2.0 / 101.0 * 1e4, rel=1e-6)
        # t=2: mid=202, spread=4 → 4/202 * 1e4 ≈ 198.0198
        assert df.iloc[2]["spread_bps"] == pytest.approx(4.0 / 202.0 * 1e4, rel=1e-6)

    def test_locf_carries_last_value(self, con, data_root):
        """Single BBO at t=0s; resample=1s over 3s should carry that value forward."""
        bbo_rows = [(BASE_NS + 0 * NS_PER_S, 100.0, 100.10)]  # 10 bps spread
        _write_bbo(data_root, "hyperliquid", "perp", "BTC", bbo_rows)

        df = quoted_spread_bps(
            con,
            venue="hyperliquid",
            product_type="perp",
            symbol="BTC",
            start_ns=BASE_NS,
            end_ns=BASE_NS + 3 * NS_PER_S,
            resample_s=1,
        )

        assert len(df) == 4  # 0,1,2,3
        expected_bps = 0.10 / 100.05 * 1e4
        for i in range(4):
            assert df.iloc[i]["spread_bps"] == pytest.approx(expected_bps, rel=1e-5)

    def test_nan_before_first_bbo(self, con, data_root):
        """Grid points before the first BBO row should be NaN."""
        # BBO only at t=2s; grid starts at t=0s
        bbo_rows = [(BASE_NS + 2 * NS_PER_S, 100.0, 100.10)]
        _write_bbo(data_root, "hyperliquid", "perp", "BTC", bbo_rows)

        df = quoted_spread_bps(
            con,
            venue="hyperliquid",
            product_type="perp",
            symbol="BTC",
            start_ns=BASE_NS,
            end_ns=BASE_NS + 3 * NS_PER_S,
            resample_s=1,
        )

        assert pd.isna(df.iloc[0]["spread_bps"])  # t=0: no BBO yet
        assert pd.isna(df.iloc[1]["spread_bps"])  # t=1: no BBO yet
        assert not pd.isna(df.iloc[2]["spread_bps"])  # t=2: BBO present

    def test_no_data_returns_nan_grid(self, con, data_root):
        """No parquet files at all → all NaN spread."""
        df = quoted_spread_bps(
            con,
            venue="hyperliquid",
            product_type="perp",
            symbol="BTC",
            start_ns=BASE_NS,
            end_ns=BASE_NS + 2 * NS_PER_S,
            resample_s=1,
        )
        assert len(df) == 3
        assert df["spread_bps"].isna().all()

    def test_output_dtypes(self, con, data_root):
        bbo_rows = [(BASE_NS, 100.0, 100.10)]
        _write_bbo(data_root, "hyperliquid", "perp", "BTC", bbo_rows)

        df = quoted_spread_bps(
            con,
            venue="hyperliquid",
            product_type="perp",
            symbol="BTC",
            start_ns=BASE_NS,
            end_ns=BASE_NS + 1 * NS_PER_S,
            resample_s=1,
        )
        assert df["ts_ns"].dtype == "int64"
        assert df["spread_bps"].dtype == "float64"


# ===========================================================================
# Tests: book_imbalance
# ===========================================================================


class TestBookImbalance:
    """AC: imb = (bid_sum - ask_sum) / (bid_sum + ask_sum); range [-1, +1]."""

    def test_all_bid_pressure_returns_plus_one(self, con, data_root):
        """bid_sz=[10], ask_sz=[0] → imbalance = +1."""
        _write_book_snapshot(
            data_root,
            "hyperliquid",
            "perp",
            "BTC",
            [(BASE_NS + NS_PER_S, [50000.0], [10.0], [50010.0], [0.0])],
        )
        df = book_imbalance(
            con,
            venue="hyperliquid",
            product_type="perp",
            symbol="BTC",
            start_ns=BASE_NS,
            end_ns=BASE_NS + 2 * NS_PER_S,
            levels=1,
        )
        assert len(df) == 1
        assert df.iloc[0]["imbalance"] == pytest.approx(1.0)

    def test_all_ask_pressure_returns_minus_one(self, con, data_root):
        """bid_sz=[0], ask_sz=[10] → imbalance = -1."""
        _write_book_snapshot(
            data_root,
            "hyperliquid",
            "perp",
            "BTC",
            [(BASE_NS + NS_PER_S, [50000.0], [0.0], [50010.0], [10.0])],
        )
        df = book_imbalance(
            con,
            venue="hyperliquid",
            product_type="perp",
            symbol="BTC",
            start_ns=BASE_NS,
            end_ns=BASE_NS + 2 * NS_PER_S,
            levels=1,
        )
        assert len(df) == 1
        assert df.iloc[0]["imbalance"] == pytest.approx(-1.0)

    def test_equal_pressure_returns_zero(self, con, data_root):
        """bid_sz=[5], ask_sz=[5] → imbalance = 0."""
        _write_book_snapshot(
            data_root,
            "hyperliquid",
            "perp",
            "BTC",
            [(BASE_NS + NS_PER_S, [50000.0], [5.0], [50010.0], [5.0])],
        )
        df = book_imbalance(
            con,
            venue="hyperliquid",
            product_type="perp",
            symbol="BTC",
            start_ns=BASE_NS,
            end_ns=BASE_NS + 2 * NS_PER_S,
            levels=1,
        )
        assert len(df) == 1
        assert df.iloc[0]["imbalance"] == pytest.approx(0.0)

    def test_levels_parameter_limits_depth(self, con, data_root):
        """With levels=1, only top level counts; deeper levels ignored."""
        # 3 levels: bid=[10, 5, 2], ask=[3, 7, 8]
        # levels=1: bid_sum=10, ask_sum=3 → (10-3)/(10+3) = 7/13
        _write_book_snapshot(
            data_root,
            "hyperliquid",
            "perp",
            "BTC",
            [
                (
                    BASE_NS + NS_PER_S,
                    [50000.0, 49990.0, 49980.0],
                    [10.0, 5.0, 2.0],
                    [50010.0, 50020.0, 50030.0],
                    [3.0, 7.0, 8.0],
                )
            ],
        )
        df = book_imbalance(
            con,
            venue="hyperliquid",
            product_type="perp",
            symbol="BTC",
            start_ns=BASE_NS,
            end_ns=BASE_NS + 2 * NS_PER_S,
            levels=1,
        )
        assert df.iloc[0]["imbalance"] == pytest.approx(7.0 / 13.0, rel=1e-6)

    def test_levels_two(self, con, data_root):
        """levels=2 sums first two levels on each side."""
        # bid=[10, 5, 2], ask=[3, 7, 8]
        # levels=2: bid_sum=15, ask_sum=10 → (15-10)/(15+10) = 5/25 = 0.2
        _write_book_snapshot(
            data_root,
            "hyperliquid",
            "perp",
            "BTC",
            [
                (
                    BASE_NS + NS_PER_S,
                    [50000.0, 49990.0, 49980.0],
                    [10.0, 5.0, 2.0],
                    [50010.0, 50020.0, 50030.0],
                    [3.0, 7.0, 8.0],
                )
            ],
        )
        df = book_imbalance(
            con,
            venue="hyperliquid",
            product_type="perp",
            symbol="BTC",
            start_ns=BASE_NS,
            end_ns=BASE_NS + 2 * NS_PER_S,
            levels=2,
        )
        assert df.iloc[0]["imbalance"] == pytest.approx(0.2, rel=1e-6)

    def test_multiple_snapshots(self, con, data_root):
        """One row per snapshot; no resampling."""
        _write_book_snapshot(
            data_root,
            "hyperliquid",
            "perp",
            "BTC",
            [
                (BASE_NS + 1 * NS_PER_S, [50000.0], [8.0], [50010.0], [2.0]),  # 0.6
                (BASE_NS + 2 * NS_PER_S, [50000.0], [2.0], [50010.0], [8.0]),  # -0.6
            ],
        )
        df = book_imbalance(
            con,
            venue="hyperliquid",
            product_type="perp",
            symbol="BTC",
            start_ns=BASE_NS,
            end_ns=BASE_NS + 3 * NS_PER_S,
            levels=1,
        )
        assert len(df) == 2
        assert df.iloc[0]["imbalance"] == pytest.approx(0.6)
        assert df.iloc[1]["imbalance"] == pytest.approx(-0.6)

    def test_zero_total_size_returns_nan(self, con, data_root):
        """bid_sz=[0] and ask_sz=[0] → NaN (divide-by-zero guard)."""
        _write_book_snapshot(
            data_root,
            "hyperliquid",
            "perp",
            "BTC",
            [(BASE_NS + NS_PER_S, [50000.0], [0.0], [50010.0], [0.0])],
        )
        df = book_imbalance(
            con,
            venue="hyperliquid",
            product_type="perp",
            symbol="BTC",
            start_ns=BASE_NS,
            end_ns=BASE_NS + 2 * NS_PER_S,
            levels=1,
        )
        assert len(df) == 1
        assert pd.isna(df.iloc[0]["imbalance"])

    def test_no_data_returns_empty_df(self, con, data_root):
        df = book_imbalance(
            con,
            venue="hyperliquid",
            product_type="perp",
            symbol="BTC",
            start_ns=BASE_NS,
            end_ns=BASE_NS + 5 * NS_PER_S,
            levels=1,
        )
        assert len(df) == 0
        assert list(df.columns) == ["ts_ns", "imbalance"]

    def test_output_dtypes(self, con, data_root):
        _write_book_snapshot(
            data_root,
            "hyperliquid",
            "perp",
            "BTC",
            [(BASE_NS + NS_PER_S, [50000.0], [5.0], [50010.0], [5.0])],
        )
        df = book_imbalance(
            con,
            venue="hyperliquid",
            product_type="perp",
            symbol="BTC",
            start_ns=BASE_NS,
            end_ns=BASE_NS + 2 * NS_PER_S,
            levels=1,
        )
        assert df["ts_ns"].dtype == "int64"
        assert df["imbalance"].dtype == "float64"


# ===========================================================================
# Tests: tob_churn
# ===========================================================================


class TestTobChurn:
    """AC: counts BBO updates per bin_s window."""

    def test_basic_counts(self, con, data_root):
        """6 BBO rows: 4 in bin 0, 2 in bin 1; bin_s=10."""
        rows = [
            (BASE_NS + 0 * NS_PER_S, 100.0, 100.1),
            (BASE_NS + 3 * NS_PER_S, 100.0, 100.1),
            (BASE_NS + 7 * NS_PER_S, 100.0, 100.1),
            (BASE_NS + 9 * NS_PER_S, 100.0, 100.1),  # all in first 10s bin
            (BASE_NS + 11 * NS_PER_S, 100.0, 100.1),
            (BASE_NS + 15 * NS_PER_S, 100.0, 100.1),  # both in second 10s bin
        ]
        _write_bbo(data_root, "hyperliquid", "perp", "BTC", rows)

        df = tob_churn(
            con,
            venue="hyperliquid",
            product_type="perp",
            symbol="BTC",
            start_ns=BASE_NS,
            end_ns=BASE_NS + 20 * NS_PER_S,
            bin_s=10,
        )

        assert list(df.columns) == ["ts_ns", "n_updates"]
        # Bins: [0,10s), [10s,20s) → ts_ns = BASE_NS + 0, BASE_NS + 10s
        assert len(df) == 2
        assert int(df.iloc[0]["n_updates"]) == 4
        assert int(df.iloc[1]["n_updates"]) == 2

    def test_empty_bins_have_zero_count(self, con, data_root):
        """Bin with no BBO updates still appears with n_updates=0."""
        # Only one row at t=0; bin at t=10s should be empty.
        rows = [(BASE_NS + 0 * NS_PER_S, 100.0, 100.1)]
        _write_bbo(data_root, "hyperliquid", "perp", "BTC", rows)

        df = tob_churn(
            con,
            venue="hyperliquid",
            product_type="perp",
            symbol="BTC",
            start_ns=BASE_NS,
            end_ns=BASE_NS + 20 * NS_PER_S,
            bin_s=10,
        )

        assert len(df) == 2
        assert int(df.iloc[0]["n_updates"]) == 1
        assert int(df.iloc[1]["n_updates"]) == 0

    def test_no_data_returns_zero_grid(self, con, data_root):
        """No parquet files at all → all zero counts."""
        df = tob_churn(
            con,
            venue="hyperliquid",
            product_type="perp",
            symbol="BTC",
            start_ns=BASE_NS,
            end_ns=BASE_NS + 60 * NS_PER_S,
            bin_s=60,
        )
        assert len(df) == 1
        assert int(df.iloc[0]["n_updates"]) == 0

    def test_ts_ns_is_bin_start(self, con, data_root):
        """ts_ns in output must equal the start of each bin."""
        rows = [(BASE_NS + 5 * NS_PER_S, 100.0, 100.1)]
        _write_bbo(data_root, "hyperliquid", "perp", "BTC", rows)

        df = tob_churn(
            con,
            venue="hyperliquid",
            product_type="perp",
            symbol="BTC",
            start_ns=BASE_NS,
            end_ns=BASE_NS + 20 * NS_PER_S,
            bin_s=10,
        )

        assert int(df.iloc[0]["ts_ns"]) == BASE_NS
        assert int(df.iloc[1]["ts_ns"]) == BASE_NS + 10 * NS_PER_S

    def test_output_dtypes(self, con, data_root):
        rows = [(BASE_NS, 100.0, 100.1)]
        _write_bbo(data_root, "hyperliquid", "perp", "BTC", rows)

        df = tob_churn(
            con,
            venue="hyperliquid",
            product_type="perp",
            symbol="BTC",
            start_ns=BASE_NS,
            end_ns=BASE_NS + 60 * NS_PER_S,
            bin_s=60,
        )
        assert df["ts_ns"].dtype == "int64"
        assert df["n_updates"].dtype == "int64"

    def test_event_at_boundary_belongs_to_lower_bin(self, con, data_root):
        """Event exactly at t=10s goes into second bin [10,20s)."""
        rows = [
            (BASE_NS + 10 * NS_PER_S, 100.0, 100.1),
        ]
        _write_bbo(data_root, "hyperliquid", "perp", "BTC", rows)

        df = tob_churn(
            con,
            venue="hyperliquid",
            product_type="perp",
            symbol="BTC",
            start_ns=BASE_NS,
            end_ns=BASE_NS + 20 * NS_PER_S,
            bin_s=10,
        )

        assert int(df.iloc[0]["n_updates"]) == 0   # [0,10s) — empty
        assert int(df.iloc[1]["n_updates"]) == 1   # [10,20s) — one event


# ===========================================================================
# Tests: returns_resampled
# ===========================================================================


class TestReturnsResampled:
    """AC: log_return = log(mid[i]/mid[i-1]); first row NaN."""

    def test_log_return_formula(self, con, data_root):
        """Linear mid path: mid rises by 1 per second from 100.
        log_return[1] = log(101/100), log_return[2] = log(102/101), etc.
        """
        # BBO rows at integer seconds 0..4, mid = 100 + i
        bbo_rows = []
        for i in range(5):
            mid = 100.0 + i
            bbo_rows.append((BASE_NS + i * NS_PER_S, mid - 0.5, mid + 0.5))
        _write_bbo(data_root, "hyperliquid", "perp", "BTC", bbo_rows)

        df = returns_resampled(
            con,
            venue="hyperliquid",
            product_type="perp",
            symbol="BTC",
            start_ns=BASE_NS,
            end_ns=BASE_NS + 4 * NS_PER_S,
            dt_ms=1000,  # 1s grid
        )

        assert list(df.columns) == ["ts_ns", "log_return"]
        assert len(df) == 5

        # First row must be NaN.
        assert pd.isna(df.iloc[0]["log_return"])

        # Rows 1–4: log(mid[i] / mid[i-1])
        for i in range(1, 5):
            expected = math.log((100.0 + i) / (100.0 + i - 1))
            assert df.iloc[i]["log_return"] == pytest.approx(expected, rel=1e-8)

    def test_first_row_always_nan(self, con, data_root):
        """Even if BBO exists at t=0, the first log_return must be NaN."""
        bbo_rows = [(BASE_NS + i * NS_PER_S, 99.5 + i, 100.5 + i) for i in range(3)]
        _write_bbo(data_root, "hyperliquid", "perp", "BTC", bbo_rows)

        df = returns_resampled(
            con,
            venue="hyperliquid",
            product_type="perp",
            symbol="BTC",
            start_ns=BASE_NS,
            end_ns=BASE_NS + 2 * NS_PER_S,
            dt_ms=1000,
        )

        assert pd.isna(df.iloc[0]["log_return"])

    def test_nan_mid_propagates(self, con, data_root):
        """Grid point with no prior BBO has NaN mid → log_return NaN there and next."""
        # BBO only at t=2s onwards.
        bbo_rows = [
            (BASE_NS + 2 * NS_PER_S, 99.5, 100.5),
            (BASE_NS + 3 * NS_PER_S, 100.5, 101.5),
        ]
        _write_bbo(data_root, "hyperliquid", "perp", "BTC", bbo_rows)

        df = returns_resampled(
            con,
            venue="hyperliquid",
            product_type="perp",
            symbol="BTC",
            start_ns=BASE_NS,
            end_ns=BASE_NS + 3 * NS_PER_S,
            dt_ms=1000,
        )

        # Rows 0 and 1 have NaN mid; row 2 has mid=100; row 3 has mid=101.
        assert pd.isna(df.iloc[0]["log_return"])  # always NaN
        assert pd.isna(df.iloc[1]["log_return"])  # prev is NaN
        assert pd.isna(df.iloc[2]["log_return"])  # prev is NaN (row 1)
        assert not pd.isna(df.iloc[3]["log_return"])  # log(101/100) valid

    def test_output_dtypes(self, con, data_root):
        bbo_rows = [(BASE_NS + i * NS_PER_S, 99.5 + i, 100.5 + i) for i in range(3)]
        _write_bbo(data_root, "hyperliquid", "perp", "BTC", bbo_rows)

        df = returns_resampled(
            con,
            venue="hyperliquid",
            product_type="perp",
            symbol="BTC",
            start_ns=BASE_NS,
            end_ns=BASE_NS + 2 * NS_PER_S,
            dt_ms=1000,
        )

        assert df["ts_ns"].dtype == "int64"
        assert df["log_return"].dtype == "float64"


# ===========================================================================
# Tests: cross_correlation  (AC2)
# ===========================================================================


class TestCrossCorrelation:
    """AC2: AR(1) with known autocorrelation lag."""

    def test_ar1_known_lag(self):
        """y[t] = 0.7 * y[t-1] + noise → CCF at lag=1 ≈ 0.7 (tolerance ±0.05)."""
        rng = np.random.default_rng(42)
        n = 2000
        phi = 0.7
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = phi * y[t - 1] + rng.normal(0, 1)

        # Autocorrelation: CCF of y against itself at lag=1 should ≈ phi.
        x = pd.Series(y)
        ys = pd.Series(y)

        result = cross_correlation(x, ys, max_lag=5)

        # Retrieve lag=1 value.
        ccf_lag1 = float(result.loc[result["lag"] == 1, "ccf"].iloc[0])
        assert abs(ccf_lag1 - phi) < 0.05, (
            f"CCF at lag=1 ({ccf_lag1:.4f}) not within ±0.05 of phi={phi}"
        )

    def test_output_columns_and_shape(self):
        """max_lag=3 → 7 rows with lags [-3..3]."""
        x = pd.Series(np.arange(100, dtype=float))
        y = pd.Series(np.arange(100, dtype=float))

        result = cross_correlation(x, y, max_lag=3)

        assert list(result.columns) == ["lag", "ccf"]
        assert len(result) == 7  # -3,-2,-1,0,1,2,3
        assert list(result["lag"]) == [-3, -2, -1, 0, 1, 2, 3]

    def test_lag_zero_equals_pearson(self):
        """At lag=0, CCF should equal the ordinary Pearson correlation."""
        rng = np.random.default_rng(7)
        x = pd.Series(rng.normal(0, 1, 200))
        y = pd.Series(rng.normal(0, 1, 200))

        result = cross_correlation(x, y, max_lag=5)
        ccf0 = float(result.loc[result["lag"] == 0, "ccf"].iloc[0])
        expected = float(x.corr(y))
        assert ccf0 == pytest.approx(expected, abs=1e-10)

    def test_perfectly_correlated_at_lag_zero(self):
        """x == y → CCF at lag=0 == 1.0."""
        x = pd.Series(np.linspace(1, 10, 100))
        y = pd.Series(np.linspace(1, 10, 100))

        result = cross_correlation(x, y, max_lag=2)
        ccf0 = float(result.loc[result["lag"] == 0, "ccf"].iloc[0])
        assert ccf0 == pytest.approx(1.0, abs=1e-9)

    def test_raises_when_max_lag_too_large(self):
        """max_lag >= len(x) should raise ValueError."""
        x = pd.Series([1.0, 2.0, 3.0])
        y = pd.Series([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="max_lag"):
            cross_correlation(x, y, max_lag=3)

    def test_raises_when_max_lag_equals_len(self):
        """max_lag == len(x) (== 5) should also raise."""
        x = pd.Series(range(5), dtype=float)
        y = pd.Series(range(5), dtype=float)

        with pytest.raises(ValueError):
            cross_correlation(x, y, max_lag=5)

    def test_output_dtypes(self):
        x = pd.Series(np.arange(50, dtype=float))
        y = pd.Series(np.arange(50, dtype=float))

        result = cross_correlation(x, y, max_lag=2)
        assert result["lag"].dtype == "int64"
        assert result["ccf"].dtype == "float64"

    def test_lead_lag_sign_convention(self):
        """Verify the lag-sign convention: lag>0 means y's past predicts x's present.

        If x[t] = y[t-2] (x leads by 2), CCF at lag=+2 should be ~1.
        """
        n = 200
        base = np.linspace(0, 10, n)
        y = pd.Series(base)
        # x is y shifted left by 2 (x[t] = y[t+2] in value terms).
        # Equivalently: y's past (at t-2) == x (at t).
        x = pd.Series(np.append(base[2:], [np.nan, np.nan]))

        result = cross_correlation(x, y, max_lag=5)
        ccf_pos2 = float(result.loc[result["lag"] == 2, "ccf"].iloc[0])
        # Should be close to 1 since y.shift(2) ≈ x (after dropping NaN rows).
        assert ccf_pos2 > 0.99
