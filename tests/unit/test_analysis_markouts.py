"""Unit tests for hlanalysis/analysis/markouts.py.

Fixtures are small synthetic parquet files written to tmp_path, laid out in
the same hive-partition structure that the recorder uses:

    data/venue=*/product_type=*/mechanism=*/event=*/symbol=*/date=*/hour=*/*.parquet

All timestamps are nanoseconds since epoch.  We use values in the range
1_000_000_000_000_000_000–... (around 2001-09-09) so they're clearly synthetic.
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
from hlanalysis.analysis.markouts import trade_markouts

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_NS = 1_000_000_000_000_000_000  # arbitrary epoch base
NS_PER_S = 1_000_000_000


# ---------------------------------------------------------------------------
# Fixture helpers
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
# Synthetic dataset builders
# ---------------------------------------------------------------------------


def _write_bbo(root: Path, venue: str, product_type: str, symbol: str, rows: list[tuple]) -> None:
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
            "seq": [None] * len(rows),
            "bid_px": [r[1] for r in rows],
            "bid_sz": [1.0] * len(rows),
            "ask_px": [r[2] for r in rows],
            "ask_sz": [1.0] * len(rows),
        }
    )
    part = _partition_path(root, venue, product_type, "bbo", symbol)
    _write_parquet(part / "part0.parquet", table)


def _write_trades(
    root: Path, venue: str, product_type: str, symbol: str, rows: list[tuple]
) -> None:
    """rows: list of (local_recv_ts, price, size, side)"""
    table = pa.table(
        {
            "venue": [venue] * len(rows),
            "product_type": [product_type] * len(rows),
            "mechanism": ["clob"] * len(rows),
            "symbol": [symbol] * len(rows),
            "event_type": ["trade"] * len(rows),
            "exchange_ts": [r[0] for r in rows],
            "local_recv_ts": [r[0] for r in rows],
            "seq": [None] * len(rows),
            "price": [float(r[1]) for r in rows],
            "size": [float(r[2]) for r in rows],
            "side": [r[3] for r in rows],
            "trade_id": [None] * len(rows),
            "block_ts": [None] * len(rows),
            "buyer": [None] * len(rows),
            "seller": [None] * len(rows),
            "block_hash": [None] * len(rows),
        }
    )
    part = _partition_path(root, venue, product_type, "trade", symbol)
    _write_parquet(part / "part0.parquet", table)


# ---------------------------------------------------------------------------
# AC1: Synthetic test — 10 trades on a known mid path, hand-computed markouts
# ---------------------------------------------------------------------------


class TestSyntheticMarkouts:
    """AC1: tolerance < 1e-9 on hand-computed values."""

    def test_ten_trades_known_mid_path(self, con, data_root):
        """Build a linear mid path and 10 trades; verify markouts to < 1e-9.

        BBO rows every 1 s from t=0 to t=20 s (21 rows), so for trades at
        t=1..10 s both the 1-s and 5-s horizon mids are always covered by an
        exact BBO row.  mid(t) = 100 + t (seconds).  All trades are buy-side.
        """
        # BBO rows at integer seconds 0..20, mid = 100 + i exactly.
        bbo_rows = []
        for i in range(21):
            t = BASE_NS + i * NS_PER_S
            mid = 100.0 + i
            bbo_rows.append((t, mid - 0.5, mid + 0.5))

        # 10 trades at integer seconds 1..10 (all buy side).
        trade_rows = []
        for i in range(1, 11):
            t = BASE_NS + i * NS_PER_S
            trade_rows.append((t, 100.0 + i, 0.1, "buy"))

        _write_bbo(data_root, "hyperliquid", "perp", "BTC", bbo_rows)
        _write_trades(data_root, "hyperliquid", "perp", "BTC", trade_rows)

        start_ns = BASE_NS
        end_ns = BASE_NS + 20 * NS_PER_S  # 20 s window; max trade+horizon = 10+5 = 15 s

        df = trade_markouts(
            con,
            venue="hyperliquid",
            product_type="perp",
            symbol="BTC",
            start_ns=start_ns,
            end_ns=end_ns,
            horizons_s=(1, 5),
        )

        assert len(df) == 10

        for idx, row in df.iterrows():
            t_s = (row["ts_ns"] - BASE_NS) / NS_PER_S  # integer seconds 1..10

            # mid at trade: linear path = 100 + t_s
            expected_mid_at_trade = 100.0 + t_s
            assert row["mid_at_trade"] == pytest.approx(expected_mid_at_trade, abs=1e-9)

            # horizon 1s: exact BBO at t_s+1 → mid = 100 + t_s + 1
            expected_mid_1s = 100.0 + t_s + 1.0
            assert row["mid_at_h_1s"] == pytest.approx(expected_mid_1s, abs=1e-9)

            # markout 1s: (mid_at_1s - mid_at_trade) * sign(buy=+1) = 1.0
            assert row["markout_h_1s"] == pytest.approx(1.0, abs=1e-9)

            # horizon 5s: exact BBO at t_s+5 → mid = 100 + t_s + 5
            expected_mid_5s = 100.0 + t_s + 5.0
            assert row["mid_at_h_5s"] == pytest.approx(expected_mid_5s, abs=1e-9)
            assert row["markout_h_5s"] == pytest.approx(5.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Edge case: trade at very start — no prior BBO
# ---------------------------------------------------------------------------


class TestNoPriorBbo:
    """Trade at start of window with no BBO before it → NaN mid and markouts."""

    def test_mid_at_trade_nan_when_no_prior_bbo(self, con, data_root):
        # BBO arrives AFTER the trade.
        bbo_rows = [
            (BASE_NS + 2 * NS_PER_S, 99.5, 100.5),  # first BBO at t=2s
        ]
        trade_rows = [
            (BASE_NS + NS_PER_S, 100.0, 1.0, "buy"),  # trade at t=1s (before any BBO)
        ]
        _write_bbo(data_root, "hyperliquid", "perp", "BTC", bbo_rows)
        _write_trades(data_root, "hyperliquid", "perp", "BTC", trade_rows)

        df = trade_markouts(
            con,
            venue="hyperliquid",
            product_type="perp",
            symbol="BTC",
            start_ns=BASE_NS,
            end_ns=BASE_NS + 10 * NS_PER_S,
            horizons_s=(1,),
        )

        assert len(df) == 1
        assert pd.isna(df.iloc[0]["mid_at_trade"])
        assert pd.isna(df.iloc[0]["markout_h_1s"])

    def test_row_still_present_when_no_prior_bbo(self, con, data_root):
        """The trade row is still included; only mid/markout columns are NaN."""
        bbo_rows = [(BASE_NS + 5 * NS_PER_S, 99.5, 100.5)]
        trade_rows = [(BASE_NS + NS_PER_S, 100.0, 1.0, "sell")]
        _write_bbo(data_root, "hyperliquid", "perp", "BTC", bbo_rows)
        _write_trades(data_root, "hyperliquid", "perp", "BTC", trade_rows)

        df = trade_markouts(
            con,
            venue="hyperliquid",
            product_type="perp",
            symbol="BTC",
            start_ns=BASE_NS,
            end_ns=BASE_NS + 10 * NS_PER_S,
            horizons_s=(1,),
        )

        assert len(df) == 1
        assert df.iloc[0]["price"] == pytest.approx(100.0)
        assert df.iloc[0]["aggressor_side"] == "sell"


# ---------------------------------------------------------------------------
# Edge case: horizon falls past end_ns
# ---------------------------------------------------------------------------


class TestHorizonPastEndNs:
    """Horizon that falls past end_ns → markout NaN."""

    def test_markout_nan_when_horizon_past_end(self, con, data_root):
        # Window is only 3 s; horizon 5s falls past it.
        bbo_rows = [
            (BASE_NS + 0 * NS_PER_S, 99.5, 100.5),
            (BASE_NS + 1 * NS_PER_S, 100.0, 101.0),
            (BASE_NS + 3 * NS_PER_S, 101.0, 102.0),
        ]
        trade_rows = [
            (BASE_NS + NS_PER_S, 100.0, 1.0, "buy"),
        ]
        _write_bbo(data_root, "hyperliquid", "perp", "BTC", bbo_rows)
        _write_trades(data_root, "hyperliquid", "perp", "BTC", trade_rows)

        end_ns = BASE_NS + 3 * NS_PER_S

        df = trade_markouts(
            con,
            venue="hyperliquid",
            product_type="perp",
            symbol="BTC",
            start_ns=BASE_NS,
            end_ns=end_ns,
            horizons_s=(5,),  # 1s trade + 5s horizon = 6s > end_ns=3s
        )

        assert len(df) == 1
        assert pd.isna(df.iloc[0]["mid_at_h_5s"])
        assert pd.isna(df.iloc[0]["markout_h_5s"])

    def test_markout_present_when_horizon_within_end(self, con, data_root):
        """Horizon within window → non-NaN markout."""
        bbo_rows = [
            (BASE_NS + 0 * NS_PER_S, 99.5, 100.5),
            (BASE_NS + 1 * NS_PER_S, 100.0, 101.0),
            (BASE_NS + 3 * NS_PER_S, 102.0, 103.0),
        ]
        trade_rows = [(BASE_NS + NS_PER_S, 100.0, 1.0, "buy")]
        _write_bbo(data_root, "hyperliquid", "perp", "BTC", bbo_rows)
        _write_trades(data_root, "hyperliquid", "perp", "BTC", trade_rows)

        end_ns = BASE_NS + 10 * NS_PER_S

        df = trade_markouts(
            con,
            venue="hyperliquid",
            product_type="perp",
            symbol="BTC",
            start_ns=BASE_NS,
            end_ns=end_ns,
            horizons_s=(2,),
        )

        assert len(df) == 1
        assert not pd.isna(df.iloc[0]["markout_h_2s"])


# ---------------------------------------------------------------------------
# Sign convention tests
# ---------------------------------------------------------------------------


class TestSignConvention:
    """Verify the four sign-combination cases."""

    def _setup(self, data_root: Path, trade_side: str) -> None:
        """Write BBO with mid going UP from 100 to 102 over 3 s, one trade at t=1s."""
        bbo_rows = [
            (BASE_NS + 0 * NS_PER_S, 99.5, 100.5),   # mid = 100
            (BASE_NS + 1 * NS_PER_S, 100.5, 101.5),   # mid = 101  (trade here)
            (BASE_NS + 2 * NS_PER_S, 101.5, 102.5),   # mid = 102
        ]
        trade_rows = [(BASE_NS + NS_PER_S, 101.0, 1.0, trade_side)]
        _write_bbo(data_root, "hyperliquid", "perp", "BTC", bbo_rows)
        _write_trades(data_root, "hyperliquid", "perp", "BTC", trade_rows)

    def test_buy_aggressor_mid_going_up_positive_markout(self, con, data_root):
        """Buy + mid goes up → positive markout (informed buy)."""
        self._setup(data_root, "buy")
        df = trade_markouts(
            con,
            venue="hyperliquid",
            product_type="perp",
            symbol="BTC",
            start_ns=BASE_NS,
            end_ns=BASE_NS + 5 * NS_PER_S,
            horizons_s=(1,),
        )
        assert df.iloc[0]["markout_h_1s"] > 0

    def test_buy_aggressor_mid_going_down_negative_markout(self, con, data_root):
        """Buy + mid goes DOWN → negative markout (uninformed buy)."""
        # Override BBO to make mid go down.
        bbo_rows = [
            (BASE_NS + 0 * NS_PER_S, 101.5, 102.5),   # mid = 102
            (BASE_NS + 1 * NS_PER_S, 100.5, 101.5),   # mid = 101 (trade here)
            (BASE_NS + 2 * NS_PER_S, 99.5, 100.5),    # mid = 100
        ]
        trade_rows = [(BASE_NS + NS_PER_S, 101.0, 1.0, "buy")]
        _write_bbo(data_root, "hyperliquid", "perp", "BTC", bbo_rows)
        _write_trades(data_root, "hyperliquid", "perp", "BTC", trade_rows)

        df = trade_markouts(
            con,
            venue="hyperliquid",
            product_type="perp",
            symbol="BTC",
            start_ns=BASE_NS,
            end_ns=BASE_NS + 5 * NS_PER_S,
            horizons_s=(1,),
        )
        assert df.iloc[0]["markout_h_1s"] < 0

    def test_sell_aggressor_mid_going_down_positive_markout(self, con, data_root):
        """Sell + mid goes down → positive markout (informed sell)."""
        bbo_rows = [
            (BASE_NS + 0 * NS_PER_S, 101.5, 102.5),
            (BASE_NS + 1 * NS_PER_S, 100.5, 101.5),   # mid = 101 (trade here)
            (BASE_NS + 2 * NS_PER_S, 99.5, 100.5),    # mid = 100
        ]
        trade_rows = [(BASE_NS + NS_PER_S, 101.0, 1.0, "sell")]
        _write_bbo(data_root, "hyperliquid", "perp", "BTC", bbo_rows)
        _write_trades(data_root, "hyperliquid", "perp", "BTC", trade_rows)

        df = trade_markouts(
            con,
            venue="hyperliquid",
            product_type="perp",
            symbol="BTC",
            start_ns=BASE_NS,
            end_ns=BASE_NS + 5 * NS_PER_S,
            horizons_s=(1,),
        )
        assert df.iloc[0]["markout_h_1s"] > 0

    def test_sell_aggressor_mid_going_up_negative_markout(self, con, data_root):
        """Sell + mid goes up → negative markout (uninformed sell)."""
        self._setup(data_root, "sell")
        df = trade_markouts(
            con,
            venue="hyperliquid",
            product_type="perp",
            symbol="BTC",
            start_ns=BASE_NS,
            end_ns=BASE_NS + 5 * NS_PER_S,
            horizons_s=(1,),
        )
        assert df.iloc[0]["markout_h_1s"] < 0


# ---------------------------------------------------------------------------
# Multiple horizons in one call
# ---------------------------------------------------------------------------


class TestMultipleHorizons:
    def test_all_horizons_returned_as_columns(self, con, data_root):
        bbo_rows = [
            (BASE_NS + 0 * NS_PER_S, 99.5, 100.5),
            (BASE_NS + 1 * NS_PER_S, 100.0, 101.0),
            (BASE_NS + 5 * NS_PER_S, 104.0, 105.0),
            (BASE_NS + 30 * NS_PER_S, 129.0, 130.0),
            (BASE_NS + 60 * NS_PER_S, 159.0, 160.0),
        ]
        trade_rows = [(BASE_NS + NS_PER_S, 100.0, 1.0, "buy")]
        _write_bbo(data_root, "hyperliquid", "perp", "BTC", bbo_rows)
        _write_trades(data_root, "hyperliquid", "perp", "BTC", trade_rows)

        df = trade_markouts(
            con,
            venue="hyperliquid",
            product_type="perp",
            symbol="BTC",
            start_ns=BASE_NS,
            end_ns=BASE_NS + 120 * NS_PER_S,
            horizons_s=(1, 5, 30, 60),
        )

        assert len(df) == 1
        for h in (1, 5, 30, 60):
            assert f"mid_at_h_{h}s" in df.columns
            assert f"markout_h_{h}s" in df.columns

    def test_correct_values_for_multiple_horizons(self, con, data_root):
        """mid path rises by 1 per second; verify 4 horizons.

        Trade at t=5 s; BBO at every second 0..65 s so each horizon (1,5,30,60)
        has an exact BBO row at t+h.  end_ns=120s so all horizons are in-window.
        """
        bbo_rows = []
        for i in range(66):  # 0..65 s, covers 5+60=65 max
            t = BASE_NS + i * NS_PER_S
            mid = 100.0 + i
            bbo_rows.append((t, mid - 0.5, mid + 0.5))
        trade_rows = [(BASE_NS + 5 * NS_PER_S, 105.0, 1.0, "buy")]
        _write_bbo(data_root, "hyperliquid", "perp", "BTC", bbo_rows)
        _write_trades(data_root, "hyperliquid", "perp", "BTC", trade_rows)

        df = trade_markouts(
            con,
            venue="hyperliquid",
            product_type="perp",
            symbol="BTC",
            start_ns=BASE_NS,
            end_ns=BASE_NS + 120 * NS_PER_S,
            horizons_s=(1, 5, 30, 60),
        )

        row = df.iloc[0]
        assert row["mid_at_trade"] == pytest.approx(105.0, abs=1e-9)
        assert row["markout_h_1s"] == pytest.approx(1.0, abs=1e-9)
        assert row["markout_h_5s"] == pytest.approx(5.0, abs=1e-9)
        assert row["markout_h_30s"] == pytest.approx(30.0, abs=1e-9)
        assert row["markout_h_60s"] == pytest.approx(60.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Output schema tests
# ---------------------------------------------------------------------------


class TestOutputSchema:
    def test_one_row_per_trade(self, con, data_root):
        bbo_rows = [(BASE_NS + i * NS_PER_S, 99.5 + i, 100.5 + i) for i in range(20)]
        trade_rows = [
            (BASE_NS + 2 * NS_PER_S, 100.0, 0.5, "buy"),
            (BASE_NS + 3 * NS_PER_S, 101.0, 0.3, "sell"),
            (BASE_NS + 7 * NS_PER_S, 105.0, 1.0, "buy"),
        ]
        _write_bbo(data_root, "hyperliquid", "perp", "BTC", bbo_rows)
        _write_trades(data_root, "hyperliquid", "perp", "BTC", trade_rows)

        df = trade_markouts(
            con,
            venue="hyperliquid",
            product_type="perp",
            symbol="BTC",
            start_ns=BASE_NS,
            end_ns=BASE_NS + 15 * NS_PER_S,
            horizons_s=(1,),
        )

        assert len(df) == 3

    def test_required_columns_present(self, con, data_root):
        bbo_rows = [(BASE_NS + i * NS_PER_S, 99.5 + i, 100.5 + i) for i in range(10)]
        trade_rows = [(BASE_NS + 2 * NS_PER_S, 100.0, 1.0, "buy")]
        _write_bbo(data_root, "hyperliquid", "perp", "BTC", bbo_rows)
        _write_trades(data_root, "hyperliquid", "perp", "BTC", trade_rows)

        df = trade_markouts(
            con,
            venue="hyperliquid",
            product_type="perp",
            symbol="BTC",
            start_ns=BASE_NS,
            end_ns=BASE_NS + 10 * NS_PER_S,
            horizons_s=(1, 5),
        )

        required = {"ts_ns", "price", "size", "aggressor_side", "mid_at_trade",
                    "mid_at_h_1s", "markout_h_1s", "mid_at_h_5s", "markout_h_5s"}
        assert required.issubset(set(df.columns))

    def test_empty_window_returns_empty_df(self, con, data_root):
        """No trades in window → empty DataFrame with correct columns."""
        bbo_rows = [(BASE_NS + i * NS_PER_S, 99.5 + i, 100.5 + i) for i in range(5)]
        _write_bbo(data_root, "hyperliquid", "perp", "BTC", bbo_rows)
        # No trades written at all.

        df = trade_markouts(
            con,
            venue="hyperliquid",
            product_type="perp",
            symbol="BTC",
            start_ns=BASE_NS,
            end_ns=BASE_NS + 5 * NS_PER_S,
            horizons_s=(1, 5),
        )

        assert len(df) == 0
        assert "mid_at_trade" in df.columns
        assert "markout_h_1s" in df.columns
        assert "markout_h_5s" in df.columns

    def test_dtypes(self, con, data_root):
        bbo_rows = [(BASE_NS + i * NS_PER_S, 99.5 + i, 100.5 + i) for i in range(10)]
        trade_rows = [(BASE_NS + 2 * NS_PER_S, 100.0, 1.0, "buy")]
        _write_bbo(data_root, "hyperliquid", "perp", "BTC", bbo_rows)
        _write_trades(data_root, "hyperliquid", "perp", "BTC", trade_rows)

        df = trade_markouts(
            con,
            venue="hyperliquid",
            product_type="perp",
            symbol="BTC",
            start_ns=BASE_NS,
            end_ns=BASE_NS + 10 * NS_PER_S,
            horizons_s=(1,),
        )

        assert df["ts_ns"].dtype == "int64"
        assert df["price"].dtype == "float64"
        assert df["size"].dtype == "float64"
        assert df["mid_at_trade"].dtype == "float64"
        assert df["markout_h_1s"].dtype == "float64"

    def test_unknown_side_excluded(self, con, data_root):
        """'unknown' aggressor side rows are filtered out upstream."""
        bbo_rows = [(BASE_NS + i * NS_PER_S, 99.5 + i, 100.5 + i) for i in range(10)]
        trade_rows = [
            (BASE_NS + 2 * NS_PER_S, 100.0, 1.0, "buy"),
            (BASE_NS + 3 * NS_PER_S, 101.0, 0.5, "unknown"),
        ]
        _write_bbo(data_root, "hyperliquid", "perp", "BTC", bbo_rows)
        _write_trades(data_root, "hyperliquid", "perp", "BTC", trade_rows)

        df = trade_markouts(
            con,
            venue="hyperliquid",
            product_type="perp",
            symbol="BTC",
            start_ns=BASE_NS,
            end_ns=BASE_NS + 10 * NS_PER_S,
            horizons_s=(1,),
        )

        assert len(df) == 1
        assert df.iloc[0]["aggressor_side"] == "buy"
