"""Equivalence tests pinning analysis-subsystem behaviour across the
vectorization refactor.

Each test embeds a *reference* implementation copied verbatim from the
pre-refactor source (the slow Python-loop / iterrows version) and asserts that
the production function produces numerically equivalent output.  These tests are
written BEFORE the rewrite and must stay green afterwards — they are the safety
net for "numerically equivalent within float tolerance".

The DuckDB-backed functions are exercised through small synthetic parquet
fixtures laid out in the recorder's hive-partition structure.
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
from hlanalysis.analysis.book import mid_path
from hlanalysis.analysis.helpers import glob_for
from hlanalysis.analysis.microstructure import (
    book_imbalance,
    cross_correlation,
    quoted_spread_bps,
    returns_resampled,
)

BASE_NS = 1_000_000_000_000_000_000
NS_PER_S = 1_000_000_000

# ---------------------------------------------------------------------------
# Fixture plumbing
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


def _write_bbo(root: Path, rows: list[tuple]) -> None:
    """rows: (local_recv_ts, bid_px, ask_px)"""
    table = pa.table(
        {
            "venue": ["hyperliquid"] * len(rows),
            "product_type": ["perp"] * len(rows),
            "mechanism": ["clob"] * len(rows),
            "symbol": ["BTC"] * len(rows),
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
    part = _partition_path(root, "hyperliquid", "perp", "bbo", "BTC")
    _write_parquet(part / "part0.parquet", table)


def _write_book_snapshot(root: Path, rows: list[tuple]) -> None:
    """rows: (local_recv_ts, bid_pxs, bid_szs, ask_pxs, ask_szs)"""
    table = pa.table(
        {
            "venue": ["hyperliquid"] * len(rows),
            "product_type": ["perp"] * len(rows),
            "mechanism": ["clob"] * len(rows),
            "symbol": ["BTC"] * len(rows),
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
    part = _partition_path(root, "hyperliquid", "perp", "book_snapshot", "BTC")
    _write_parquet(part / "part0.parquet", table)


# ---------------------------------------------------------------------------
# Reference (pre-refactor) implementations, copied verbatim from source.
# ---------------------------------------------------------------------------


def _old_mid_path(con, *, start_ns, end_ns, resample_ms) -> pd.DataFrame:
    step_ns = resample_ms * 1_000_000
    glob = glob_for(venue="hyperliquid", product_type="perp", event="bbo", symbol="BTC")
    sql = f"""
        SELECT local_recv_ts AS ts_ns,
               (bid_px + ask_px) / 2.0 AS mid
        FROM read_parquet('{glob}', hive_partitioning=true)
        WHERE local_recv_ts <= ?
        ORDER BY local_recv_ts ASC
    """
    try:
        raw = con.execute(sql, [end_ns]).df()
    except duckdb.IOException:
        raw = pd.DataFrame({"ts_ns": pd.Series([], dtype="int64"), "mid": pd.Series([], dtype="float64")})

    grid = list(range(start_ns, end_ns + 1, step_ns))
    result_ts: list[int] = []
    result_mid: list[float | None] = []

    if raw.empty:
        return pd.DataFrame({"ts_ns": pd.array(grid, dtype="int64"), "mid": [None] * len(grid)})

    raw_ts = raw["ts_ns"].to_numpy()
    raw_mid = raw["mid"].to_numpy()

    for g in grid:
        idx = int(np.searchsorted(raw_ts, g, side="right")) - 1
        if idx < 0:
            result_ts.append(g)
            result_mid.append(None)
        else:
            result_ts.append(g)
            result_mid.append(float(raw_mid[idx]))

    return pd.DataFrame(
        {
            "ts_ns": pd.array(result_ts, dtype="int64"),
            "mid": pd.array(result_mid, dtype="float64"),
        }
    )


def _old_quoted_spread_bps(con, *, start_ns, end_ns, resample_s) -> pd.DataFrame:
    glob = glob_for(venue="hyperliquid", product_type="perp", event="bbo", symbol="BTC")
    sql = f"""
        SELECT
            local_recv_ts                                AS ts_ns,
            (ask_px - bid_px) / ((ask_px + bid_px) / 2.0) * 1e4 AS spread_bps
        FROM read_parquet('{glob}', hive_partitioning=true)
        WHERE local_recv_ts <= ?
        ORDER BY local_recv_ts ASC
    """
    try:
        raw = con.execute(sql, [end_ns]).df()
    except duckdb.IOException:
        raw = pd.DataFrame({"ts_ns": pd.Series([], dtype="int64"), "spread_bps": pd.Series([], dtype="float64")})

    raw["ts_ns"] = raw["ts_ns"].astype("int64")
    raw["spread_bps"] = raw["spread_bps"].astype("float64")

    step_ns = resample_s * NS_PER_S
    grid = list(range(start_ns, end_ns + 1, step_ns))

    if raw.empty:
        return pd.DataFrame(
            {
                "ts_ns": pd.array(grid, dtype="int64"),
                "spread_bps": pd.array([float("nan")] * len(grid), dtype="float64"),
            }
        )

    raw_ts = raw["ts_ns"].to_numpy()
    raw_spread = raw["spread_bps"].to_numpy()

    result_ts: list[int] = []
    result_spread: list[float] = []
    for g in grid:
        idx = int(np.searchsorted(raw_ts, g, side="right")) - 1
        if idx < 0:
            result_ts.append(g)
            result_spread.append(float("nan"))
        else:
            result_ts.append(g)
            result_spread.append(float(raw_spread[idx]))

    return pd.DataFrame(
        {
            "ts_ns": pd.array(result_ts, dtype="int64"),
            "spread_bps": pd.array(result_spread, dtype="float64"),
        }
    )


def _old_book_imbalance(con, *, start_ns, end_ns, levels) -> pd.DataFrame:
    glob = glob_for(venue="hyperliquid", product_type="perp", event="book_snapshot", symbol="BTC")
    sql = f"""
        SELECT local_recv_ts AS ts_ns, bid_sz, ask_sz
        FROM read_parquet('{glob}', hive_partitioning=true)
        WHERE local_recv_ts BETWEEN ? AND ?
        ORDER BY local_recv_ts ASC
    """
    try:
        raw = con.execute(sql, [start_ns, end_ns]).df()
    except duckdb.IOException:
        return pd.DataFrame({"ts_ns": pd.Series([], dtype="int64"), "imbalance": pd.Series([], dtype="float64")})

    if raw.empty:
        return pd.DataFrame({"ts_ns": pd.Series([], dtype="int64"), "imbalance": pd.Series([], dtype="float64")})

    ts_list: list[int] = []
    imb_list: list[float] = []
    for _, row in raw.iterrows():
        ts_list.append(int(row["ts_ns"]))
        bid_szs = row["bid_sz"]
        ask_szs = row["ask_sz"]
        bid_sum = sum(float(x) for x in bid_szs[:levels])
        ask_sum = sum(float(x) for x in ask_szs[:levels])
        total = bid_sum + ask_sum
        if total == 0.0:
            imb_list.append(float("nan"))
        else:
            imb_list.append((bid_sum - ask_sum) / total)

    return pd.DataFrame(
        {
            "ts_ns": pd.array(ts_list, dtype="int64"),
            "imbalance": pd.array(imb_list, dtype="float64"),
        }
    )


def _old_log_return_loop(mid_vals: np.ndarray) -> np.ndarray:
    log_ret = np.full(len(mid_vals), float("nan"), dtype="float64")
    for i in range(1, len(mid_vals)):
        prev = mid_vals[i - 1]
        curr = mid_vals[i]
        if not (math.isnan(prev) or math.isnan(curr)) and prev > 0.0:
            log_ret[i] = math.log(curr / prev)
    return log_ret


def _old_cross_correlation(x: pd.Series, y: pd.Series, max_lag: int) -> pd.DataFrame:
    n = len(x)
    if max_lag >= n:
        raise ValueError("max_lag")
    x_arr = x.reset_index(drop=True)
    y_arr = y.reset_index(drop=True)
    lags: list[int] = []
    ccfs: list[float] = []
    for lag in range(-max_lag, max_lag + 1):
        lags.append(lag)
        ccfs.append(float(x_arr.corr(y_arr.shift(lag))))
    return pd.DataFrame({"lag": pd.array(lags, dtype="int64"), "ccf": pd.array(ccfs, dtype="float64")})


# ---------------------------------------------------------------------------
# Comparison helper
# ---------------------------------------------------------------------------


def _assert_close(a: pd.DataFrame, b: pd.DataFrame, value_col: str, *, rtol=1e-12, atol=1e-12) -> None:
    assert list(a.columns) == list(b.columns)
    assert len(a) == len(b)
    np.testing.assert_array_equal(a.iloc[:, 0].to_numpy(), b.iloc[:, 0].to_numpy())
    np.testing.assert_allclose(
        a[value_col].to_numpy(dtype="float64"),
        b[value_col].to_numpy(dtype="float64"),
        rtol=rtol,
        atol=atol,
        equal_nan=True,
    )


# ---------------------------------------------------------------------------
# A. as-of LOCF helper (new shared function) — RED first, then GREEN.
# ---------------------------------------------------------------------------


class TestAsofLocf:
    def test_matches_old_asof_mid_semantics(self):
        from hlanalysis.analysis.helpers import asof_locf

        source_ts = np.array([10, 20, 30], dtype="int64")
        source_val = np.array([1.0, 2.0, 3.0], dtype="float64")
        query = np.array([5, 10, 15, 20, 25, 30, 35], dtype="int64")
        # last-known value at or before each query ts; NaN before first source.
        out = asof_locf(query, source_ts, source_val)
        expected = np.array([np.nan, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
        np.testing.assert_array_equal(out, expected)

    def test_empty_source_returns_all_nan(self):
        from hlanalysis.analysis.helpers import asof_locf

        out = asof_locf(np.array([1, 2, 3], dtype="int64"), np.array([], dtype="int64"), np.array([], dtype="float64"))
        assert out.shape == (3,)
        assert np.isnan(out).all()

    def test_empty_query_returns_empty(self):
        from hlanalysis.analysis.helpers import asof_locf

        out = asof_locf(np.array([], dtype="int64"), np.array([10], dtype="int64"), np.array([1.0]))
        assert out.shape == (0,)

    def test_randomised_against_naive_loop(self):
        from hlanalysis.analysis.helpers import asof_locf

        rng = np.random.default_rng(123)
        for _ in range(20):
            src_ts = np.sort(rng.integers(0, 1000, size=rng.integers(0, 30)).astype("int64"))
            src_val = rng.normal(size=len(src_ts))
            q = np.sort(rng.integers(-50, 1050, size=rng.integers(0, 40)).astype("int64"))
            out = asof_locf(q, src_ts, src_val)
            naive = np.full(len(q), np.nan)
            for i, t in enumerate(q):
                idx = int(np.searchsorted(src_ts, t, side="right")) - 1
                if idx >= 0:
                    naive[i] = src_val[idx]
            np.testing.assert_array_equal(out, naive)


# ---------------------------------------------------------------------------
# A/D. mid_path, quoted_spread_bps equivalence (LOCF consolidation).
# ---------------------------------------------------------------------------


class TestMidPathEquivalence:
    def test_matches_reference(self, con, data_root):
        rng = np.random.default_rng(7)
        rows = []
        for i in range(40):
            t = BASE_NS + int(rng.integers(0, 50)) * NS_PER_S + i  # jittered, strictly increasing via +i
            mid = 100.0 + rng.normal()
            rows.append((BASE_NS + i * 7 * 10**8, mid - 0.5, mid + 0.5))
        rows.sort(key=lambda r: r[0])
        _write_bbo(data_root, rows)
        for resample_ms in (100, 250, 1000):
            new = mid_path(
                con,
                venue="hyperliquid",
                product_type="perp",
                symbol="BTC",
                start_ns=BASE_NS,
                end_ns=BASE_NS + 30 * NS_PER_S,
                resample_ms=resample_ms,
            )
            old = _old_mid_path(con, start_ns=BASE_NS, end_ns=BASE_NS + 30 * NS_PER_S, resample_ms=resample_ms)
            _assert_close(new, old, "mid")

    def test_matches_reference_no_data(self, con, data_root):
        new = mid_path(
            con,
            venue="hyperliquid",
            product_type="perp",
            symbol="BTC",
            start_ns=BASE_NS,
            end_ns=BASE_NS + 5 * NS_PER_S,
            resample_ms=1000,
        )
        old = _old_mid_path(con, start_ns=BASE_NS, end_ns=BASE_NS + 5 * NS_PER_S, resample_ms=1000)
        _assert_close(new, old, "mid")


class TestQuotedSpreadEquivalence:
    def test_matches_reference(self, con, data_root):
        rng = np.random.default_rng(11)
        rows = []
        for i in range(30):
            mid = 100.0 + rng.normal()
            half = abs(rng.normal()) * 0.05 + 0.01
            rows.append((BASE_NS + i * 13 * 10**8, mid - half, mid + half))
        _write_bbo(data_root, rows)
        for resample_s in (1, 2, 5):
            new = quoted_spread_bps(
                con,
                venue="hyperliquid",
                product_type="perp",
                symbol="BTC",
                start_ns=BASE_NS,
                end_ns=BASE_NS + 40 * NS_PER_S,
                resample_s=resample_s,
            )
            old = _old_quoted_spread_bps(con, start_ns=BASE_NS, end_ns=BASE_NS + 40 * NS_PER_S, resample_s=resample_s)
            _assert_close(new, old, "spread_bps")

    def test_matches_reference_no_data(self, con, data_root):
        new = quoted_spread_bps(
            con,
            venue="hyperliquid",
            product_type="perp",
            symbol="BTC",
            start_ns=BASE_NS,
            end_ns=BASE_NS + 5 * NS_PER_S,
            resample_s=1,
        )
        old = _old_quoted_spread_bps(con, start_ns=BASE_NS, end_ns=BASE_NS + 5 * NS_PER_S, resample_s=1)
        _assert_close(new, old, "spread_bps")


# ---------------------------------------------------------------------------
# B. book_imbalance equivalence (iterrows -> column ops).
# ---------------------------------------------------------------------------


class TestBookImbalanceEquivalence:
    def test_matches_reference_ragged_levels(self, con, data_root):
        rng = np.random.default_rng(3)
        rows = []
        for i in range(25):
            depth = int(rng.integers(1, 6))
            bid_sz = list(np.abs(rng.normal(size=depth)).round(4))
            ask_sz = list(np.abs(rng.normal(size=depth)).round(4))
            bid_px = list((50000.0 - np.arange(depth) * 10.0))
            ask_px = list((50010.0 + np.arange(depth) * 10.0))
            rows.append((BASE_NS + i * NS_PER_S, bid_px, bid_sz, ask_px, ask_sz))
        # add a zero-total row to exercise the NaN branch
        rows.append((BASE_NS + 30 * NS_PER_S, [50000.0], [0.0], [50010.0], [0.0]))
        _write_book_snapshot(data_root, rows)
        for levels in (1, 2, 3):
            new = book_imbalance(
                con,
                venue="hyperliquid",
                product_type="perp",
                symbol="BTC",
                start_ns=BASE_NS,
                end_ns=BASE_NS + 60 * NS_PER_S,
                levels=levels,
            )
            old = _old_book_imbalance(con, start_ns=BASE_NS, end_ns=BASE_NS + 60 * NS_PER_S, levels=levels)
            _assert_close(new, old, "imbalance")

    def test_matches_reference_no_data(self, con, data_root):
        new = book_imbalance(
            con,
            venue="hyperliquid",
            product_type="perp",
            symbol="BTC",
            start_ns=BASE_NS,
            end_ns=BASE_NS + 5 * NS_PER_S,
            levels=1,
        )
        old = _old_book_imbalance(con, start_ns=BASE_NS, end_ns=BASE_NS + 5 * NS_PER_S, levels=1)
        _assert_close(new, old, "imbalance")


# ---------------------------------------------------------------------------
# D. returns_resampled equivalence (log loop -> np.diff(np.log)).
# ---------------------------------------------------------------------------


class TestReturnsResampledEquivalence:
    def test_matches_reference_end_to_end(self, con, data_root):
        rng = np.random.default_rng(5)
        rows = []
        for i in range(50):
            mid = 100.0 + np.cumsum(rng.normal(size=1))[0]
            rows.append((BASE_NS + i * NS_PER_S, mid - 0.5, mid + 0.5))
        _write_bbo(data_root, rows)
        df = returns_resampled(
            con,
            venue="hyperliquid",
            product_type="perp",
            symbol="BTC",
            start_ns=BASE_NS,
            end_ns=BASE_NS + 49 * NS_PER_S,
            dt_ms=1000,
        )
        # Reference: recompute log-returns from the same resampled mids.
        mids = mid_path(
            con,
            venue="hyperliquid",
            product_type="perp",
            symbol="BTC",
            start_ns=BASE_NS,
            end_ns=BASE_NS + 49 * NS_PER_S,
            resample_ms=1000,
        )
        expected = _old_log_return_loop(mids["mid"].to_numpy(dtype="float64"))
        np.testing.assert_allclose(
            df["log_return"].to_numpy(dtype="float64"), expected, rtol=1e-12, atol=1e-12, equal_nan=True
        )

    def test_matches_reference_with_leading_nan_mids(self, con, data_root):
        # first BBO only at t=3s -> grid points 0,1,2 have NaN mid (LOCF gap).
        rows = [(BASE_NS + (i + 3) * NS_PER_S, 99.5 + i, 100.5 + i) for i in range(6)]
        _write_bbo(data_root, rows)
        df = returns_resampled(
            con,
            venue="hyperliquid",
            product_type="perp",
            symbol="BTC",
            start_ns=BASE_NS,
            end_ns=BASE_NS + 8 * NS_PER_S,
            dt_ms=1000,
        )
        mids = mid_path(
            con,
            venue="hyperliquid",
            product_type="perp",
            symbol="BTC",
            start_ns=BASE_NS,
            end_ns=BASE_NS + 8 * NS_PER_S,
            resample_ms=1000,
        )
        expected = _old_log_return_loop(mids["mid"].to_numpy(dtype="float64"))
        np.testing.assert_allclose(
            df["log_return"].to_numpy(dtype="float64"), expected, rtol=1e-12, atol=1e-12, equal_nan=True
        )


# ---------------------------------------------------------------------------
# C. cross_correlation equivalence (pandas per-lag loop -> scipy correlate).
# ---------------------------------------------------------------------------


class TestCrossCorrelationEquivalence:
    def test_matches_reference_random(self):
        rng = np.random.default_rng(17)
        for trial in range(10):
            n = int(rng.integers(50, 400))
            x = pd.Series(rng.normal(size=n))
            y = pd.Series(rng.normal(size=n))
            max_lag = int(rng.integers(1, min(20, n - 1)))
            new = cross_correlation(x, y, max_lag=max_lag)
            old = _old_cross_correlation(x, y, max_lag=max_lag)
            np.testing.assert_array_equal(new["lag"].to_numpy(), old["lag"].to_numpy())
            np.testing.assert_allclose(
                new["ccf"].to_numpy(dtype="float64"),
                old["ccf"].to_numpy(dtype="float64"),
                rtol=1e-9,
                atol=1e-9,
                equal_nan=True,
            )

    def test_matches_reference_with_nans(self):
        rng = np.random.default_rng(29)
        n = 250
        x = rng.normal(size=n)
        y = rng.normal(size=n)
        # punch holes
        x[rng.integers(0, n, size=20)] = np.nan
        y[rng.integers(0, n, size=20)] = np.nan
        xs, ys = pd.Series(x), pd.Series(y)
        for max_lag in (1, 3, 7, 15):
            new = cross_correlation(xs, ys, max_lag=max_lag)
            old = _old_cross_correlation(xs, ys, max_lag=max_lag)
            np.testing.assert_allclose(
                new["ccf"].to_numpy(dtype="float64"),
                old["ccf"].to_numpy(dtype="float64"),
                rtol=1e-9,
                atol=1e-9,
                equal_nan=True,
            )

    def test_matches_reference_autocorrelation(self):
        rng = np.random.default_rng(42)
        n = 2000
        phi = 0.7
        v = np.zeros(n)
        for t in range(1, n):
            v[t] = phi * v[t - 1] + rng.normal()
        s = pd.Series(v)
        new = cross_correlation(s, s, max_lag=10)
        old = _old_cross_correlation(s, s, max_lag=10)
        np.testing.assert_allclose(
            new["ccf"].to_numpy(dtype="float64"),
            old["ccf"].to_numpy(dtype="float64"),
            rtol=1e-9,
            atol=1e-9,
            equal_nan=True,
        )
