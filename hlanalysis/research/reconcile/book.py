"""Recorded market book queries over parquet data."""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import pandas as pd

# BookReader is a callable that can be injected for testing.
# Signature: (leg_symbol: str, ts_ns: int, data_root: Path) -> pd.DataFrame | None
# Returns DataFrame with columns: exchange_ts, bid_px, bid_sz, ask_px, ask_sz
# Returns the single closest row at or before ts_ns (LOCF).
BookReader = Callable[[str, int, Path], "pd.DataFrame | None"]


def _default_book_reader(
    leg_symbol: str,
    ts_ns: int,
    data_root: Path,
) -> pd.DataFrame | None:
    """Read the bbo parquet for leg_symbol, returning the row closest to ts_ns.

    Parameters
    ----------
    leg_symbol:
        HL leg symbol, e.g. ``"#4010"``.
    ts_ns:
        Query timestamp in nanoseconds; returns the last row at or before this.
    data_root:
        Root of the HL recorded data tree.

    Returns
    -------
    Single-row DataFrame with cols: exchange_ts, bid_px, bid_sz, ask_px, ask_sz,
    or None if no data is found.
    """
    try:
        import duckdb  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("duckdb is required for recorded book queries") from exc

    pattern = str(
        data_root
        / f"venue=hyperliquid/product_type=prediction_binary/mechanism=clob/event=bbo/symbol={leg_symbol}/**/*.parquet"
    )
    try:
        con = duckdb.connect()
        df: pd.DataFrame = con.execute(
            f"""
            SELECT exchange_ts, bid_px, bid_sz, ask_px, ask_sz
            FROM read_parquet('{pattern}', hive_partitioning=true)
            WHERE exchange_ts <= {ts_ns}
            ORDER BY exchange_ts DESC
            LIMIT 1
            """
        ).df()
        con.close()
        return df if not df.empty else None
    except Exception:
        return None


# RefPriceReader is a callable that can be injected for testing.
# Signature: (ref_symbol: str, ts_ns: int, data_root: Path | None) -> float | None
# Returns the reference (perp mark) price at or before ts_ns (LOCF), or None.
RefPriceReader = Callable[[str, int, "Path | None"], "float | None"]


def _default_ref_price_reader(
    ref_symbol: str,
    ts_ns: int,
    data_root: Path | None,
) -> float | None:
    """Read the recorded HL perp ``mark`` price for ``ref_symbol`` at/before ts_ns.

    Parameters
    ----------
    ref_symbol:
        Reference perp symbol, e.g. ``"BTC"``.
    ts_ns:
        Query timestamp in nanoseconds; returns the last mark at or before this.
    data_root:
        Root of the HL recorded data tree; None yields None (no data).

    Returns
    -------
    The ``mark_px`` of the closest row at or before ts_ns, or None if not found.
    """
    if data_root is None:
        return None
    try:
        import duckdb  # noqa: PLC0415
    except ImportError:
        return None

    pattern = str(
        data_root / f"venue=hyperliquid/product_type=perp/mechanism=clob/event=mark/symbol={ref_symbol}/**/*.parquet"
    )
    try:
        con = duckdb.connect()
        df: pd.DataFrame = con.execute(
            f"""
            SELECT mark_px
            FROM read_parquet('{pattern}', hive_partitioning=true)
            WHERE exchange_ts <= {ts_ns}
            ORDER BY exchange_ts DESC
            LIMIT 1
            """
        ).df()
        con.close()
        if df.empty:
            return None
        return float(df["mark_px"].iloc[0])
    except Exception:
        return None


def recorded_ref_price_at(
    ref_symbol: str,
    ts_ns: int,
    data_root: Path | None,
    reader: RefPriceReader | None = None,
) -> float | None:
    """Return the recorded perp ``mark`` price closest to (at or before) ts_ns.

    Parameters
    ----------
    ref_symbol:
        Reference perp symbol, e.g. ``"BTC"``.
    ts_ns:
        Query timestamp in nanoseconds.
    data_root:
        Root of the HL recorded data tree; may be None when ``reader`` is given.
    reader:
        Injectable reference reader for testing; defaults to the duckdb reader.

    Returns
    -------
    The reference price at that time, or None if unavailable.
    """
    fn = reader or _default_ref_price_reader
    return fn(ref_symbol, ts_ns, data_root)


def recorded_book_at(
    leg_symbol: str,
    ts_ns: int,
    data_root: Path,
    reader: BookReader | None = None,
) -> pd.DataFrame | None:
    """Return the recorded BBO row closest to (and at or before) ts_ns.

    Parameters
    ----------
    leg_symbol:
        HL leg symbol, e.g. ``"#4010"``.
    ts_ns:
        Query timestamp in nanoseconds.
    data_root:
        Root of the HL recorded data tree (e.g. ``../../data``).
    reader:
        Injectable book reader for testing; defaults to the duckdb parquet reader.

    Returns
    -------
    Single-row DataFrame with cols: exchange_ts, bid_px, bid_sz, ask_px, ask_sz,
    or None if no data is found.
    """
    fn = reader or _default_book_reader
    return fn(leg_symbol, ts_ns, data_root)


def book_has_price(
    leg_symbol: str,
    ts_ns: int,
    side: str,
    price: float,
    tol: float = 0.005,
    data_root: Path | None = None,
    reader: BookReader | None = None,
) -> bool:
    """Return True if ``price`` was present in the recorded book at ts_ns within tol.

    Parameters
    ----------
    leg_symbol:
        HL leg symbol.
    ts_ns:
        Timestamp in nanoseconds.
    side:
        ``"BUY"`` or ``"SELL"``.
    price:
        Fill price to verify.
    tol:
        Price tolerance (default 0.5%).  For BUY: price <= ask_px + tol.
        For SELL: price >= bid_px - tol.
    data_root:
        Root of the HL recorded data tree.  Required unless ``reader`` is provided.
    reader:
        Injectable book reader; if provided, ``data_root`` may be None.

    Returns
    -------
    True if the price was achievable in the recorded book at that time.
    """
    if reader is not None:
        # Injectable reader path (data_root may be None in tests)
        row_df = reader(leg_symbol, ts_ns, data_root or Path("."))
    else:
        if data_root is None:
            return False
        row_df = recorded_book_at(leg_symbol, ts_ns, data_root, reader=None)

    if row_df is None or row_df.empty:
        return False

    row = row_df.iloc[0]
    if side.upper() == "BUY":
        ask_px = float(row.get("ask_px", math.nan))
        return not math.isnan(ask_px) and price <= ask_px + tol
    else:
        bid_px = float(row.get("bid_px", math.nan))
        return not math.isnan(bid_px) and price >= bid_px - tol


def book_parity_pct(
    fills_df: pd.DataFrame,
    data_root: Path,
    reader: BookReader | None = None,
) -> float:
    """Return fraction of fills whose price was present in the recorded book.

    Parameters
    ----------
    fills_df:
        DataFrame with cols: ts_ns, side (BUY/SELL), price, symbol.
    data_root:
        Root of the HL recorded data tree.
    reader:
        Injectable book reader for testing.

    Returns
    -------
    Float in [0.0, 1.0], or nan if fills_df is empty.
    """
    if fills_df.empty:
        return math.nan

    hits = 0
    total = len(fills_df)
    for _, row in fills_df.iterrows():
        if book_has_price(
            leg_symbol=str(row["symbol"]),
            ts_ns=int(row["ts_ns"]),
            side=str(row["side"]),
            price=float(row["price"]),
            data_root=data_root,
            reader=reader,
        ):
            hits += 1
    return hits / total
