"""Pyth Benchmarks WTI 1-minute kline fetcher with CL contract-roll lookup.

WTI Crude Light (CL) futures roll on a fixed schedule: the active "front
month" contract changes roughly 3–5 business days before the prior month's
last-trade day.  Polymarket WTI markets resolve against Pyth's published
"Active Month" price, which follows the same roll.

Pyth exposes historical minute bars via a TradingView UDF shim at:
    https://benchmarks.pyth.network/v1/shims/tradingview/history

The shim caps each response at approximately 5 days of 1-minute bars.
Callers that need longer windows must chunk requests themselves; PM market
windows are 24–48 h so each market fits in a single call.

Contract symbols follow the pattern ``Commodities.WTI{month_code}{year}/USD``
where month codes are the standard futures month letters (H=Mar, J=Apr,
K=May, M=Jun, N=Jul, Q=Aug, …).
"""

import bisect
import datetime

import requests

__all__ = ["active_cl_contract", "fetch_klines_1m", "fetch_window_for_market"]

_PYTH_SHIM_URL = "https://benchmarks.pyth.network/v1/shims/tradingview/history"

# Each entry is (effective_from_iso, pyth_symbol).
# ``effective_from`` is the first calendar date on which the listed contract
# is the active front month on Pyth (i.e. the roll date).  Dates are
# inclusive; the final entry applies until a newer entry is added.
_CL_ACTIVE_TABLE: list[tuple[str, str]] = [
    ("2026-01-21", "Commodities.WTIH6/USD"),  # March-26 contract
    ("2026-02-19", "Commodities.WTIJ6/USD"),  # April-26 contract
    ("2026-03-19", "Commodities.WTIK6/USD"),  # May-26 contract
    ("2026-04-20", "Commodities.WTIM6/USD"),  # June-26 contract
    ("2026-05-19", "Commodities.WTIN6/USD"),  # July-26 contract
    ("2026-06-18", "Commodities.WTIQ6/USD"),  # August-26 contract
]

# Pre-compute sorted keys for bisect
_CL_DATES = [row[0] for row in _CL_ACTIVE_TABLE]


def active_cl_contract(date_iso: str) -> str:
    """Return the Pyth symbol for the active CL front-month contract on *date_iso*.

    Args:
        date_iso: Calendar date in ``YYYY-MM-DD`` format (UTC).

    Returns:
        Pyth symbol string, e.g. ``"Commodities.WTIN6/USD"``.

    For dates before the first entry in ``_CL_ACTIVE_TABLE`` the first
    entry is returned (clamped).
    """
    # bisect_right gives the insertion point after any existing equal key.
    # Subtracting 1 gives the index of the latest entry whose date <= date_iso.
    idx = bisect.bisect_right(_CL_DATES, date_iso) - 1
    # Clamp to first entry for pre-corpus dates
    idx = max(idx, 0)
    return _CL_ACTIVE_TABLE[idx][1]


def fetch_klines_1m(pyth_symbol: str, start_ts_ns: int, end_ts_ns: int) -> list[dict]:
    """Fetch 1-minute kline bars from the Pyth Benchmarks TradingView shim.

    Args:
        pyth_symbol: Pyth symbol, e.g. ``"Commodities.WTIN6/USD"``.
        start_ts_ns: Start timestamp in nanoseconds (inclusive).
        end_ts_ns: End timestamp in nanoseconds (inclusive).

    Returns:
        List of kline dicts with keys ``ts_ns``, ``open``, ``high``, ``low``,
        ``close``.  Returns an empty list if the shim reports a non-ok status
        (e.g. too many datapoints, symbol not found).

    Note:
        No pagination is performed.  The shim caps each call at ~5 days of
        1-minute bars.  Callers must chunk if a longer window is needed.
    """
    params = {
        "symbol": pyth_symbol,
        "resolution": "1",
        "from": start_ts_ns // 1_000_000_000,
        "to": end_ts_ns // 1_000_000_000,
    }
    resp = requests.get(_PYTH_SHIM_URL, params=params)
    resp.raise_for_status()
    data = resp.json()

    if data.get("s") != "ok":
        return []

    rows = []
    for ts, o, h, l, c in zip(
        data["t"], data["o"], data["h"], data["l"], data["c"]
    ):
        rows.append(
            {
                "ts_ns": ts * 1_000_000_000,
                "open": o,
                "high": h,
                "low": l,
                "close": c,
            }
        )
    return rows


def fetch_window_for_market(
    market_end_ts_ns: int,
    *,
    lookback_seconds: int = 86400 * 3,
) -> list[dict]:
    """Fetch WTI 1m bars for the window ending at *market_end_ts_ns*.

    Derives the calendar date from *market_end_ts_ns* (UTC), looks up the
    active CL contract for that date, then fetches bars for
    ``[end - lookback_seconds, end + 3600]`` (1-hour look-ahead included so
    the final candle is captured even with slight timestamp drift).

    The default 3-day lookback covers Monday markets that need Friday 21:00Z
    prior close data across the weekend.

    Args:
        market_end_ts_ns: Market resolution timestamp in nanoseconds (UTC).
        lookback_seconds: How many seconds before *market_end_ts_ns* to fetch.

    Returns:
        List of kline dicts (same shape as :func:`fetch_klines_1m`).
    """
    end_ts_s = market_end_ts_ns // 1_000_000_000
    date_iso = datetime.datetime.fromtimestamp(end_ts_s, tz=datetime.timezone.utc).strftime(
        "%Y-%m-%d"
    )
    symbol = active_cl_contract(date_iso)

    start_ts_ns = (end_ts_s - lookback_seconds) * 1_000_000_000
    # 1-hour look-ahead buffer
    end_with_buffer_ns = (end_ts_s + 3600) * 1_000_000_000

    return fetch_klines_1m(symbol, start_ts_ns, end_with_buffer_ns)
