"""Polymarket reference-feed loading helpers.

Covers:
- Constants for Binance BBO/kline symbol mapping and path layout
- _bucket_to_ref_events: OHLC bucketing helper (used by BBO + 1s-klines paths)

Extracted verbatim from polymarket.py — no logic changes.
"""

from __future__ import annotations

from hlanalysis.marketdata.ohlc import resample_ohlc

from ..core.events import ReferenceEvent

# One 1m bar in ns — coverage tolerance for the strike guard.
_BAR_NS = 60 * 1_000_000_000

# Map PM reference_symbol → Binance perp partition symbol for the BBO-tick
# reference variant (cadence-sweep path). Only BTC has overlapping recorded
# tick coverage right now.
_BINANCE_PERP_REF_SYMBOL = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT"}
_BINANCE_PERP_DATA_SUBPATH = "venue=binance/product_type=perp/mechanism=clob"
# reference_symbol → Binance SPOT kline symbol used for strike resolution +
# the coupled fetch (SHR-54). Spot symbols match the perp strings on Binance.
# Underlyings absent here (e.g. WTI → Pyth klines) are not Binance-fetchable.
_BINANCE_SPOT_KLINE_SYMBOL = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT"}

_DEFAULT_KLINES_SUBDIR = "btc_klines"
# Genuine Binance 1s klines for the `klines_1s` reference source — kept in a
# separate subdir so the canonical 1m `btc_klines` (PM strike resolution) is
# untouched. Populated on demand by scripts/fetch_binance_1s_klines.py.
_DEFAULT_KLINES_1S_SUBDIR = "btc_klines_1s"


# ---- OHLC bucketing helper --------------------------------------------------


def _bucket_to_ref_events(
    samples: list[tuple[int, float, float, float, float]],
    *,
    symbol: str,
    bucket_ns: int,
) -> list[ReferenceEvent]:
    """Bucket ``(ts_ns, high, low, close, open_val)`` samples into ``bucket_ns``-wide
    OHLC bars and return a list of ``ReferenceEvent`` objects.

    The H/L/C/ts computation is delegated to the canonical ``resample_ohlc``
    function (single source of truth).  The ``open`` field — the *first* price
    seen in each bucket, used by the runner for strike derivation — is tracked
    in a parallel first-price accumulator so it is preserved without altering
    the resampling contract.

    ``samples`` must be a list (not a one-shot iterator) because we iterate it
    twice: once to collect per-bucket opens, once inside ``resample_ohlc``.
    """
    if not samples:
        return []

    # Collect the open (first-seen price) for each bucket in one pass.
    bucket_opens: dict[int, float] = {}
    for ts, _high, _low, _close, open_val in samples:
        b = ts // bucket_ns
        if b not in bucket_opens:
            bucket_opens[b] = open_val

    # resample_ohlc consumes (ts, high, low, close) and yields (last_ts, h, l, c).
    ohlc_bars = list(
        resample_ohlc(
            ((ts, high, low, close) for ts, high, low, close, _open in samples),
            bucket_ns=bucket_ns,
        )
    )

    # Pair each bar with its bucket open.  Bars are emitted in bucket order so
    # the i-th bar corresponds to the i-th distinct bucket key.
    bucket_keys = list(dict.fromkeys(ts // bucket_ns for ts, *_ in samples))
    return [
        ReferenceEvent(
            ts_ns=last_ts,
            symbol=symbol,
            high=h,
            low=l,
            close=c,
            open=bucket_opens[bucket_keys[i]],
        )
        for i, (last_ts, h, l, c) in enumerate(ohlc_bars)
    ]
