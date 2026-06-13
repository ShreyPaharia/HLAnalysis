"""Canonical OHLC bucketer — the ONE implementation of "bucket reference-price
ticks into dt-wide OHLC bars" shared by the live engine and the backtest.

Three independent copies of this logic used to exist:
  - ``engine.market_state._ingest_reference_price`` (streaming, LIVE path)
  - ``backtest.data.hl_hip4._resample_reference``    (generator)
  - ``backtest.data._fastpath_core._resample_reference_rows`` (list twin)

They had to stay bit-identical by manual discipline; drift produced the
σ-sampling regressions in the project history. This module collapses them so
the strategy's σ formula sees the same bars whether it runs live or in sim.

Semantics (preserved exactly from the prior three copies):
  * Bucket assignment is ``floor(ts_ns / bucket_ns)`` (integer division).
  * Within a bucket: ``high = max``, ``low = min`` (ties keep the existing
    extreme), ``close = last tick``.
  * Each emitted bar carries the timestamp of the bucket's LAST tick, so a
    resampled stream stays monotone in ``ts``.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator


def bucket_index(ts_ns: int, bucket_ns: int) -> int:
    """The dt-wide bucket a tick at ``ts_ns`` falls into: ``floor(ts/dt)``."""
    return ts_ns // bucket_ns


def update_bar(bar: tuple[float, float, float], high: float, low: float, close: float) -> tuple[float, float, float]:
    """Merge a ``(high, low, close)`` sample into an in-progress ``(h, l, c)`` bar.

    ``high`` becomes ``max`` (ties keep the existing high), ``low`` becomes
    ``min`` (ties keep the existing low), ``close`` is always the new sample's
    close. The ``>=`` / ``<=`` tie rule matches both prior forms:
    ``h if h >= x else x`` (engine) and ``if x > h: h = x`` (loaders).
    """
    h, l, _c = bar
    return (h if h >= high else high, l if l <= low else low, close)


def resample_ohlc(
    samples: Iterable[tuple[int, float, float, float]], *, bucket_ns: int
) -> Iterator[tuple[int, float, float, float]]:
    """Aggregate ``(ts_ns, high, low, close)`` samples into ``bucket_ns``-wide
    OHLC bars, yielding ``(last_ts_ns, high, low, close)`` per bucket.

    The batch form used by the backtest loaders. The live engine streams ticks
    one at a time and keeps its own ring buffer, so it composes ``bucket_index``
    + ``update_bar`` directly rather than calling this generator — but the merge
    rule is identical, which is the whole point.
    """
    cur_bucket: int | None = None
    h: float = 0.0
    l: float = 0.0
    c: float = 0.0
    last_ts: int = 0
    for ts, high, low, close in samples:
        bucket = bucket_index(ts, bucket_ns)
        if cur_bucket is None:
            cur_bucket = bucket
            h, l, c = high, low, close
            last_ts = ts
        elif bucket != cur_bucket:
            yield last_ts, h, l, c
            cur_bucket = bucket
            h, l, c = high, low, close
            last_ts = ts
        else:
            h, l, c = update_bar((h, l, c), high, low, close)
            last_ts = ts
    if cur_bucket is not None:
        yield last_ts, h, l, c


__all__ = ["bucket_index", "update_bar", "resample_ohlc"]
