"""HL HIP-4 recorded-data `DataSource`.

Reads the recorder's parquet partitions under
``data/venue=hyperliquid/product_type=prediction_binary/mechanism=clob/event=*/``
and emits the §3 ``MarketEvent`` union in monotone-``ts_ns`` order.

Reference-price source: HL perp BTC BBO mid (falls back to perp ``mark`` if BBO
is empty over the requested window). HIP-4 binaries settle off the HL perp mark,
so perp is the correct underlying. BBO is denser than mark in the recorded data
(~2× rows historical), so it's the primary reference.
"""
from __future__ import annotations

import heapq
import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import duckdb

from hlanalysis.strategy.types import QuestionView

from ..core.data_source import DataSource, QuestionDescriptor
from ..core.events import (
    BookSnapshot,
    MarketEvent,
    ReferenceEvent,
    SettlementEvent,
    TradeEvent,
)
from ._hl_hip4_fastpath import FastPathBundle, build_fast_path_bundle

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_LEG_RE = re.compile(r"^#(\d+)$")
_HL_PREDICTION_PATH = "venue=hyperliquid/product_type=prediction_binary/mechanism=clob"
_HL_PERP_PATH = "venue=hyperliquid/product_type=perp/mechanism=clob"
_BINANCE_PERP_PATH = "venue=binance/product_type=perp/mechanism=clob"
# Map HL underlying → Binance perp symbol for the reference-price swap.
_BINANCE_REF_SYMBOL = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT"}

# Default reference-feed resampling period (60s = legacy behavior).
# Per-instance value lives on HLHip4DataSource and must match the strategy's
# vol_sampling_dt_seconds contract.
_DEFAULT_REFERENCE_RESAMPLE_NS = 60 * 1_000_000_000


def _resample_reference(
    inner: Iterator[tuple[int, ReferenceEvent]],
    *,
    resample_ns: int,
) -> Iterator[tuple[int, ReferenceEvent]]:
    """Aggregate consecutive ReferenceEvents into ``resample_ns`` OHLC bars.

    Why: the strategy's σ formula annualizes the last N returns assuming a
    fixed inter-sample spacing of ``vol_sampling_dt_seconds``. Raw HL
    BBO/mark feeds are far denser than that. Bucketing by floor(ts/Δ) and
    emitting one bar per bucket restores the assumed contract — high/low
    track the bucket extremes, close is the bucket's last tick, ts is the
    bucket's last-tick timestamp so monotone ordering is preserved.
    """
    cur_bucket: int | None = None
    h: float = 0.0
    l: float = 0.0
    c: float = 0.0
    last_ts: int = 0
    sym: str = "BTC"
    for ts, ev in inner:
        bucket = ts // resample_ns
        if cur_bucket is None:
            cur_bucket = bucket
            h, l, c = ev.high, ev.low, ev.close
            last_ts = ts
            sym = ev.symbol
        elif bucket != cur_bucket:
            yield last_ts, ReferenceEvent(last_ts, sym, h, l, c)
            cur_bucket = bucket
            h, l, c = ev.high, ev.low, ev.close
            last_ts = ts
            sym = ev.symbol
        else:
            if ev.high > h:
                h = ev.high
            if ev.low < l:
                l = ev.low
            c = ev.close
            last_ts = ts
    if cur_bucket is not None:
        yield last_ts, ReferenceEvent(last_ts, sym, h, l, c)


def _leg_outcome_side(symbol: str) -> tuple[int, int]:
    """`#220` → `(22, 0)` (YES of outcome 22). `#221` → `(22, 1)` (NO)."""
    m = _LEG_RE.match(symbol)
    if not m:
        raise ValueError(f"not a leg symbol: {symbol!r}")
    n = int(m.group(1))
    return n // 10, n % 10


def _leg_for(outcome_idx: int, side_idx: int) -> str:
    return f"#{outcome_idx * 10 + side_idx}"


def _parse_kv(keys: list[str], values: list[str]) -> dict[str, str]:
    return {str(k): str(v) for k, v in zip(keys or [], values or [])}


def _parse_description(desc: str) -> dict[str, str]:
    """``class:priceBucket|underlying:BTC|expiry:20260511-0600|priceThresholds:79043,82270|period:1d``
    -> dict.
    """
    return dict(p.split(":", 1) for p in (desc or "").split("|") if ":" in p)


def _expiry_ns(expiry_str: str) -> int:
    """`20260511-0600` -> ns since epoch (UTC)."""
    dt = datetime.strptime(expiry_str, "%Y%m%d-%H%M").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1e9)


def _iso_to_ns(s: str) -> int:
    """`'2026-05-10'` or `'2026-05-10T14:00:00Z'` -> ns since epoch."""
    if "T" not in s:
        s = s + "T00:00:00+00:00"
    elif s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return int(datetime.fromisoformat(s).timestamp() * 1e9)


def _sql_in(items) -> str:
    """Render a Python iterable of strings as comma-separated SQL string literals
    suitable for placement inside an ``IN (...)`` clause (no surrounding parens)."""
    return ",".join(repr(x) for x in items)


def _date_partitions_in_range(start_ns: int, end_ns: int) -> list[str]:
    """Return ``['YYYY-MM-DD', ...]`` covering the UTC days between start_ns and end_ns
    (inclusive of both edges, +1 day padding on the end side to catch tick-boundary spills).
    """
    start = datetime.fromtimestamp(start_ns / 1e9, tz=timezone.utc).date()
    end = datetime.fromtimestamp(end_ns / 1e9, tz=timezone.utc).date()
    out: list[str] = []
    d = start
    while d <= end + timedelta(days=1):
        out.append(d.isoformat())
        d += timedelta(days=1)
    return out


# ---------------------------------------------------------------------------
# DataSource
# ---------------------------------------------------------------------------


class HLHip4DataSource:
    """Recorded HL HIP-4 parquet → ``MarketEvent`` stream."""

    name = "hl_hip4"

    def __init__(
        self,
        data_root: Path | str = "data",
        *,
        ref_event: Literal["bbo", "mark"] = "bbo",
        ref_source: Literal["hl_perp", "binance_perp"] = "hl_perp",
        reference_resample_seconds: int = 60,
    ) -> None:
        self.data_root = Path(data_root)
        self.ref_event = ref_event
        # Reference-price venue. `hl_perp` (default) reads HL BBO/mark from
        # data/venue=hyperliquid/product_type=perp/...; `binance_perp` reads
        # Binance perp BBO from data/venue=binance/product_type=perp/...
        # The Binance feed leads HL by tens-of-ms to seconds on macro moves;
        # this lets us A/B whether feeding p_model (and σ) from Binance
        # changes the strategy's edges on HL HIP-4 markets.
        self.ref_source = ref_source
        # Reference resample period must match strategy.vol_sampling_dt_seconds —
        # the CLI threads this from the same param so train/serve stay coupled.
        if reference_resample_seconds <= 0:
            raise ValueError(
                f"reference_resample_seconds must be positive, got {reference_resample_seconds}"
            )
        self.reference_resample_seconds = int(reference_resample_seconds)
        self._reference_resample_ns = int(reference_resample_seconds) * 1_000_000_000
        # Cached per-instance: question_id -> parsed metadata bundle.
        self._meta_cache: dict[str, _QuestionMeta] = {}
        # Cached per-instance: question_id -> settled outcome. resolved_outcome
        # is deterministic per question (settlement data / final BTC ref don't
        # change within a process) but the runner calls it up to 3x per question
        # at settlement, each doing a duckdb settlement scan + BTC-ref lookup.
        self._outcome_cache: dict[str, Literal["yes", "no", "unknown"]] = {}

    # ------------------------------ discovery -----------------------------

    def discover(
        self,
        *,
        start: str,
        end: str,
        underlying: str = "BTC",
        kinds: tuple[str, ...] = ("priceBinary", "priceBucket"),
    ) -> list[QuestionDescriptor]:
        start_ns = _iso_to_ns(start)
        end_ns = _iso_to_ns(end)
        date_list = _date_partitions_in_range(start_ns - int(1e9 * 86400), end_ns + int(1e9 * 86400))

        glob = self._partition_glob("question_meta", symbol="Q*")
        con = duckdb.connect()
        df = con.sql(
            f"""
            SELECT symbol, question_idx, named_outcome_idxs, fallback_outcome_idx,
                   keys, values, exchange_ts
            FROM read_parquet('{glob}', hive_partitioning=1)
            WHERE date IN ({','.join(repr(d) for d in date_list)})
            ORDER BY symbol, exchange_ts
            """
        ).to_df()

        by_qid: dict[str, list[dict]] = {}
        for _, row in df.iterrows():
            by_qid.setdefault(str(row["symbol"]), []).append(row.to_dict())

        out: list[QuestionDescriptor] = []
        for qid, rows in by_qid.items():
            first = rows[0]
            kv = _parse_kv(list(first["keys"]), list(first["values"]))
            desc_fields = _parse_description(kv.get("question_description", ""))
            klass = desc_fields.get("class") or kv.get("class", "")
            qm_underlying = desc_fields.get("underlying") or kv.get("underlying", "")
            expiry_raw = desc_fields.get("expiry") or kv.get("expiry", "")
            if klass not in kinds:
                continue
            if qm_underlying != underlying:
                continue
            if not expiry_raw:
                continue
            try:
                expiry_ns = _expiry_ns(expiry_raw)
            except ValueError:
                continue
            if not (start_ns <= expiry_ns < end_ns):
                continue

            named_raw = first["named_outcome_idxs"]
            named = [int(x) for x in (named_raw if named_raw is not None else [])]
            fb_raw = first["fallback_outcome_idx"]
            try:
                fallback = int(fb_raw) if fb_raw is not None and str(fb_raw) != "<NA>" else None
            except (TypeError, ValueError):
                fallback = None
            leg_symbols: list[str] = []
            ordering: list[int] = list(named)
            if fallback is not None:
                ordering.append(fallback)
            for o in ordering:
                leg_symbols.extend([_leg_for(o, 0), _leg_for(o, 1)])

            start_ts_ns = int(first["exchange_ts"])
            out.append(
                QuestionDescriptor(
                    question_id=qid,
                    question_idx=int(first["question_idx"]),
                    start_ts_ns=start_ts_ns,
                    end_ts_ns=expiry_ns,
                    leg_symbols=tuple(leg_symbols),
                    klass=klass,
                    underlying=qm_underlying,
                )
            )

        out.sort(key=lambda q: q.start_ts_ns)
        return out

    # -------------------------------- events ------------------------------

    def events_arrays(self, q: QuestionDescriptor) -> FastPathBundle:
        """Arrow-backed fast path for the hftbacktest runner.

        Returns per-leg pre-built ``event_dtype`` numpy arrays plus
        ``ReferenceEvent`` / ``SettlementEvent`` lists, skipping the
        ``BookSnapshot`` / ``TradeEvent`` dataclass round-trip that ``events()``
        does. The runner detects this method (via ``getattr``) and routes
        around the legacy `_build_leg_event_array` path when it is present.

        Producing pre-built numpy arrays here lets the data source read
        parquet → Arrow → flat numpy columns once, then assemble the event
        array fully vectorised. On HL HIP-4 with 20-level books, this is
        ~4× faster than the dataclass path for the per-leg build alone and
        avoids creating ~160k ``BookSnapshot`` objects per leg per question.

        The assembled bundle is cached on disk keyed by source-file
        metadata (mtime + size) plus BUILD_VERSION from ``_fastpath_core``,
        so re-runs over the same data skip the DuckDB reads entirely.
        """
        from ._event_array_cache import cached_bundle

        source_files = self._fastpath_source_files(q)
        cache_dir = Path(self.data_root) / "_event_array_cache"

        def _build() -> FastPathBundle:
            date_list = _date_partitions_in_range(q.start_ts_ns, q.end_ts_ns)
            con = duckdb.connect()
            try:
                # Reference rows: same query as ``_reference_iter`` but kept as
                # raw fetchall tuples so the fast path can build the events list
                # without going through the gen() generator.
                evt = self.ref_event
                ref_rows = self._reference_rows(con, evt, q.start_ts_ns, q.end_ts_ns, date_list)
                if not ref_rows:
                    fallback = "mark" if evt == "bbo" else "bbo"
                    log.warning(
                        "HL perp BTC %s yielded 0 rows in [%d, %d); falling back to %s",
                        evt, q.start_ts_ns, q.end_ts_ns, fallback,
                    )
                    ref_rows = self._reference_rows(con, fallback, q.start_ts_ns, q.end_ts_ns, date_list)
                    evt = fallback
                return build_fast_path_bundle(
                    con=con,
                    q=q,
                    date_list=date_list,
                    book_glob_for=lambda leg: self._partition_glob("book_snapshot", symbol=leg),
                    trade_glob_for=lambda leg: self._partition_glob("trade", symbol=leg),
                    settlement_glob_for=lambda leg: self._partition_glob("settlement", symbol=leg),
                    reference_rows=ref_rows,
                    ref_event_kind=evt,
                    reference_resample_ns=self._reference_resample_ns,
                )
            finally:
                con.close()

        return cached_bundle(
            cache_dir,
            q.question_id,
            source_files,
            _build,
            force_rebuild=getattr(self, "_force_rebuild_cache", False),
            config_sig=self._bundle_config_sig(),
        )

    def _bundle_config_sig(self) -> str:
        """Cache-key signature: every non-source-file input that changes the
        built bundle. A dt or feed-kind change MUST alter this, or a dt=5 bundle
        aliases to a dt=60 request (the sigma-inflation footgun).

        ``ref_source`` is already keyed transitively (it changes the reference
        parquet paths in ``_fastpath_source_files``), but it is included here
        defensively so the contract holds even if that file-resolution is later
        refactored. Keep this in sync with the params that flow into ``_build``.
        """
        return (
            f"rrs={self._reference_resample_ns}"
            f"|ref={self.ref_event}"
            f"|refsrc={self.ref_source}"
        )

    def _fastpath_source_files(self, q: QuestionDescriptor) -> list[Path]:
        """Return the concrete parquet paths feeding the fast-path bundle.

        Expands the same globs that ``build_fast_path_bundle`` reads so the
        cache key tracks the right parquet files. Includes book_snapshot,
        trade, and settlement partitions for each leg, plus the reference
        (perp BBO/mark) partitions.
        """
        from glob import glob as _glob

        files: list[str] = []
        for leg in q.leg_symbols:
            for event in ("book_snapshot", "trade", "settlement"):
                g = self._partition_glob(event, symbol=leg)
                files.extend(_glob(g, recursive=True))
        # Reference feed (perp BBO primary, mark fallback).
        for evt in ("bbo", "mark"):
            g = self._perp_partition_glob(evt, symbol="BTC")
            files.extend(_glob(g, recursive=True))
        return [Path(f) for f in files]

    def events(self, q: QuestionDescriptor) -> Iterator[MarketEvent]:
        date_list = _date_partitions_in_range(q.start_ts_ns, q.end_ts_ns)
        # One shared duckdb connection per question. Previously each per-leg
        # iterator opened its own; that was 14+ connections for an 8-leg bucket.
        # Note: we tried batching all legs into a single ``read_parquet([...])``
        # union query but it was 3-5× slower than per-leg queries on this
        # workload — DuckDB reads each partition fast and avoids cross-leg
        # ORDER BY work; the per-leg-then-heapq.merge path is the winner.
        con = duckdb.connect()
        try:
            iters: list[Iterator[tuple[int, MarketEvent]]] = []
            for leg in q.leg_symbols:
                iters.append(self._book_iter(con, leg, q.start_ts_ns, q.end_ts_ns, date_list))
                iters.append(self._trade_iter(con, leg, q.start_ts_ns, q.end_ts_ns, date_list))
            iters.append(self._reference_iter(con, q.start_ts_ns, q.end_ts_ns, date_list))
            iters.append(self._settlement_iter(con, q, q.start_ts_ns, q.end_ts_ns, date_list))
            for _ts, ev in heapq.merge(*iters, key=lambda x: x[0]):
                yield ev
        finally:
            con.close()

    def _book_iter(
        self,
        con: "duckdb.DuckDBPyConnection",
        leg: str,
        start_ns: int,
        end_ns: int,
        date_list: list[str],
    ) -> Iterator[tuple[int, BookSnapshot]]:
        glob = self._partition_glob("book_snapshot", symbol=leg)
        if not self._partition_has_files(glob):
            return iter(())
        rows = con.sql(
            f"""
            SELECT exchange_ts, bid_px, bid_sz, ask_px, ask_sz
            FROM read_parquet('{glob}', hive_partitioning=1)
            WHERE date IN ({_sql_in(date_list)})
              AND exchange_ts >= {start_ns} AND exchange_ts < {end_ns}
            ORDER BY exchange_ts
            """
        ).fetchall()

        def gen() -> Iterator[tuple[int, BookSnapshot]]:
            for ts, bid_px, bid_sz, ask_px, ask_sz in rows:
                bids = tuple(zip(bid_px or [], bid_sz or [], strict=False))
                asks = tuple(zip(ask_px or [], ask_sz or [], strict=False))
                yield int(ts), BookSnapshot(int(ts), leg, bids, asks)

        return gen()

    def _trade_iter(
        self,
        con: "duckdb.DuckDBPyConnection",
        leg: str,
        start_ns: int,
        end_ns: int,
        date_list: list[str],
    ) -> Iterator[tuple[int, TradeEvent]]:
        glob = self._partition_glob("trade", symbol=leg)
        if not self._partition_has_files(glob):
            return iter(())
        rows = con.sql(
            f"""
            SELECT exchange_ts, price, size, side
            FROM read_parquet('{glob}', hive_partitioning=1)
            WHERE date IN ({_sql_in(date_list)})
              AND exchange_ts >= {start_ns} AND exchange_ts < {end_ns}
            ORDER BY exchange_ts
            """
        ).fetchall()

        def gen() -> Iterator[tuple[int, TradeEvent]]:
            for ts, price, size, side in rows:
                # Trade side is the aggressor; recorder writes "buy" / "sell" / "unknown".
                # Collapse "unknown" to "buy" so the type stays narrow; strategies do not
                # rely on aggressor for HL HIP-4.
                s = "buy" if side != "sell" else "sell"
                yield int(ts), TradeEvent(int(ts), leg, s, float(price), float(size))

        return gen()

    def _reference_iter(
        self,
        con: "duckdb.DuckDBPyConnection",
        start_ns: int,
        end_ns: int,
        date_list: list[str],
    ) -> Iterator[tuple[int, ReferenceEvent]]:
        evt = self.ref_event
        rows = self._reference_rows(con, evt, start_ns, end_ns, date_list)
        if not rows:
            fallback = "mark" if evt == "bbo" else "bbo"
            log.warning(
                "HL perp BTC %s yielded 0 rows in [%d, %d); falling back to %s",
                evt, start_ns, end_ns, fallback,
            )
            rows = self._reference_rows(con, fallback, start_ns, end_ns, date_list)
            evt = fallback
        if not rows:
            return iter(())

        # 2026-05-21: resample to OHLC bars of width vol_sampling_dt_seconds.
        # Why: the strategy's σ formula (`theta_harvester._sigma`) takes the LAST
        # `vol_lookback / vol_sampling_dt` returns and annualizes assuming each
        # return spans `vol_sampling_dt_seconds`. HL BTC perp BBO ticks ~6/s
        # and markPx ~1.2/s — without bucketing, those returns span 5-30s of
        # price action, but the annualization treats them as `vol_sampling_dt`
        # apart, giving a ~100-650× time-scale mismatch when dt=60s. Bucketing
        # by floor(ts/dt) restores the assumed contract. PM's source already
        # feeds 1m bars, so this brings HL into the same contract — and the
        # per-instance period lets us A/B sub-minute sampling without touching
        # call sites.
        resample_ns = self._reference_resample_ns
        if evt == "bbo":
            def gen_bbo_raw() -> Iterator[tuple[int, ReferenceEvent]]:
                for ts, bid, ask in rows:
                    mid = (float(bid) + float(ask)) / 2.0
                    yield int(ts), ReferenceEvent(int(ts), "BTC", mid, mid, mid)
            return _resample_reference(gen_bbo_raw(), resample_ns=resample_ns)

        def gen_mark_raw() -> Iterator[tuple[int, ReferenceEvent]]:
            for ts, px in rows:
                p = float(px)
                yield int(ts), ReferenceEvent(int(ts), "BTC", p, p, p)
        return _resample_reference(gen_mark_raw(), resample_ns=resample_ns)

    def _reference_rows(
        self,
        con: "duckdb.DuckDBPyConnection",
        evt: str,
        start_ns: int,
        end_ns: int,
        date_list: list[str],
    ) -> list[tuple]:
        glob = self._perp_partition_glob(evt, symbol="BTC")
        if not self._partition_has_files(glob):
            return []
        if evt == "bbo":
            sql = f"""
                SELECT exchange_ts, bid_px, ask_px
                FROM read_parquet('{glob}', hive_partitioning=1)
                WHERE date IN ({_sql_in(date_list)})
                  AND exchange_ts >= {start_ns} AND exchange_ts < {end_ns}
                ORDER BY exchange_ts
                """
        else:
            sql = f"""
                SELECT exchange_ts, mark_px
                FROM read_parquet('{glob}', hive_partitioning=1)
                WHERE date IN ({_sql_in(date_list)})
                  AND exchange_ts >= {start_ns} AND exchange_ts < {end_ns}
                ORDER BY exchange_ts
                """
        return con.sql(sql).fetchall()

    def _settlement_iter(
        self,
        con: "duckdb.DuckDBPyConnection",
        q: QuestionDescriptor,
        start_ns: int,
        end_ns: int,
        date_list: list[str],
    ) -> Iterator[tuple[int, SettlementEvent]]:
        # The recorder writes one settlement row per leg keyed by leg symbol; we read
        # all leg partitions for this question. settle_ts/exchange_ts is the settle
        # boundary. settled_side_idx=0 → YES paid; 1 → NO.
        rows_out: list[tuple[int, SettlementEvent]] = []
        for leg in q.leg_symbols:
            glob = self._partition_glob("settlement", symbol=leg)
            if not self._partition_has_files(glob):
                continue
            try:
                rs = con.sql(
                    f"""
                    SELECT exchange_ts, settle_ts, settled_side_idx
                    FROM read_parquet('{glob}', hive_partitioning=1)
                    WHERE date IN ({_sql_in(date_list)})
                      AND exchange_ts >= {start_ns} AND exchange_ts <= {end_ns}
                    ORDER BY exchange_ts
                    """
                ).fetchall()
            except duckdb.Error as e:
                log.warning("settlement read failed for %s: %s", leg, e)
                continue
            for ts, settle_ts, side_idx in rs:
                outcome: Literal["yes", "no", "unknown"] = (
                    "yes" if int(side_idx) == 0 else "no"
                )
                t = int(settle_ts or ts)
                rows_out.append((t, SettlementEvent(t, q.question_idx, outcome)))
        rows_out.sort(key=lambda x: x[0])
        return iter(rows_out)

    # ----------------------------- question view --------------------------

    def question_view(
        self,
        q: QuestionDescriptor,
        *,
        now_ns: int,
        settled: bool,
    ) -> QuestionView:
        meta = self._load_meta(q)
        kv = meta.kv
        period = kv.get("period", "1d")
        if q.klass == "priceBinary":
            yes_symbol = q.leg_symbols[0] if q.leg_symbols else ""
            no_symbol = q.leg_symbols[1] if len(q.leg_symbols) > 1 else ""
            strike = float(kv.get("targetPrice") or kv.get("strike") or 0.0)
        else:
            yes_symbol = ""
            no_symbol = ""
            thr_raw = kv.get("priceThresholds", "")
            strike_list = [float(t) for t in thr_raw.split(",") if t.strip()]
            strike = strike_list[0] if strike_list else 0.0

        settled_side: Literal["yes", "no", "unknown"] | None
        if settled:
            settled_side = self.resolved_outcome(q)
        else:
            settled_side = None

        return QuestionView(
            question_idx=q.question_idx,
            yes_symbol=yes_symbol,
            no_symbol=no_symbol,
            strike=strike,
            expiry_ns=q.end_ts_ns,
            underlying=q.underlying,
            klass=q.klass,
            period=period,
            settled=settled,
            settled_side=settled_side,
            leg_symbols=q.leg_symbols,
            name=meta.name,
            kv=tuple(meta.kv.items()),
        )

    # ---------------------------- resolved outcome ------------------------

    def resolved_outcome(
        self, q: QuestionDescriptor
    ) -> Literal["yes", "no", "unknown"]:
        cached = self._outcome_cache.get(q.question_id)
        if cached is not None:
            return cached
        outcome = self._resolve_outcome_impl(q)
        self._outcome_cache[q.question_id] = outcome
        return outcome

    def _resolve_outcome_impl(
        self, q: QuestionDescriptor
    ) -> Literal["yes", "no", "unknown"]:
        # 1) Try real settlement events. Per-leg short-circuit on first hit.
        date_list = _date_partitions_in_range(q.start_ts_ns, q.end_ts_ns)
        con = duckdb.connect()
        try:
            for leg in q.leg_symbols:
                glob = self._partition_glob("settlement", symbol=leg)
                if not self._partition_has_files(glob):
                    continue
                try:
                    row = con.sql(
                        f"""
                        SELECT settled_side_idx
                        FROM read_parquet('{glob}', hive_partitioning=1)
                        WHERE date IN ({_sql_in(date_list)})
                        ORDER BY exchange_ts
                        LIMIT 1
                        """
                    ).fetchone()
                except duckdb.Error:
                    row = None
                if row is not None:
                    return "yes" if int(row[0]) == 0 else "no"

            # 2) Fallback: infer for binaries from final HL perp BTC mark/mid.
            # Reuse the same connection rather than opening a new one inside
            # ``_last_btc_ref_at_or_before``.
            if q.klass == "priceBinary":
                last_btc = self._last_btc_ref_at_or_before(q.end_ts_ns, con=con)
                if last_btc is None:
                    return "unknown"
                meta = self._load_meta(q)
                strike = float(meta.kv.get("targetPrice") or meta.kv.get("strike") or 0.0)
                if not strike:
                    return "unknown"
                return "yes" if last_btc > strike else "no"
        finally:
            con.close()

        # 3) Bucket without settlement → unknown (per-leg outcome is what runners care about).
        return "unknown"

    def leg_payoff(self, q: QuestionDescriptor, leg_symbol: str) -> float:
        """Per-leg payoff at settlement: 1.0 if the leg won, 0.0 otherwise.

        Binary: delegates to ``resolved_outcome`` and matches against the
        canonical (yes=leg0, no=leg1) layout.

        Bucket: uses final HL perp BTC and the question's ``priceThresholds``
        to determine which bucket BTC ended in, then pays:
          - the YES leg of the winning bucket → 1.0
          - the NO leg of every non-winning *named* bucket → 1.0
          - everything else → 0.0
        Falls back to a market-implied winner (the leg whose bid was ≥ 0.95
        within the last minute pre-expiry) if no priceThresholds metadata or
        no BTC reference is available.
        """
        if q.klass == "priceBinary":
            outcome = self.resolved_outcome(q)
            if outcome == "yes" and q.leg_symbols and leg_symbol == q.leg_symbols[0]:
                return 1.0
            if outcome == "no" and len(q.leg_symbols) > 1 and leg_symbol == q.leg_symbols[1]:
                return 1.0
            return 0.0

        if q.klass != "priceBucket" or leg_symbol not in q.leg_symbols:
            return 0.0

        # Bucket: figure out the winning outcome_pos from BTC + thresholds.
        meta = self._load_meta(q)
        thr_raw = meta.kv.get("priceThresholds", "")
        thr = [float(t) for t in thr_raw.split(",") if t.strip()]
        last_btc = self._last_btc_ref_at_or_before(q.end_ts_ns)
        winning_pos: int | None = None
        if thr and last_btc is not None:
            if last_btc <= thr[0]:
                winning_pos = 0
            elif last_btc > thr[-1]:
                winning_pos = len(thr)
            else:
                for i in range(1, len(thr)):
                    if thr[i - 1] < last_btc <= thr[i]:
                        winning_pos = i
                        break

        # Fallback: pick the leg whose bid was nearest 1.0 just before expiry.
        if winning_pos is None:
            winning_pos = self._winning_pos_from_books(q)

        if winning_pos is None:
            return 0.0

        idx = q.leg_symbols.index(leg_symbol)
        held_pos = idx // 2
        held_side = idx % 2  # 0 = YES, 1 = NO
        n_named_buckets = max(1, len(q.leg_symbols) // 2 - 1)  # excludes fallback leg
        # Held leg's outcome_pos may exceed n_named_buckets when it's the
        # fallback bucket — treat it as "never wins" (HL fallback doesn't
        # correspond to a price-range outcome).
        if held_pos >= n_named_buckets:
            return 0.0
        if held_side == 0:
            return 1.0 if held_pos == winning_pos else 0.0
        # NO leg: pays out for every non-winning *named* bucket.
        return 1.0 if held_pos != winning_pos else 0.0

    def _winning_pos_from_books(self, q: QuestionDescriptor) -> int | None:
        """Last-resort: scan each named YES leg's final book and return the
        outcome_pos whose bid was ≥ 0.95 within 60s of expiry."""
        end_ns = q.end_ts_ns
        date_list = _date_partitions_in_range(end_ns - int(86400 * 1e9), end_ns)
        n_named_buckets = max(1, len(q.leg_symbols) // 2 - 1)
        for pos in range(n_named_buckets):
            yes_leg = q.leg_symbols[2 * pos]
            glob = self._partition_glob("book_snapshot", symbol=yes_leg)
            if not self._partition_has_files(glob):
                continue
            con = duckdb.connect()
            try:
                row = con.sql(
                    f"""
                    SELECT bid_px FROM read_parquet('{glob}', hive_partitioning=1)
                    WHERE date IN ({','.join(repr(d) for d in date_list)})
                      AND exchange_ts <= {end_ns}
                      AND exchange_ts > {end_ns - 60 * int(1e9)}
                      AND bid_px IS NOT NULL AND len(bid_px) > 0
                    ORDER BY exchange_ts DESC LIMIT 1
                    """
                ).fetchone()
            except duckdb.Error:
                row = None
            if row is None or row[0] is None or len(row[0]) == 0:
                continue
            if float(row[0][0]) >= 0.95:
                return pos
        return None

    # -------------------------------- helpers -----------------------------

    def _load_meta(self, q: QuestionDescriptor) -> "_QuestionMeta":
        if q.question_id in self._meta_cache:
            return self._meta_cache[q.question_id]
        # question_meta carries question-level kv; market_meta carries per-leg kv we
        # fold in. The values are dicts of str.
        date_list = _date_partitions_in_range(q.start_ts_ns, q.end_ts_ns)
        con = duckdb.connect()
        glob = self._partition_glob("question_meta", symbol=q.question_id)
        try:
            qm = con.sql(
                f"""
                SELECT keys, values
                FROM read_parquet('{glob}', hive_partitioning=1)
                WHERE date IN ({','.join(repr(d) for d in date_list)})
                ORDER BY exchange_ts
                LIMIT 1
                """
            ).fetchone()
        except duckdb.Error:
            qm = None
        question_kv: dict[str, str] = {}
        if qm is not None:
            question_kv = _parse_kv(list(qm[0] or []), list(qm[1] or []))
            desc_fields = _parse_description(question_kv.get("question_description", ""))
            for k, v in desc_fields.items():
                question_kv.setdefault(k, v)
        name = question_kv.get("question_name", "")
        meta = _QuestionMeta(name=name, kv=question_kv)
        self._meta_cache[q.question_id] = meta
        return meta

    def _last_btc_ref_at_or_before(self, ts_ns: int, con: "duckdb.DuckDBPyConnection | None" = None) -> float | None:
        date_list = _date_partitions_in_range(ts_ns - int(2 * 86400 * 1e9), ts_ns)
        owned = con is None
        if owned:
            con = duckdb.connect()
        try:
            return self._last_btc_ref_impl(con, ts_ns, date_list)
        finally:
            if owned:
                con.close()

    def _last_btc_ref_impl(
        self,
        con: "duckdb.DuckDBPyConnection",
        ts_ns: int,
        date_list: list[str],
    ) -> float | None:
        for evt in ("bbo", "mark"):
            glob = self._perp_partition_glob(evt, symbol="BTC")
            if not self._partition_has_files(glob):
                continue
            if evt == "bbo":
                q = f"""
                    SELECT (bid_px + ask_px)/2.0 AS mid
                    FROM read_parquet('{glob}', hive_partitioning=1)
                    WHERE date IN ({_sql_in(date_list)})
                      AND exchange_ts <= {ts_ns}
                    ORDER BY exchange_ts DESC LIMIT 1
                """
            else:
                q = f"""
                    SELECT mark_px AS mid
                    FROM read_parquet('{glob}', hive_partitioning=1)
                    WHERE date IN ({_sql_in(date_list)})
                      AND exchange_ts <= {ts_ns}
                    ORDER BY exchange_ts DESC LIMIT 1
                """
            try:
                row = con.sql(q).fetchone()
            except duckdb.Error:
                row = None
            if row is not None:
                return float(row[0])
        return None

    def _partition_glob(self, event: str, *, symbol: str) -> str:
        # Hive-partitioned scan path. `symbol` may include `*` (e.g. `Q*`).
        return str(
            self.data_root
            / _HL_PREDICTION_PATH
            / f"event={event}"
            / f"symbol={symbol}"
            / "**" / "*.parquet"
        )

    def _perp_partition_glob(self, event: str, *, symbol: str) -> str:
        if self.ref_source == "binance_perp":
            return str(
                self.data_root
                / _BINANCE_PERP_PATH
                / f"event={event}"
                / f"symbol={_BINANCE_REF_SYMBOL.get(symbol, symbol)}"
                / "**" / "*.parquet"
            )
        return str(
            self.data_root
            / _HL_PERP_PATH
            / f"event={event}"
            / f"symbol={symbol}"
            / "**" / "*.parquet"
        )

    @lru_cache(maxsize=4096)
    def _partition_has_files(self, glob: str) -> bool:
        # Cheap existence check that avoids duckdb raising on missing paths.
        from glob import glob as _glob
        return bool(_glob(glob, recursive=True))


@dataclass(frozen=True, slots=True)
class _QuestionMeta:
    name: str
    kv: dict[str, str]


__all__ = [
    "HLHip4DataSource",
    "QuestionDescriptor",
    "BookSnapshot",
    "TradeEvent",
    "ReferenceEvent",
    "SettlementEvent",
    "MarketEvent",
    "DataSource",
]
