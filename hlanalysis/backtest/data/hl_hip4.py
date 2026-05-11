"""HL HIP-4 recorded-data `DataSource`.

Reads the recorder's parquet partitions under
``data/venue=hyperliquid/product_type=prediction_binary/mechanism=clob/event=*/``
and emits the §3 ``MarketEvent`` union in monotone-``ts_ns`` order.

Reference-price source: HL perp BTC BBO mid (falls back to perp ``mark`` if BBO
is empty over the requested window). HIP-4 binaries settle off the HL perp mark,
so perp is the correct underlying. BBO is denser than mark in the recorded data
(~2× rows historical), so it's the primary reference.

The §3.1/§3.2 dataclasses + ``DataSource`` Protocol live inline in this module
as a temporary mirror until task A's ``hlanalysis/backtest/core/`` PR lands. The
integration pass (task E) drops the mirror and re-points imports.
TODO(task-E): drop ``_LocalMirror`` block and import from ``hlanalysis.backtest.core``.
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
from typing import Any, Literal, Protocol, Union

import duckdb

from hlanalysis.strategy.types import QuestionView

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Local mirror of the §3 contract types. Drop once task A merges.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BookSnapshot:
    ts_ns: int
    symbol: str
    bids: tuple[tuple[float, float], ...]
    asks: tuple[tuple[float, float], ...]


@dataclass(frozen=True, slots=True)
class TradeEvent:
    ts_ns: int
    symbol: str
    side: Literal["buy", "sell"]
    price: float
    size: float


@dataclass(frozen=True, slots=True)
class ReferenceEvent:
    ts_ns: int
    symbol: str
    high: float
    low: float
    close: float


@dataclass(frozen=True, slots=True)
class SettlementEvent:
    ts_ns: int
    question_idx: int
    outcome: Literal["yes", "no", "unknown"]


MarketEvent = Union[BookSnapshot, TradeEvent, ReferenceEvent, SettlementEvent]


@dataclass(frozen=True, slots=True)
class QuestionDescriptor:
    question_id: str
    question_idx: int
    start_ts_ns: int
    end_ts_ns: int
    leg_symbols: tuple[str, ...]
    klass: str
    underlying: str


class DataSource(Protocol):
    name: str

    def discover(self, *, start: str, end: str, **filters: Any) -> list[QuestionDescriptor]: ...
    def events(self, q: QuestionDescriptor) -> Iterator[MarketEvent]: ...
    def question_view(self, q: QuestionDescriptor, *, now_ns: int, settled: bool) -> QuestionView: ...
    def resolved_outcome(self, q: QuestionDescriptor) -> Literal["yes", "no", "unknown"]: ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_LEG_RE = re.compile(r"^#(\d+)$")
_HL_PREDICTION_PATH = "venue=hyperliquid/product_type=prediction_binary/mechanism=clob"
_HL_PERP_PATH = "venue=hyperliquid/product_type=perp/mechanism=clob"


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
    ) -> None:
        self.data_root = Path(data_root)
        self.ref_event = ref_event
        # Cached per-instance: question_id -> parsed metadata bundle.
        self._meta_cache: dict[str, _QuestionMeta] = {}

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

    def events(self, q: QuestionDescriptor) -> Iterator[MarketEvent]:
        date_list = _date_partitions_in_range(q.start_ts_ns, q.end_ts_ns)
        iters: list[Iterator[tuple[int, MarketEvent]]] = []
        for leg in q.leg_symbols:
            iters.append(self._book_iter(leg, q.start_ts_ns, q.end_ts_ns, date_list))
            iters.append(self._trade_iter(leg, q.start_ts_ns, q.end_ts_ns, date_list))
        iters.append(self._reference_iter(q.start_ts_ns, q.end_ts_ns, date_list))
        iters.append(self._settlement_iter(q, q.start_ts_ns, q.end_ts_ns, date_list))
        for _ts, ev in heapq.merge(*iters, key=lambda x: x[0]):
            yield ev

    def _book_iter(
        self, leg: str, start_ns: int, end_ns: int, date_list: list[str]
    ) -> Iterator[tuple[int, BookSnapshot]]:
        glob = self._partition_glob("book_snapshot", symbol=leg)
        if not self._partition_has_files(glob):
            return iter(())
        con = duckdb.connect()
        result = con.sql(
            f"""
            SELECT exchange_ts, bid_px, bid_sz, ask_px, ask_sz
            FROM read_parquet('{glob}', hive_partitioning=1)
            WHERE date IN ({','.join(repr(d) for d in date_list)})
              AND exchange_ts >= {start_ns} AND exchange_ts < {end_ns}
            ORDER BY exchange_ts
            """
        )

        def gen() -> Iterator[tuple[int, BookSnapshot]]:
            for ts, bid_px, bid_sz, ask_px, ask_sz in result.fetchall():
                bids = tuple(zip(bid_px or [], bid_sz or [], strict=False))
                asks = tuple(zip(ask_px or [], ask_sz or [], strict=False))
                yield int(ts), BookSnapshot(int(ts), leg, bids, asks)

        return gen()

    def _trade_iter(
        self, leg: str, start_ns: int, end_ns: int, date_list: list[str]
    ) -> Iterator[tuple[int, TradeEvent]]:
        glob = self._partition_glob("trade", symbol=leg)
        if not self._partition_has_files(glob):
            return iter(())
        con = duckdb.connect()
        result = con.sql(
            f"""
            SELECT exchange_ts, price, size, side
            FROM read_parquet('{glob}', hive_partitioning=1)
            WHERE date IN ({','.join(repr(d) for d in date_list)})
              AND exchange_ts >= {start_ns} AND exchange_ts < {end_ns}
            ORDER BY exchange_ts
            """
        )

        def gen() -> Iterator[tuple[int, TradeEvent]]:
            for ts, price, size, side in result.fetchall():
                # Trade side is the aggressor; recorder writes "buy" / "sell" / "unknown".
                # Collapse "unknown" to "buy" so the type stays narrow; strategies do not
                # rely on aggressor for HL HIP-4.
                s = "buy" if side != "sell" else "sell"
                yield int(ts), TradeEvent(int(ts), leg, s, float(price), float(size))

        return gen()

    def _reference_iter(
        self, start_ns: int, end_ns: int, date_list: list[str]
    ) -> Iterator[tuple[int, ReferenceEvent]]:
        evt = self.ref_event
        glob = self._perp_partition_glob(evt, symbol="BTC")
        primary_iter = self._reference_iter_for(evt, glob, start_ns, end_ns, date_list)
        # We must materialise to check emptiness without losing the first elem; the
        # captured windows are small (~hundreds of rows) so this is cheap.
        rows = list(primary_iter)
        if rows:
            return iter(rows)
        # Fallback: try the other event type. Logs and proceeds.
        fallback = "mark" if evt == "bbo" else "bbo"
        log.warning(
            "HL perp BTC %s yielded 0 rows in [%d, %d); falling back to %s",
            evt, start_ns, end_ns, fallback,
        )
        glob2 = self._perp_partition_glob(fallback, symbol="BTC")
        return self._reference_iter_for(fallback, glob2, start_ns, end_ns, date_list)

    def _reference_iter_for(
        self,
        evt: str,
        glob: str,
        start_ns: int,
        end_ns: int,
        date_list: list[str],
    ) -> Iterator[tuple[int, ReferenceEvent]]:
        if not self._partition_has_files(glob):
            return iter(())
        con = duckdb.connect()
        if evt == "bbo":
            result = con.sql(
                f"""
                SELECT exchange_ts, bid_px, ask_px
                FROM read_parquet('{glob}', hive_partitioning=1)
                WHERE date IN ({','.join(repr(d) for d in date_list)})
                  AND exchange_ts >= {start_ns} AND exchange_ts < {end_ns}
                ORDER BY exchange_ts
                """
            )

            def gen() -> Iterator[tuple[int, ReferenceEvent]]:
                for ts, bid, ask in result.fetchall():
                    mid = (float(bid) + float(ask)) / 2.0
                    yield int(ts), ReferenceEvent(int(ts), "BTC", mid, mid, mid)

            return gen()
        # mark
        result = con.sql(
            f"""
            SELECT exchange_ts, mark_px
            FROM read_parquet('{glob}', hive_partitioning=1)
            WHERE date IN ({','.join(repr(d) for d in date_list)})
              AND exchange_ts >= {start_ns} AND exchange_ts < {end_ns}
            ORDER BY exchange_ts
            """
        )

        def gen() -> Iterator[tuple[int, ReferenceEvent]]:
            for ts, px in result.fetchall():
                p = float(px)
                yield int(ts), ReferenceEvent(int(ts), "BTC", p, p, p)

        return gen()

    def _settlement_iter(
        self,
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
            con = duckdb.connect()
            try:
                rs = con.sql(
                    f"""
                    SELECT exchange_ts, settle_ts, settled_side_idx
                    FROM read_parquet('{glob}', hive_partitioning=1)
                    WHERE date IN ({','.join(repr(d) for d in date_list)})
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
                rows_out.append(
                    (t, SettlementEvent(t, q.question_idx, outcome))
                )
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
        # 1) Try real settlement events.
        date_list = _date_partitions_in_range(q.start_ts_ns, q.end_ts_ns)
        for leg in q.leg_symbols:
            glob = self._partition_glob("settlement", symbol=leg)
            if not self._partition_has_files(glob):
                continue
            con = duckdb.connect()
            try:
                row = con.sql(
                    f"""
                    SELECT settled_side_idx
                    FROM read_parquet('{glob}', hive_partitioning=1)
                    WHERE date IN ({','.join(repr(d) for d in date_list)})
                    ORDER BY exchange_ts
                    LIMIT 1
                    """
                ).fetchone()
            except duckdb.Error:
                row = None
            if row is not None:
                return "yes" if int(row[0]) == 0 else "no"

        # 2) Fallback: infer for binaries from final HL perp BTC mark/mid.
        if q.klass == "priceBinary":
            last_btc = self._last_btc_ref_at_or_before(q.end_ts_ns)
            if last_btc is None:
                return "unknown"
            meta = self._load_meta(q)
            strike = float(meta.kv.get("targetPrice") or meta.kv.get("strike") or 0.0)
            if not strike:
                return "unknown"
            return "yes" if last_btc > strike else "no"

        # 3) Bucket without settlement → unknown (per-leg outcome is what runners care about).
        return "unknown"

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

    def _last_btc_ref_at_or_before(self, ts_ns: int) -> float | None:
        date_list = _date_partitions_in_range(ts_ns - int(2 * 86400 * 1e9), ts_ns)
        for evt in ("bbo", "mark"):
            glob = self._perp_partition_glob(evt, symbol="BTC")
            if not self._partition_has_files(glob):
                continue
            con = duckdb.connect()
            if evt == "bbo":
                q = f"""
                    SELECT (bid_px + ask_px)/2.0 AS mid
                    FROM read_parquet('{glob}', hive_partitioning=1)
                    WHERE date IN ({','.join(repr(d) for d in date_list)})
                      AND exchange_ts <= {ts_ns}
                    ORDER BY exchange_ts DESC LIMIT 1
                """
            else:
                q = f"""
                    SELECT mark_px AS mid
                    FROM read_parquet('{glob}', hive_partitioning=1)
                    WHERE date IN ({','.join(repr(d) for d in date_list)})
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
