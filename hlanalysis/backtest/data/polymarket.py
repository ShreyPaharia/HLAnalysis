"""Polymarket data source — binary (BTC Up/Down daily) and bucket
(Bitcoin Multi Strikes Hourly) markets.

Implements the §3.2 `DataSource` protocol against a local cache populated by
`fetch_and_cache(...)`. The cache layout is a single `manifest.json` keyed by
question_id (PM `condition_id` for binaries, PM event slug for buckets) plus
per-leg trade parquet files under `pm_trades/` and 1m BTC klines under
`btc_klines/*.json`. See `docs/specs/2026-05-11-task-c-plan.md` §3 for the
on-disk shape.

The synthetic L2 model (one snapshot per PM trade, single level at ±half_spread)
is preserved from the previous `sim/hftbt_adapter.build_event_stream`. Within
each (YES, NO) leg pair we infer the complementary leg's price as `1 − p` per
the PM CLOB parity identity; we do NOT cross-infer across strike-pairs in
bucket markets (those are independent above-X binaries).
"""
from __future__ import annotations

import heapq
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Literal

import pyarrow.parquet as pq
import requests
from loguru import logger

from hlanalysis.strategy.types import QuestionView

from ..core.data_source import QuestionDescriptor
from ..core.events import (
    BookSnapshot,
    MarketEvent,
    ReferenceEvent,
    SettlementEvent,
    TradeEvent,
)
from ._synthetic_l2 import L2Snapshot, trade_to_l2

_GAMMA_BASE = "https://gamma-api.polymarket.com"
_CLOB_DATA_BASE = "https://data-api.polymarket.com"
_CLOB_BASE = "https://clob.polymarket.com"
_BTC_UPDOWN_SERIES_SLUG = "btc-up-or-down-daily"
_BTC_BUCKET_SERIES_SLUG = "bitcoin-multi-strikes-hourly"
_SERIES_PAGE_LIMIT = 100  # Gamma /events caps responses at 100 even when limit > 100;
# requesting 500 silently truncates and we mistake the short response for "end of data".
_TRADES_PAGE_SIZE = 500
_HALF_SPREAD_DEFAULT = 0.005
_DEPTH_DEFAULT = 10_000.0
_P_CLIP_LO = 1e-6
_P_CLIP_HI = 1.0 - 1e-6


# ---- HTTP helpers (also used by fetch_and_cache) ----------------------------


def _http_get(url: str, params: dict | None = None) -> dict | list:
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def _parse_iso_ns(s: str) -> int:
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1e9)


def _event_in_window(ev: dict, start_iso: str, end_iso: str) -> bool:
    end = ev.get("endDate") or ""
    if len(end) < 10:
        return False
    return start_iso <= end[:10] < end_iso


def _fetch_series_events(series_slug: str) -> list[dict]:
    """Paginate Gamma `/events?series_slug=...&closed=true`."""
    out: list[dict] = []
    offset = 0
    while True:
        try:
            page = _http_get(
                f"{_GAMMA_BASE}/events",
                params={
                    "series_slug": series_slug,
                    "closed": "true",
                    "limit": _SERIES_PAGE_LIMIT,
                    "offset": offset,
                },
            )
        except Exception as e:
            logger.warning(f"PM series fetch failed at offset={offset}: {e}")
            break
        if not isinstance(page, list) or not page:
            break
        out.extend(page)
        if len(page) < _SERIES_PAGE_LIMIT:
            break
        offset += len(page)
    return out


def _fetch_trades_raw(condition_id: str) -> list[dict]:
    """Page PM data-api `/trades?market=<conditionId>`."""
    out: list[dict] = []
    offset = 0
    while True:
        try:
            page = _http_get(
                f"{_CLOB_DATA_BASE}/trades",
                params={"market": condition_id, "limit": _TRADES_PAGE_SIZE, "offset": offset},
            )
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 400 and offset > 0:
                logger.warning(
                    f"PM trades pagination capped at offset={offset} for {condition_id}; "
                    f"returning partial result."
                )
                break
            raise
        if not page:
            break
        out.extend(page)
        if len(page) < _TRADES_PAGE_SIZE:
            break
        offset += _TRADES_PAGE_SIZE
    return out


_BTC_UPDOWN_STRIKE_RULE = re.compile(
    r"Binance 1 minute candle for BTC/USDT\s+(\w+)\s+(\d+)\s+'(\d{2})\s+(\d{1,2}):(\d{2})\s+in the ET timezone",
    re.IGNORECASE,
)
_MONTHS = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
           "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}


def _parse_strike_ref_ts_ns(description: str) -> int | None:
    """Pull the strike-reference timestamp out of a PM 'BTC Up or Down' market
    description. The text says, e.g., "...Binance 1 minute candle for BTC/USDT
    Nov 27 '25 12:00 in the ET timezone...". The strike is the CLOSE of that
    candle. Returns ns since epoch (UTC) or None if the description doesn't
    match the expected pattern.
    """
    if not description:
        return None
    m = _BTC_UPDOWN_STRIKE_RULE.search(description)
    if not m:
        return None
    try:
        from zoneinfo import ZoneInfo
        et = ZoneInfo("America/New_York")
    except Exception:
        return None
    mon, day, yr2, hh, mm = m.groups()
    if mon not in _MONTHS:
        return None
    year = 2000 + int(yr2)
    dt = datetime(year, _MONTHS[mon], int(day), int(hh), int(mm), tzinfo=et)
    return int(dt.astimezone(timezone.utc).timestamp() * 1e9)


def _parse_binary_event(ev: dict) -> dict | None:
    """Parse a Gamma binary event into a manifest-shaped record. Returns None
    if it doesn't look like a single-market binary."""
    markets = ev.get("markets") or []
    if not markets:
        return None
    mk = markets[0]
    token_ids_raw = mk.get("clobTokenIds")
    if not token_ids_raw:
        return None
    token_ids = json.loads(token_ids_raw) if isinstance(token_ids_raw, str) else token_ids_raw
    if len(token_ids) != 2:
        return None
    cond_id = mk.get("conditionId") or mk.get("id")
    if not cond_id:
        return None
    start_iso = mk.get("startDate") or ev.get("startDate")
    end_iso = mk.get("endDate") or ev.get("endDate")
    if not (start_iso and end_iso):
        return None
    outcome: Literal["yes", "no", "unknown"] = "unknown"
    op = mk.get("outcomePrices")
    if op:
        prices = json.loads(op) if isinstance(op, str) else op
        if len(prices) == 2:
            yes_p, no_p = float(prices[0]), float(prices[1])
            if yes_p >= 0.99:
                outcome = "yes"
            elif no_p >= 0.99:
                outcome = "no"
    description = mk.get("description") or ev.get("description") or ""
    strike_ref_ts_ns = _parse_strike_ref_ts_ns(description)
    out: dict = {
        "condition_id": str(cond_id),
        "yes_token_id": str(token_ids[0]),
        "no_token_id": str(token_ids[1]),
        "start_ts_ns": _parse_iso_ns(start_iso),
        "end_ts_ns": _parse_iso_ns(end_iso),
        "resolved_outcome": outcome,
        "total_volume_usd": float(mk.get("volume") or 0.0),
        "n_trades": 0,
    }
    if strike_ref_ts_ns is not None:
        out["strike_ref_ts_ns"] = strike_ref_ts_ns
    return out


def _parse_bucket_event(ev: dict) -> dict | None:
    """Parse a Gamma multi-strike event into a bucket manifest record.

    Each sub-market is a binary "BTC above $X" with its own conditionId + (YES,
    NO) clob token pair. We sort sub-markets by strike ascending so leg_tokens
    + thresholds align positionally with the canonical
    (yes_o0, no_o0, yes_o1, no_o1, ...) layout.
    """
    markets = ev.get("markets") or []
    if len(markets) < 2:
        return None
    slug = ev.get("slug") or ev.get("ticker")
    end_iso = ev.get("endDate")
    start_iso = ev.get("startDate")
    if not (slug and start_iso and end_iso):
        return None
    legs: list[tuple[float, str, str, str, str]] = []  # (strike, cond_id, yes_tok, no_tok, res)
    for mk in markets:
        tok_raw = mk.get("clobTokenIds")
        if not tok_raw:
            continue
        tok = json.loads(tok_raw) if isinstance(tok_raw, str) else tok_raw
        if len(tok) != 2:
            continue
        cond_id = mk.get("conditionId") or mk.get("id")
        strike_raw = mk.get("groupItemTitle") or ""
        try:
            strike = float(str(strike_raw).replace(",", ""))
        except ValueError:
            continue
        outcome: Literal["yes", "no", "unknown"] = "unknown"
        op = mk.get("outcomePrices")
        if op:
            prices = json.loads(op) if isinstance(op, str) else op
            if len(prices) == 2:
                yes_p, no_p = float(prices[0]), float(prices[1])
                if yes_p >= 0.99:
                    outcome = "yes"
                elif no_p >= 0.99:
                    outcome = "no"
        legs.append((strike, str(cond_id), str(tok[0]), str(tok[1]), outcome))
    if not legs:
        return None
    legs.sort(key=lambda x: x[0])
    return {
        "event_slug": str(slug),
        "start_ts_ns": _parse_iso_ns(start_iso),
        "end_ts_ns": _parse_iso_ns(end_iso),
        "thresholds": [float(l[0]) for l in legs],
        "leg_tokens": [[l[2], l[3]] for l in legs],
        "leg_condition_ids": [l[1] for l in legs],
        "leg_resolutions": [l[4] for l in legs],
    }


# ---- DataSource implementation ---------------------------------------------


def _question_idx(question_id: str) -> int:
    return hash(question_id) & 0x7FFFFFFF


@dataclass(frozen=True, slots=True)
class _StreamCfg:
    half_spread: float
    depth: float


class PolymarketDataSource:
    """Cache-driven PM data source. Discovery returns descriptors built from
    the on-disk manifest; `events()` reads cached parquet / json and yields
    `MarketEvent`s in monotone non-decreasing `ts_ns`.

    Use `fetch_and_cache(...)` to populate the cache from the live Gamma +
    CLOB APIs; tests bypass it by writing directly into the cache root.
    """

    name = "polymarket"

    def __init__(
        self,
        *,
        cache_root: Path,
        half_spread: float = _HALF_SPREAD_DEFAULT,
        depth: float = _DEPTH_DEFAULT,
    ) -> None:
        self._cache_root = Path(cache_root)
        self._stream_cfg = _StreamCfg(half_spread=half_spread, depth=depth)
        # Lazy caches populated on first read. Significant for tuning workers
        # that backtest dozens of markets per cell — without caches each
        # market would re-parse the manifest + the (large) BTC klines JSON.
        self._manifest_cache: dict | None = None
        self._klines_cache: list[dict] | None = None

    # -- public API --------------------------------------------------------

    def discover(
        self,
        *,
        start: str,
        end: str,
        kind: Literal["binary", "bucket", "both"] = "both",
        **_filters: object,
    ) -> list[QuestionDescriptor]:
        """Return descriptors for cached questions with `endDate` in [start, end).

        `start` / `end` are ISO date strings (`YYYY-MM-DD`). Filtering uses the
        cache manifest's stored `end_ts_ns` — convert to ISO and compare.
        Wire-side discovery (Gamma API) lives in `fetch_and_cache`.
        """
        manifest = self._load_manifest()
        out: list[QuestionDescriptor] = []
        for qid, entry in manifest.items():
            entry_kind = entry.get("kind", "binary")
            if kind != "both" and entry_kind != kind:
                continue
            if entry_kind == "binary":
                d = self._descriptor_from_binary(qid, entry)
            elif entry_kind == "bucket":
                d = self._descriptor_from_bucket(qid, entry)
            else:
                continue
            if d is None:
                continue
            if not _ts_ns_in_iso_window(d.end_ts_ns, start, end):
                continue
            out.append(d)
        out.sort(key=lambda d: d.start_ts_ns)
        return out

    def events(self, q: QuestionDescriptor) -> Iterator[MarketEvent]:
        manifest = self._load_manifest()
        entry = manifest.get(q.question_id)
        if entry is None:
            return iter(())
        if entry.get("kind", "binary") == "binary":
            return self._events_binary(q, entry)
        return self._events_bucket(q, entry)

    def question_view(
        self, q: QuestionDescriptor, *, now_ns: int, settled: bool,
    ) -> QuestionView:
        manifest = self._load_manifest()
        entry = manifest.get(q.question_id) or {}
        kind = entry.get("kind", "binary")
        if kind == "binary":
            return self._question_view_binary(q, entry, now_ns=now_ns, settled=settled)
        return self._question_view_bucket(q, entry, now_ns=now_ns, settled=settled)

    def resolved_outcome(self, q: QuestionDescriptor) -> Literal["yes", "no", "unknown"]:
        manifest = self._load_manifest()
        entry = manifest.get(q.question_id) or {}
        kind = entry.get("kind", "binary")
        if kind == "binary":
            mk = entry.get("market") or {}
            return mk.get("resolved_outcome", "unknown")  # type: ignore[return-value]
        # Bucket: outcome aggregates the per-leg resolutions; we report
        # "unknown" because there's no single bucket outcome. Per-leg outcomes
        # are emitted via `SettlementEvent.symbol` in events().
        return "unknown"

    # -- discovery + cache population from the live API --------------------

    def fetch_and_cache(
        self,
        *,
        start: str,
        end: str,
        kind: Literal["binary", "bucket", "both"] = "both",
        min_trades: int = 0,
        min_volume_usd: float = 0.0,
        refresh: bool = False,
    ) -> list[QuestionDescriptor]:
        """Discover from Gamma + cache trades. Returns descriptors of cached
        questions. Use this from the CLI / scripts before backtesting.
        """
        manifest = self._load_manifest()
        if kind in ("binary", "both"):
            self._fetch_and_cache_binary(
                manifest, start_iso=start, end_iso=end,
                min_trades=min_trades, min_volume_usd=min_volume_usd, refresh=refresh,
            )
        if kind in ("bucket", "both"):
            self._fetch_and_cache_bucket(
                manifest, start_iso=start, end_iso=end, refresh=refresh,
            )
        self._write_manifest(manifest)
        return self.discover(start=start, end=end, kind=kind)

    # -- internals: descriptors --------------------------------------------

    def _descriptor_from_binary(self, qid: str, entry: dict) -> QuestionDescriptor | None:
        mk = entry.get("market")
        if not mk:
            return None
        return QuestionDescriptor(
            question_id=qid,
            question_idx=_question_idx(qid),
            start_ts_ns=int(mk["start_ts_ns"]),
            end_ts_ns=int(mk["end_ts_ns"]),
            leg_symbols=(str(mk["yes_token_id"]), str(mk["no_token_id"])),
            klass="priceBinary",
            underlying="BTC",
        )

    def _descriptor_from_bucket(self, qid: str, entry: dict) -> QuestionDescriptor | None:
        b = entry.get("bucket")
        if not b:
            return None
        legs: tuple[str, ...] = tuple(
            sym for pair in b["leg_tokens"] for sym in pair
        )
        return QuestionDescriptor(
            question_id=qid,
            question_idx=_question_idx(qid),
            start_ts_ns=int(b["start_ts_ns"]),
            end_ts_ns=int(b["end_ts_ns"]),
            leg_symbols=legs,
            klass="priceBucket",
            underlying="BTC",
        )

    # -- internals: question_view ------------------------------------------

    def _question_view_binary(
        self, q: QuestionDescriptor, entry: dict, *, now_ns: int, settled: bool,
    ) -> QuestionView:
        mk = entry.get("market") or {}
        is_settled = settled or (now_ns > q.end_ts_ns)
        side: Literal["yes", "no", "unknown"] | None = (
            mk.get("resolved_outcome", "unknown") if is_settled else None
        )
        return QuestionView(
            question_idx=q.question_idx,
            yes_symbol=q.leg_symbols[0],
            no_symbol=q.leg_symbols[1],
            strike=self._binary_strike(q),
            expiry_ns=q.end_ts_ns,
            underlying=q.underlying,
            klass=q.klass,
            period="24h",
            settled=is_settled,
            settled_side=side,
            leg_symbols=q.leg_symbols,
        )

    def _binary_strike(self, q: QuestionDescriptor) -> float:
        """Strike convention for 'BTC Up or Down on DATE' PM markets.

        Per the PM rule (parsed from each market's Gamma description), the
        market resolves UP iff the Binance 1m BTC/USDT close at a specific
        strike-reference timestamp < the close at expiry. The strike-reference
        ts is stored in the manifest as ``strike_ref_ts_ns`` when the
        description matches the known rule pattern. For older cache entries
        that predate the parser, fall back to ``end_ts - 24h`` — empirically
        every BTC Up/Down market in the current corpus uses exactly that
        offset, so the fallback matches the parsed value.

        Returns the Binance 1m CLOSE at the strike-reference ts, looked up
        from the cached kline file. Uses the nearest-preceding 1m candle if
        the exact ts isn't present in the data (data gap).
        """
        entry = self._load_manifest().get(q.question_id, {})
        strike_ts_ns = entry.get("strike_ref_ts_ns")
        if strike_ts_ns is None:
            strike_ts_ns = q.end_ts_ns - 24 * 3600 * 1_000_000_000
        strike_ts_ns = int(strike_ts_ns)
        klines = self._load_all_klines()
        if not klines:
            return 0.0
        import bisect
        ts_list = [int(k["ts_ns"]) for k in klines]
        idx = bisect.bisect_right(ts_list, strike_ts_ns) - 1
        if idx < 0:
            return 0.0
        return float(klines[idx]["close"])

    def _question_view_bucket(
        self, q: QuestionDescriptor, entry: dict, *, now_ns: int, settled: bool,
    ) -> QuestionView:
        b = entry.get("bucket") or {}
        thresholds: list[float] = b.get("thresholds") or []
        kv = (("priceThresholds", ",".join(f"{t:.0f}" for t in thresholds)),)
        is_settled = settled or (now_ns > q.end_ts_ns)
        return QuestionView(
            question_idx=q.question_idx,
            yes_symbol="",
            no_symbol="",
            strike=0.0,
            expiry_ns=q.end_ts_ns,
            underlying=q.underlying,
            klass=q.klass,
            period="1h",
            settled=is_settled,
            settled_side=None,
            leg_symbols=q.leg_symbols,
            kv=kv,
        )

    # -- internals: events --------------------------------------------------

    def _events_binary(
        self, q: QuestionDescriptor, entry: dict,
    ) -> Iterator[MarketEvent]:
        cond_id = q.question_id
        yes_t, no_t = q.leg_symbols[0], q.leg_symbols[1]
        leg_pairs = ((yes_t, no_t),)
        leg_trades = {(yes_t, no_t): self._read_trades(cond_id)}
        klines = self._load_klines_window(q.start_ts_ns, q.end_ts_ns)
        outcome = (entry.get("market") or {}).get("resolved_outcome", "unknown")
        per_leg_outcomes: dict[str, Literal["yes", "no", "unknown"]] = {
            yes_t: outcome,
            no_t: _flip(outcome),
        }
        return self._build_stream(
            q=q,
            leg_pairs=leg_pairs,
            trades_by_pair=leg_trades,
            klines=klines,
            per_leg_outcomes=per_leg_outcomes,
        )

    def _events_bucket(
        self, q: QuestionDescriptor, entry: dict,
    ) -> Iterator[MarketEvent]:
        b = entry.get("bucket") or {}
        leg_tokens: list[list[str]] = b["leg_tokens"]
        leg_cond_ids: list[str] = b["leg_condition_ids"]
        leg_resolutions: list[str] = b["leg_resolutions"]
        leg_pairs: tuple[tuple[str, str], ...] = tuple(
            (str(p[0]), str(p[1])) for p in leg_tokens
        )
        trades_by_pair: dict[tuple[str, str], list[_RawTrade]] = {}
        per_leg_outcomes: dict[str, Literal["yes", "no", "unknown"]] = {}
        for pair, cond_id, res in zip(leg_pairs, leg_cond_ids, leg_resolutions):
            trades_by_pair[pair] = self._read_trades(cond_id)
            yes_outcome: Literal["yes", "no", "unknown"] = (
                "yes" if res == "yes" else ("no" if res == "no" else "unknown")
            )
            per_leg_outcomes[pair[0]] = yes_outcome
            per_leg_outcomes[pair[1]] = _flip(yes_outcome)
        klines = self._load_klines_window(q.start_ts_ns, q.end_ts_ns)
        return self._build_stream(
            q=q,
            leg_pairs=leg_pairs,
            trades_by_pair=trades_by_pair,
            klines=klines,
            per_leg_outcomes=per_leg_outcomes,
        )

    def _build_stream(
        self,
        *,
        q: QuestionDescriptor,
        leg_pairs: tuple[tuple[str, str], ...],
        trades_by_pair: dict[tuple[str, str], list["_RawTrade"]],
        klines: list[dict],
        per_leg_outcomes: dict[str, Literal["yes", "no", "unknown"]],
    ) -> Iterator[MarketEvent]:
        """Merge per-leg PM trades (→ BookSnapshot + TradeEvent + within-pair
        parity snapshots) with BTC klines (→ ReferenceEvent) and per-leg
        settlement events at end_ts_ns. Yields in monotone non-decreasing ts.
        """
        cfg = self._stream_cfg
        leg_event_streams: list[Iterable[MarketEvent]] = []
        # Precompute pair lookup for parity inference.
        pair_of: dict[str, tuple[str, str]] = {}
        for pair in leg_pairs:
            pair_of[pair[0]] = pair
            pair_of[pair[1]] = pair

        leg_events: list[MarketEvent] = []
        for pair, trades in trades_by_pair.items():
            for t in sorted(trades, key=lambda r: r.ts_ns):
                snap = trade_to_l2(
                    ts_ns=t.ts_ns, token_id=t.token_id, price=t.price,
                    half_spread=cfg.half_spread, depth=cfg.depth,
                )
                leg_events.append(_book_from_l2(snap))
                leg_events.append(TradeEvent(
                    ts_ns=t.ts_ns, symbol=t.token_id,
                    side=t.side, price=t.price, size=t.size,
                ))
                # Within-pair parity: emit complementary BookSnapshot at 1−p.
                pair_yes, pair_no = pair
                other = pair_no if t.token_id == pair_yes else (
                    pair_yes if t.token_id == pair_no else None
                )
                if other is not None:
                    comp_price = max(_P_CLIP_LO, min(_P_CLIP_HI, 1.0 - t.price))
                    comp_snap = trade_to_l2(
                        ts_ns=t.ts_ns, token_id=other, price=comp_price,
                        half_spread=cfg.half_spread, depth=cfg.depth,
                    )
                    leg_events.append(_book_from_l2(comp_snap))
        leg_event_streams.append(iter(leg_events))

        # Reference stream from klines. Populate `open` so binary-market runners
        # can use the first bar's open as the canonical strike — matching the
        # legacy `day_open_btc` convention.
        leg_event_streams.append(iter(
            ReferenceEvent(
                ts_ns=int(k["ts_ns"]), symbol="BTC",
                high=float(k["high"]), low=float(k["low"]),
                close=float(k["close"]), open=float(k["open"]),
            )
            for k in sorted(klines, key=lambda k: int(k["ts_ns"]))
        ))

        # Per-leg settlement at end_ts_ns.
        settle_events: list[SettlementEvent] = []
        for sym in q.leg_symbols:
            settle_events.append(SettlementEvent(
                ts_ns=q.end_ts_ns,
                question_idx=q.question_idx,
                outcome=per_leg_outcomes.get(sym, "unknown"),
                symbol=sym,
            ))
        leg_event_streams.append(iter(settle_events))

        yield from heapq.merge(*leg_event_streams, key=lambda e: e.ts_ns)

    # -- internals: cache IO -----------------------------------------------

    def _manifest_path(self) -> Path:
        return self._cache_root / "manifest.json"

    def _load_manifest(self) -> dict:
        if self._manifest_cache is not None:
            return self._manifest_cache
        path = self._manifest_path()
        if not path.exists():
            self._manifest_cache = {}
            return self._manifest_cache
        self._manifest_cache = json.loads(path.read_text())
        return self._manifest_cache

    def _write_manifest(self, manifest: dict) -> None:
        self._cache_root.mkdir(parents=True, exist_ok=True)
        self._manifest_path().write_text(json.dumps(manifest, indent=2))
        self._manifest_cache = manifest

    def _read_trades(self, condition_id: str) -> list["_RawTrade"]:
        path = self._cache_root / "pm_trades" / f"{condition_id}.parquet"
        if not path.exists():
            return []
        table = pq.read_table(path)
        rows = table.to_pylist()
        return [
            _RawTrade(
                ts_ns=int(r["ts_ns"]),
                token_id=str(r["token_id"]),
                side=str(r["side"]),  # type: ignore[arg-type]
                price=float(r["price"]),
                size=float(r["size"]),
            )
            for r in rows
        ]

    def _load_all_klines(self) -> list[dict]:
        """Load and cache all BTC kline rows from disk once per instance.

        The kline JSON files are large (~500k rows / ~30 MB for a year of 1m
        BTC bars). Re-parsing them on every `events(q)` call was the dominant
        backtester cost — visible in profiling as ~0.5s per market on the
        v1 tuning grid. Caching collapses that to a one-time hit per worker.
        """
        if self._klines_cache is not None:
            return self._klines_cache
        klines_dir = self._cache_root / "btc_klines"
        rows: list[dict] = []
        if klines_dir.exists():
            for f in sorted(klines_dir.glob("*.json")):
                rows.extend(json.loads(f.read_text()))
        # Sort once so windowed slices via bisect would be possible later.
        rows.sort(key=lambda r: int(r["ts_ns"]))
        self._klines_cache = rows
        return rows

    def _load_klines_window(self, start_ns: int, end_ns: int) -> list[dict]:
        rows = self._load_all_klines()
        return [r for r in rows if start_ns <= int(r["ts_ns"]) <= end_ns]

    # -- internals: live fetch (used by fetch_and_cache only) --------------

    def _fetch_and_cache_binary(
        self, manifest: dict, *, start_iso: str, end_iso: str,
        min_trades: int, min_volume_usd: float, refresh: bool,
    ) -> None:
        raw = _fetch_series_events(_BTC_UPDOWN_SERIES_SLUG)
        in_window = [ev for ev in raw if _event_in_window(ev, start_iso, end_iso)]
        for ev in in_window:
            mkt = _parse_binary_event(ev)
            if mkt is None:
                continue
            cond_id = mkt["condition_id"]
            if cond_id in manifest and not refresh:
                continue
            trades = _fetch_trades_raw(cond_id)
            vol_below_floor = (mkt["total_volume_usd"] > 0) and (
                mkt["total_volume_usd"] < min_volume_usd
            )
            if len(trades) < min_trades or vol_below_floor:
                continue
            self._write_trades_parquet(cond_id, trades)
            mkt["n_trades"] = len(trades)
            manifest[cond_id] = {
                "kind": "binary",
                "n_rows": len(trades),
                "last_pull_ts_ns": int(datetime.now(timezone.utc).timestamp() * 1e9),
                "market": mkt,
            }
            # Persist after each market so a crash mid-fetch leaves a usable
            # resume point. Without this the manifest is only written at the
            # end of fetch_and_cache, so a transient network failure orphans
            # every parquet written so far.
            self._write_manifest(manifest)

    def _fetch_and_cache_bucket(
        self, manifest: dict, *, start_iso: str, end_iso: str, refresh: bool,
    ) -> None:
        raw = _fetch_series_events(_BTC_BUCKET_SERIES_SLUG)
        in_window = [ev for ev in raw if _event_in_window(ev, start_iso, end_iso)]
        for ev in in_window:
            parsed = _parse_bucket_event(ev)
            if parsed is None:
                continue
            slug = parsed["event_slug"]
            if slug in manifest and not refresh:
                continue
            total_rows = 0
            for cond_id in parsed["leg_condition_ids"]:
                trades = _fetch_trades_raw(cond_id)
                self._write_trades_parquet(cond_id, trades)
                total_rows += len(trades)
            manifest[slug] = {
                "kind": "bucket",
                "n_rows": total_rows,
                "last_pull_ts_ns": int(datetime.now(timezone.utc).timestamp() * 1e9),
                "bucket": parsed,
            }
            self._write_manifest(manifest)

    def _write_trades_parquet(self, condition_id: str, raw_trades: list[dict]) -> None:
        import pyarrow as pa  # local to keep top-level import surface small
        path = self._cache_root / "pm_trades" / f"{condition_id}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        ts: list[int] = []
        toks: list[str] = []
        sides: list[str] = []
        prices: list[float] = []
        sizes: list[float] = []
        for row in raw_trades:
            try:
                t = _parse_clob_trade(row)
            except (KeyError, ValueError, TypeError):
                continue
            if t is None:
                continue
            ts.append(t.ts_ns)
            toks.append(t.token_id)
            sides.append(t.side)
            prices.append(t.price)
            sizes.append(t.size)
        table = pa.table({
            "ts_ns": ts, "token_id": toks, "side": sides,
            "price": prices, "size": sizes,
        })
        pq.write_table(table, path)


# ---- Helpers ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _RawTrade:
    ts_ns: int
    token_id: str
    side: str  # "buy" | "sell"
    price: float
    size: float


def _parse_clob_trade(row: dict) -> _RawTrade | None:
    try:
        ts = float(row.get("timestamp", 0))
        return _RawTrade(
            ts_ns=int(ts * 1e9),
            token_id=str(row["asset"]),
            side="buy" if str(row.get("side", "")).upper() == "BUY" else "sell",
            price=float(row["price"]),
            size=float(row["size"]),
        )
    except (KeyError, ValueError, TypeError):
        return None


def _book_from_l2(s: L2Snapshot) -> BookSnapshot:
    return BookSnapshot(
        ts_ns=s.ts_ns,
        symbol=s.token_id,
        bids=((s.bid_px, s.bid_sz),),
        asks=((s.ask_px, s.ask_sz),),
    )


def _flip(outcome: str) -> Literal["yes", "no", "unknown"]:
    if outcome == "yes":
        return "no"
    if outcome == "no":
        return "yes"
    return "unknown"


def _ts_ns_in_iso_window(ts_ns: int, start_iso: str, end_iso: str) -> bool:
    """Inclusive lower, exclusive upper, compared on the date-prefix of the
    ISO representation of `ts_ns`. Mirrors `_event_in_window` for cache rows.
    """
    iso_date = datetime.fromtimestamp(ts_ns / 1e9, tz=timezone.utc).strftime("%Y-%m-%d")
    return start_iso <= iso_date < end_iso


__all__ = ["PolymarketDataSource"]
