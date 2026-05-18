"""Kalshi daily multi-bucket BTC `DataSource`.

Implements the §3.2 ``DataSource`` protocol against a local cache populated
by ``fetch_and_cache(...)``. Cache layout is a single ``manifest.json`` keyed
by Kalshi ``event_ticker`` plus per-market trade parquet files under
``kalshi_trades/``. BTC 1m klines are reused from the existing
``data/binance_klines/`` path (the same one the PM adapter consumes).

Design doc: ``docs/superpowers/specs/2026-05-18-kalshi-buckets-design.md``.
"""
from __future__ import annotations

import heapq
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from ._kalshi_client import (
    fetch_event_detail,
    fetch_events_page,
    iter_events,
    iter_market_trades,
)

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

_HALF_SPREAD_DEFAULT = 0.005

# Known aliases — Kalshi has migrated tickers historically. Probe in order
# until one returns a non-empty page.
_SERIES_TICKER_CANDIDATES = ("KXBTCD", "BTCD", "BTC-D")
_DEPTH_DEFAULT = 10_000.0
_P_CLIP_LO = 1e-6
_P_CLIP_HI = 1.0 - 1e-6
_FLOAT_EPS = 1e-6


class ContiguityError(ValueError):
    """Raised when a Kalshi bucket event's markets don't form a contiguous,
    boundary-covering partition of the BTC price line."""


def _thresholds_from_markets(markets: list[dict]) -> tuple[list[float], list[dict]]:
    """Given Kalshi market dicts with ``floor_strike`` / ``cap_strike`` fields,
    return ``(thresholds, ordered_markets)``. Validates:

    - Lowest market has ``floor_strike is None`` (open at the bottom).
    - Highest market has ``cap_strike is None`` (open at the top).
    - For adjacent markets ``cap_strike[k] == floor_strike[k+1]`` within
      ``_FLOAT_EPS``.

    Raises ``ContiguityError`` on any violation.
    """
    if not markets:
        raise ContiguityError("no markets to derive thresholds from")
    ordered = sorted(
        markets,
        key=lambda m: float("-inf") if m.get("floor_strike") is None else float(m["floor_strike"]),
    )
    if ordered[0].get("floor_strike") is not None:
        raise ContiguityError(
            f"lowest bucket missing open-bottom boundary: {ordered[0].get('ticker')}"
        )
    if ordered[-1].get("cap_strike") is not None:
        raise ContiguityError(
            f"highest bucket missing open-top boundary: {ordered[-1].get('ticker')}"
        )
    thresholds: list[float] = []
    for i in range(len(ordered) - 1):
        cap = ordered[i].get("cap_strike")
        flo = ordered[i + 1].get("floor_strike")
        if cap is None or flo is None:
            raise ContiguityError(
                f"missing inner boundary between {ordered[i].get('ticker')} and "
                f"{ordered[i + 1].get('ticker')}"
            )
        cap_f, flo_f = float(cap), float(flo)
        if abs(cap_f - flo_f) > _FLOAT_EPS:
            kind = "overlap" if cap_f > flo_f else "gap"
            raise ContiguityError(
                f"{kind} between {ordered[i].get('ticker')} (cap={cap_f}) and "
                f"{ordered[i + 1].get('ticker')} (floor={flo_f})"
            )
        thresholds.append(cap_f)
    return thresholds, ordered


def _parse_iso_ns(s: str) -> int:
    if not s:
        return 0
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1e9)


def _question_idx(qid: str) -> int:
    return hash(qid) & 0x7FFFFFFF


def _ts_ns_in_iso_window(ts_ns: int, start_iso: str, end_iso: str) -> bool:
    """`start_iso` / `end_iso` are ``YYYY-MM-DD``. ``ts_ns`` is treated as UTC."""
    dt = datetime.fromtimestamp(ts_ns / 1e9, tz=timezone.utc).date().isoformat()
    return start_iso <= dt < end_iso


@dataclass(frozen=True)
class _StreamCfg:
    half_spread: float
    depth: float


class KalshiDataSource:
    name = "kalshi"

    def __init__(
        self,
        *,
        cache_root: Path,
        half_spread: float = _HALF_SPREAD_DEFAULT,
        depth: float = _DEPTH_DEFAULT,
    ) -> None:
        self._cache_root = Path(cache_root)
        self._stream_cfg = _StreamCfg(half_spread=half_spread, depth=depth)
        self._manifest_cache: dict | None = None
        self._klines_cache: list[dict] | None = None

    # ---- public DataSource protocol -------------------------------------

    def discover(
        self, *, start: str, end: str, **_filters: object,
    ) -> list[QuestionDescriptor]:
        manifest = self._load_manifest()
        out: list[QuestionDescriptor] = []
        for qid, entry in manifest.items():
            if entry.get("kind") != "bucket":
                continue
            d = self._descriptor_from_bucket(qid, entry)
            if d is None:
                continue
            if not _ts_ns_in_iso_window(d.end_ts_ns, start, end):
                continue
            out.append(d)
        out.sort(key=lambda d: d.start_ts_ns)
        return out

    def question_view(
        self, q: QuestionDescriptor, *, now_ns: int, settled: bool,
    ) -> QuestionView:
        entry = self._load_manifest().get(q.question_id) or {}
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
            period="1d",
            settled=is_settled,
            settled_side=None,
            leg_symbols=q.leg_symbols,
            kv=kv,
        )

    def events(self, q: QuestionDescriptor) -> Iterator[MarketEvent]:
        manifest = self._load_manifest()
        entry = manifest.get(q.question_id)
        if entry is None:
            return iter(())
        b = entry.get("bucket") or {}
        leg_markets: list[str] = list(b.get("leg_markets") or [])
        leg_settlements: list[str] = list(b.get("leg_settlements") or [])
        return self._build_stream(
            q=q,
            leg_markets=leg_markets,
            leg_settlements=leg_settlements,
            end_ts_ns=q.end_ts_ns,
        )

    def _build_stream(
        self,
        *,
        q: QuestionDescriptor,
        leg_markets: list[str],
        leg_settlements: list[str],
        end_ts_ns: int,
    ) -> Iterator[MarketEvent]:
        cfg = self._stream_cfg

        # 1) Per-market trades → BookSnapshot + TradeEvent + within-market parity.
        leg_events: list[MarketEvent] = []
        for market in leg_markets:
            trades = self._read_market_trades(market)
            for t in sorted(trades, key=lambda r: int(r["ts_ns"])):
                ts_ns = int(t["ts_ns"])
                p_yes = max(_P_CLIP_LO, min(_P_CLIP_HI, float(t["yes_price"])))
                size = float(t["size"])
                taker_side = str(t["taker_side"])
                yes_sym = f"{market}|yes"
                no_sym = f"{market}|no"
                yes_snap = trade_to_l2(
                    ts_ns=ts_ns, token_id=yes_sym, price=p_yes,
                    half_spread=cfg.half_spread, depth=cfg.depth,
                )
                leg_events.append(_book_from_l2(yes_snap))
                leg_events.append(TradeEvent(
                    ts_ns=ts_ns, symbol=yes_sym,
                    side="buy" if taker_side == "yes" else "sell",
                    price=p_yes, size=size,
                ))
                # Within-market parity: NO leg book at 1 - p_yes.
                no_price = max(_P_CLIP_LO, min(_P_CLIP_HI, 1.0 - p_yes))
                no_snap = trade_to_l2(
                    ts_ns=ts_ns, token_id=no_sym, price=no_price,
                    half_spread=cfg.half_spread, depth=cfg.depth,
                )
                leg_events.append(_book_from_l2(no_snap))

        # 2) BTC ReferenceEvent stream from cached klines.
        klines = self._load_klines_window(q.start_ts_ns, end_ts_ns)
        ref_events = [
            ReferenceEvent(
                ts_ns=int(k["ts_ns"]), symbol="BTC",
                high=float(k["high"]), low=float(k["low"]),
                close=float(k["close"]), open=float(k["open"]),
            )
            for k in sorted(klines, key=lambda k: int(k["ts_ns"]))
        ]

        # 3) Per-leg settlement at end_ts_ns.
        settle: list[SettlementEvent] = []
        for market, market_settlement in zip(leg_markets, leg_settlements):
            yes_outcome = "yes" if market_settlement == "yes" else "no"
            no_outcome = "no" if market_settlement == "yes" else "yes"
            settle.append(SettlementEvent(
                ts_ns=end_ts_ns, question_idx=q.question_idx,
                outcome=yes_outcome, symbol=f"{market}|yes",
            ))
            settle.append(SettlementEvent(
                ts_ns=end_ts_ns, question_idx=q.question_idx,
                outcome=no_outcome, symbol=f"{market}|no",
            ))

        yield from heapq.merge(
            iter(leg_events), iter(ref_events), iter(settle),
            key=lambda e: e.ts_ns,
        )

    def _read_market_trades(self, market: str) -> list[dict]:
        path = self._cache_root / "kalshi_trades" / f"{market}.parquet"
        if not path.exists():
            return []
        return pq.read_table(path).to_pylist()

    def _load_klines_window(self, start_ns: int, end_ns: int) -> list[dict]:
        # Load all cached 1m BTC klines; the cache is filtered to a ~1y window
        # in practice, so loading the lot is fine and matches PM.
        if self._klines_cache is None:
            klines_root = Path(
                os.environ.get("HLBT_BINANCE_KLINES", "data/binance_klines")
            )
            all_rows: list[dict] = []
            if klines_root.exists():
                for p in sorted(klines_root.glob("BTCUSDT-1m*.json")):
                    try:
                        all_rows.extend(json.loads(p.read_text()))
                    except Exception:
                        continue
            self._klines_cache = all_rows
        return [k for k in self._klines_cache
                if start_ns <= int(k["ts_ns"]) <= end_ns]

    # ---- public: audit --------------------------------------------------

    def audit(self) -> dict:
        """Run §7 mutex + contiguity checks against the cached corpus.

        Returns the `fetch_summary` dict and persists it to
        `<cache_root>/fetch_summary.json`. Caller (the CLI) decides exit code
        based on counts.
        """
        manifest = self._load_manifest()
        mutex_pass = mutex_fail_zero = mutex_fail_multi = contiguity_fail = 0
        events_settled = events_open = 0
        failing: list[str] = []

        for qid, entry in manifest.items():
            if entry.get("kind") != "bucket":
                continue
            b = entry.get("bucket") or {}
            settlements: list[str] = list(b.get("leg_settlements") or [])
            ranges: list[list] = list(b.get("leg_strike_ranges") or [])
            if not settlements:
                events_open += 1
                continue
            events_settled += 1
            ranges_as_markets = [
                {"ticker": m, "floor_strike": r[0], "cap_strike": r[1]}
                for m, r in zip(b.get("leg_markets") or [], ranges)
            ]
            try:
                _thresholds_from_markets(ranges_as_markets)
            except ContiguityError:
                contiguity_fail += 1
                failing.append(qid)
                continue
            yes_count = sum(1 for s in settlements if s == "yes")
            if yes_count == 1:
                mutex_pass += 1
            elif yes_count == 0:
                mutex_fail_zero += 1
                failing.append(qid)
            else:
                mutex_fail_multi += 1
                failing.append(qid)

        denom = max(events_settled, 1)
        summary = {
            "series_ticker_resolved": next(
                (e.get("bucket", {}).get("series_ticker") for e in manifest.values()
                 if e.get("kind") == "bucket"),
                None,
            ),
            "events_fetched": sum(1 for e in manifest.values() if e.get("kind") == "bucket"),
            "events_settled": events_settled,
            "events_open": events_open,
            "mutex_pass": mutex_pass,
            "mutex_fail_zero_yes": mutex_fail_zero,
            "mutex_fail_multi_yes": mutex_fail_multi,
            "contiguity_fail": contiguity_fail,
            "mutex_rate": mutex_pass / denom,
            "failing_event_tickers": sorted(set(failing)),
        }
        (self._cache_root / "fetch_summary.json").write_text(json.dumps(summary, indent=2))
        return summary

    # ---- public: fetch --------------------------------------------------

    def fetch_and_cache(
        self,
        *,
        start: str,
        end: str,
        series_ticker: str | None = None,
        refresh: bool = False,
    ) -> list[QuestionDescriptor]:
        """Discover from Kalshi public REST + cache trades. Returns descriptors
        of cached questions in [start, end).
        """
        self._cache_root.mkdir(parents=True, exist_ok=True)
        (self._cache_root / "kalshi_trades").mkdir(parents=True, exist_ok=True)

        resolved = self._resolve_series_ticker(series_ticker)
        manifest = self._load_manifest()

        for ev_summary in iter_events(series_ticker=resolved, status="settled"):
            event_ticker = ev_summary.get("event_ticker")
            if not event_ticker:
                continue
            exp = ev_summary.get("expiration_time") or ""
            end_ts_ns = _parse_iso_ns(exp)
            if end_ts_ns == 0:
                continue
            if not _ts_ns_in_iso_window(end_ts_ns, start, end):
                continue
            if event_ticker in manifest and not refresh:
                continue
            event, markets = fetch_event_detail(event_ticker)
            try:
                thresholds, ordered_markets = _thresholds_from_markets(markets)
            except ContiguityError as e:
                from loguru import logger
                logger.warning(f"kalshi {event_ticker}: contiguity error: {e}")
                continue

            leg_market_tickers = [m["ticker"] for m in ordered_markets]
            leg_settlements = [
                str(m.get("settlement_value") or "unknown")
                for m in ordered_markets
            ]
            leg_strike_ranges = [
                [m.get("floor_strike"), m.get("cap_strike")]
                for m in ordered_markets
            ]
            start_ts_ns = min(
                (_parse_iso_ns(m.get("open_time") or "") for m in ordered_markets),
                default=end_ts_ns - 86_400 * 1_000_000_000,
            )

            n_rows = 0
            for market in leg_market_tickers:
                rows = self._fetch_market_trades_cached(
                    market, refresh=refresh,
                )
                n_rows += len(rows)

            yes_count = sum(1 for s in leg_settlements if s == "yes")
            manifest[event_ticker] = {
                "n_rows": n_rows,
                "last_pull_ts_ns": int(datetime.now(timezone.utc).timestamp() * 1e9),
                "kind": "bucket",
                "bucket": {
                    "event_ticker": event_ticker,
                    "series_ticker": resolved,
                    "start_ts_ns": int(start_ts_ns),
                    "end_ts_ns": int(end_ts_ns),
                    "thresholds": thresholds,
                    "leg_markets": leg_market_tickers,
                    "leg_strike_ranges": leg_strike_ranges,
                    "leg_settlements": leg_settlements,
                    "mutex_verified": yes_count == 1,
                    "settlement_close_price": None,
                },
            }

        self._write_manifest(manifest)
        return self.discover(start=start, end=end)

    def _resolve_series_ticker(self, override: str | None) -> str:
        if override:
            return override
        for cand in _SERIES_TICKER_CANDIDATES:
            page, _ = fetch_events_page(
                series_ticker=cand, status="settled", limit=1,
            )
            if page:
                return cand
        raise SystemExit(
            f"kalshi: no series_ticker probe succeeded among "
            f"{_SERIES_TICKER_CANDIDATES}; pass --series-ticker explicitly."
        )

    def _fetch_market_trades_cached(
        self, market: str, *, refresh: bool,
    ) -> list[dict]:
        path = self._cache_root / "kalshi_trades" / f"{market}.parquet"
        existing: list[dict] = []
        last_ts_ns: int | None = None
        if path.exists() and not refresh:
            existing = pq.read_table(path).to_pylist()
            if existing:
                last_ts_ns = max(int(r["ts_ns"]) for r in existing)
        new_rows: list[dict] = []
        for t in iter_market_trades(
            market, min_ts=last_ts_ns // 1_000_000_000 if last_ts_ns else None,
        ):
            ts_ns = _parse_iso_ns(t.get("created_time") or "")
            if not ts_ns:
                continue
            new_rows.append({
                "ts_ns": ts_ns,
                "yes_price": float(t.get("yes_price", 50)) / 100.0,
                "size": float(t.get("count", 0)),
                "taker_side": str(t.get("taker_side") or "yes"),
            })
        all_rows = existing + new_rows
        if all_rows:
            table = pa.Table.from_pylist(sorted(all_rows, key=lambda r: int(r["ts_ns"])))
            pq.write_table(table, path)
        return all_rows

    def _write_manifest(self, manifest: dict) -> None:
        self._manifest_path().write_text(json.dumps(manifest, indent=2))
        self._manifest_cache = manifest

    # ---- internals: descriptor build ------------------------------------

    def _descriptor_from_bucket(
        self, qid: str, entry: dict,
    ) -> QuestionDescriptor | None:
        b = entry.get("bucket")
        if not b:
            return None
        markets: list[str] = list(b.get("leg_markets") or [])
        if not markets:
            return None
        legs: tuple[str, ...] = tuple(
            sym
            for m in markets
            for sym in (f"{m}|yes", f"{m}|no")
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

    # ---- internals: cache IO --------------------------------------------

    def _manifest_path(self) -> Path:
        return self._cache_root / "manifest.json"

    def _load_manifest(self) -> dict:
        if self._manifest_cache is not None:
            return self._manifest_cache
        p = self._manifest_path()
        self._manifest_cache = json.loads(p.read_text()) if p.exists() else {}
        return self._manifest_cache


def _book_from_l2(s: L2Snapshot) -> BookSnapshot:
    return BookSnapshot(
        ts_ns=s.ts_ns,
        symbol=s.token_id,
        bids=((s.bid_px, s.bid_sz),),
        asks=((s.ask_px, s.ask_sz),),
    )
