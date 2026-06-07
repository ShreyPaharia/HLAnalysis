"""Polymarket data source — binary (e.g. BTC / WTI Up-or-Down daily) and
bucket (Bitcoin Multi Strikes Hourly) markets.

Implements the §3.2 `DataSource` protocol against a local cache populated by
`fetch_and_cache(...)`. The cache layout is a single `manifest.json` keyed by
question_id (PM `condition_id` for binaries, PM event slug for buckets) plus
per-leg trade parquet files under `pm_trades/` and 1m reference klines under
`<klines_subdir>/*.json` (default `btc_klines/`; `wti_klines/` when
constructed with `reference_symbol="WTI"`). See
`docs/specs/2026-05-11-task-c-plan.md` §3 for the on-disk shape.

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

from .binance_klines import fetch_klines
from ..core.data_source import QuestionDescriptor
from ..core.events import (
    BookSnapshot,
    MarketEvent,
    ReferenceEvent,
    SettlementEvent,
    TradeEvent,
)
from ._synthetic_l2 import L2Snapshot, trade_to_l2
from hlanalysis.adapters.polymarket_normalize import parse_bucket_event as _shared_parse_bucket_event

_GAMMA_BASE = "https://gamma-api.polymarket.com"
_CLOB_DATA_BASE = "https://data-api.polymarket.com"
_CLOB_BASE = "https://clob.polymarket.com"
_DEFAULT_REF_SYMBOL = "BTC"
_DEFAULT_BINARY_SERIES_SLUG = "btc-up-or-down-daily"
_DEFAULT_KLINES_SUBDIR = "btc_klines"
# Genuine Binance 1s klines for the `klines_1s` reference source — kept in a
# separate subdir so the canonical 1m `btc_klines` (PM strike resolution) is
# untouched. Populated on demand by scripts/fetch_binance_1s_klines.py.
_DEFAULT_KLINES_1S_SUBDIR = "btc_klines_1s"
# Map PM reference_symbol → Binance perp partition symbol for the BBO-tick
# reference variant (cadence-sweep path). Only BTC has overlapping recorded
# tick coverage right now.
_BINANCE_PERP_REF_SYMBOL = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT"}
_BINANCE_PERP_DATA_SUBPATH = "venue=binance/product_type=perp/mechanism=clob"
# reference_symbol → Binance SPOT kline symbol used for strike resolution +
# the coupled fetch (SHR-54). Spot symbols match the perp strings on Binance.
# Underlyings absent here (e.g. WTI → Pyth klines) are not Binance-fetchable.
_BINANCE_SPOT_KLINE_SYMBOL = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT"}
# One 1m bar in ns — coverage tolerance for the strike guard.
_BAR_NS = 60 * 1_000_000_000


class StrikeCoverageError(Exception):
    """Raised when a PM market's strike-reference ts falls outside the cached
    Binance kline series, so the strike cannot be resolved without fabricating
    a frozen value. Fails loud at strike-resolution time rather than letting
    `_binary_strike` silently return a stale close (SHR-54); the coupled fetch
    in `fetch_and_cache` keeps the cache covered so this should not fire in the
    normal populate→run flow."""
# Recorded PM L2 book partitions (native recorder; coverage starts 2026-05-27).
# Symbol partition is the PM token_id, joining to manifest yes/no token ids.
_PM_BOOK_DATA_SUBPATH = (
    "venue=polymarket/product_type=prediction_binary/mechanism=clob/event=book_snapshot"
)
# Recorded PM settlement (on-chain redemption): the winning leg token redeems
# at settle_price≈1.0. Authoritative for the resolved outcome when present.
_PM_SETTLEMENT_DATA_SUBPATH = (
    "venue=polymarket/product_type=prediction_binary/mechanism=clob/event=settlement"
)
_DEFAULT_BUCKET_SERIES_SLUG = "btc-multi-strikes-weekly"
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

    Delegates leg-collection + strike-sorting to the shared
    `parse_bucket_event` (polymarket_normalize) so the live adapter and
    backtest always produce identical leg orderings. The backtest-only fields
    (`event_slug`, `start_ts_ns`, `end_ts_ns`, `leg_resolutions`) are
    appended here; `thresholds`/`leg_tokens`/`leg_condition_ids` come verbatim
    from the shared helper to avoid drift.
    """
    base = _shared_parse_bucket_event(ev)
    if base is None:
        return None
    slug = ev.get("slug") or ev.get("ticker")
    start_iso, end_iso = ev.get("startDate"), ev.get("endDate")
    if not (slug and start_iso and end_iso):
        return None
    # leg_resolutions stays backtest-only (settlement replay); recompute in
    # the SAME strike-ascending order the shared parser used.
    pairs = []  # (strike, yes_p, no_p)
    for mk in ev.get("markets") or []:
        raw = mk.get("clobTokenIds")
        if not raw:
            continue
        tok = json.loads(raw) if isinstance(raw, str) else raw
        if len(tok) != 2:
            continue
        try:
            # Match the shared parser's strike cleaning (lstrip "$") so the
            # re-sorted leg_resolutions stay positionally aligned with the
            # shared helper's thresholds/leg_tokens even for "$80,000" titles.
            strike = float(str(mk.get("groupItemTitle") or "").lstrip("$").replace(",", ""))
        except ValueError:
            continue
        res = "unknown"
        op = mk.get("outcomePrices")
        if op:
            pr = json.loads(op) if isinstance(op, str) else op
            if len(pr) == 2:
                if float(pr[0]) >= 0.99:
                    res = "yes"
                elif float(pr[1]) >= 0.99:
                    res = "no"
        pairs.append((strike, res))
    pairs.sort(key=lambda x: x[0])
    return {
        "event_slug": str(slug),
        "start_ts_ns": _parse_iso_ns(start_iso),
        "end_ts_ns": _parse_iso_ns(end_iso),
        "thresholds": base["thresholds"],
        "leg_tokens": base["leg_tokens"],
        "leg_condition_ids": base["leg_condition_ids"],
        "leg_resolutions": [r for _, r in pairs],
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
        reference_symbol: str = _DEFAULT_REF_SYMBOL,
        series_slug: str = _DEFAULT_BINARY_SERIES_SLUG,
        bucket_series_slug: str = _DEFAULT_BUCKET_SERIES_SLUG,
        klines_subdir: str = _DEFAULT_KLINES_SUBDIR,
        klines_1s_subdir: str = _DEFAULT_KLINES_1S_SUBDIR,
        reference_source: Literal["klines", "binance_bbo", "klines_1s"] = "klines",
        reference_resample_seconds: int = 60,
        binance_data_root: Path | str | None = None,
        binance_bbo_product_type: Literal["perp", "spot"] = "perp",
        book_source: Literal["synthetic", "recorded"] = "synthetic",
        pm_book_root: Path | str | None = None,
        liquidity_profile_path: Path | str | None = None,
    ) -> None:
        self._cache_root = Path(cache_root)
        self._stream_cfg = _StreamCfg(half_spread=half_spread, depth=depth)
        self._reference_symbol = reference_symbol
        self._series_slug = series_slug
        self._bucket_series_slug = bucket_series_slug
        self._klines_subdir = klines_subdir
        self._klines_1s_subdir = klines_1s_subdir
        # Reference-feed mode.
        # - "klines" (default): emit cached 1m Binance klines as ReferenceEvents.
        #   This is the 12-month-corpus path used by the v3.1/v3.5/v3.6 tune.
        # - "binance_bbo": read recorded Binance BBO ticks from
        #   `<binance_data_root>/venue=binance/product_type=<perp|spot>/...` and
        #   bucket to `reference_resample_seconds` OHLC. Lets us probe
        #   sub-minute cadence on the BBO-tick overlap window; symmetrical to
        #   the HL HIP-4 cadence-parameterization story.
        #   Use `binance_bbo_product_type="spot"` to match PM's settlement
        #   instrument (Binance SPOT 1m close); default is "perp" for
        #   backward-compatibility with existing BBO cadence sweeps.
        if reference_source not in ("klines", "binance_bbo", "klines_1s"):
            raise ValueError(
                "reference_source must be 'klines', 'binance_bbo', or 'klines_1s', "
                f"got {reference_source!r}"
            )
        if reference_resample_seconds <= 0:
            raise ValueError(f"reference_resample_seconds must be positive, got {reference_resample_seconds}")
        if binance_bbo_product_type not in ("perp", "spot"):
            raise ValueError(
                f"binance_bbo_product_type must be 'perp' or 'spot', got {binance_bbo_product_type!r}"
            )
        self._reference_source = reference_source
        self._reference_resample_seconds = int(reference_resample_seconds)
        self._reference_resample_ns = int(reference_resample_seconds) * 1_000_000_000
        self._binance_bbo_product_type = binance_bbo_product_type
        # Binance tick partitions live next to PM cache root by default
        # (`data/venue=binance/...` while PM cache lives in `data/sim/`).
        self._binance_data_root = (
            Path(binance_data_root) if binance_data_root is not None
            else self._cache_root.parent
        )
        # Book-fill source.
        # - "synthetic" (default): one flat 1-level book per PM trade print at
        #   ±half_spread + within-pair `1−p` parity. Bit-identical to the
        #   12-month-corpus tune.
        # - "recorded": real multi-level L2 `book_snapshot` parquet per leg
        #   (native recorder; coverage from 2026-05-27). Drops the synthetic
        #   book + parity; the matching engine walks the real depth. HL parity.
        if book_source not in ("synthetic", "recorded"):
            raise ValueError(f"book_source must be 'synthetic' or 'recorded', got {book_source!r}")
        self._book_source = book_source
        # PM book partitions live next to the binance ticks under the same repo
        # `data/` root by default (PM cache is `data/sim/`, so `.parent`).
        self._pm_book_root = (
            Path(pm_book_root) if pm_book_root is not None
            else self._cache_root.parent
        )
        # Optional per-bucket liquidity calibration for the synthetic book builder.
        # When set, trade_to_l2() uses the profile's half_spread/depth per price
        # bucket instead of the flat _StreamCfg constants.
        self._liquidity_profile: "LiquidityProfile | None" = None
        if liquidity_profile_path:
            import json as _json
            from ._synthetic_l2 import LiquidityProfile as _LiquidityProfile
            with open(liquidity_profile_path) as _f:
                _d = _json.load(_f)
            try:
                self._liquidity_profile = _LiquidityProfile(
                    bucket_width=_d["bucket_width"],
                    half_spread=_d["half_spread"],
                    depth=_d["depth"],
                    global_half_spread=_d["global_half_spread"],
                    global_depth=_d["global_depth"],
                )
            except KeyError as _e:
                raise ValueError(
                    f"liquidity profile {liquidity_profile_path} is missing "
                    f"required key {_e}; expected bucket_width, half_spread, "
                    f"depth, global_half_spread, global_depth"
                ) from _e
        # Lazy caches populated on first read. Significant for tuning workers
        # that backtest dozens of markets per cell — without caches each
        # market would re-parse the manifest + the (large) BTC klines JSON.
        self._manifest_cache: dict | None = None
        self._klines_cache: list[dict] | None = None
        self._klines_1s_cache: list[dict] | None = None
        # Markets we've already warned about for settlement→manifest fallback,
        # so the per-market warning fires once (resolved_outcome is queried
        # several times per market across the runner + CLI + event stream).
        self._settlement_fallback_warned: set[str] = set()
        # BBO ticks are window-scoped (per-question) so we don't pre-cache.

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

    def events_arrays(self, q: QuestionDescriptor):
        """Return a ``FastPathBundle`` for both *recorded* and *synthetic* mode,
        skipping the per-market dataclass overhead so repeated grid-cell replays
        hit the disk cache rather than rebuilding arrays from scratch each time.

        Both modes are supported:

        - ``book_source="recorded"``: reads the real multi-level L2
          ``book_snapshot`` parquet directly via the column-vectorised assembler.
          Bit-equivalent to the legacy ``events()`` path for that mode.
        - ``book_source="synthetic"`` (default): drives the same legacy
          ``events()`` stream the runner consumes, partitions events by type and
          symbol exactly as the runner does, and calls the shared in-memory
          assembler — producing arrays that are **bit-identical** to the legacy
          path while enabling the disk cache for grid-cell reuse.

        The bundle is cached on disk under ``<cache_root>/_event_array_cache``
        keyed by ``(question_id, config_sig, source-file mtimes)``.  A second
        call for the same question + config returns the cached bundle without
        re-reading any source files.
        """

        from ._event_array_cache import inproc_lookup

        force_rebuild = getattr(self, "_force_rebuild_cache", False)
        config_sig = self._bundle_config_sig()
        # Short-circuit BEFORE the manifest load + source-file glob on a
        # process-memo hit (tune replays the same question across param cells).
        memo_hit = inproc_lookup(q.question_id, config_sig, force_rebuild=force_rebuild)
        if memo_hit is not None:
            return memo_hit

        manifest = self._load_manifest()
        entry = manifest.get(q.question_id)
        if entry is None:
            import numpy as np
            from ._fastpath_core import FastPathBundle, LegArrays, event_dtype
            empty = np.zeros(0, dtype=event_dtype)
            return FastPathBundle(
                leg_arrays={sym: LegArrays(events=empty, book_ts=np.zeros(0, dtype=np.int64)) for sym in q.leg_symbols},
                reference_events=[],
                settlement_events=[],
            )

        from ._pm_fastpath import build_pm_fast_path_bundle, build_pm_synthetic_fast_path_bundle

        # All source reads (trades, klines/BBO reference, book parquet,
        # settlement) are deferred into ``_build`` so a cache HIT skips ALL of
        # them — only the cheap source-file stat + npz load runs.
        def _build() -> "FastPathBundle":
            if self._book_source == "synthetic":
                # Synthetic: assemble arrays from the SAME legacy event stream
                # the runner consumes. Partitions exactly as the runner does and
                # calls the same shared assembler → bit-identical output.
                return build_pm_synthetic_fast_path_bundle(
                    q=q,
                    events_iter=self.events(q),
                )

            # Recorded: vectorised column path (the original recorded assembler).
            if entry.get("kind", "binary") == "binary":
                trades = self._read_trades(q.question_id)
                outcome = self._binary_outcome(q, entry)
                yes_t, no_t = q.leg_symbols[0], q.leg_symbols[1]
                per_leg_outcomes: dict[str, str] = {yes_t: outcome, no_t: _flip(outcome)}
            else:
                b = entry.get("bucket") or {}
                leg_tokens: list[list[str]] = b.get("leg_tokens", [])
                leg_cond_ids: list[str] = b.get("leg_condition_ids", [])
                leg_resolutions: list[str] = b.get("leg_resolutions", [])
                trades = []
                per_leg_outcomes = {}
                for pair, cond_id, res in zip(leg_tokens, leg_cond_ids, leg_resolutions):
                    trades.extend(self._read_trades(cond_id))
                    yes_outcome: Literal["yes", "no", "unknown"] = (
                        "yes" if res == "yes" else ("no" if res == "no" else "unknown")
                    )
                    per_leg_outcomes[str(pair[0])] = yes_outcome
                    per_leg_outcomes[str(pair[1])] = _flip(yes_outcome)

            # Reference events (klines / BBO / 1s-klines — same logic as _build_stream).
            if self._reference_source == "binance_bbo":
                ref_events = self._load_binance_bbo_reference(q.start_ts_ns, q.end_ts_ns)
            elif self._reference_source == "klines_1s":
                ref_events = self._load_klines_1s_reference(q.start_ts_ns, q.end_ts_ns)
            else:
                klines = self._load_klines_window(q.start_ts_ns, q.end_ts_ns)
                ref_events = [
                    ReferenceEvent(
                        ts_ns=int(k["ts_ns"]), symbol=self._reference_symbol,
                        high=float(k["high"]), low=float(k["low"]),
                        close=float(k["close"]), open=float(k["open"]),
                    )
                    for k in sorted(klines, key=lambda k: int(k["ts_ns"]))
                ]

            settle_events = [
                SettlementEvent(
                    ts_ns=q.end_ts_ns,
                    question_idx=q.question_idx,
                    outcome=per_leg_outcomes.get(sym, "unknown"),  # type: ignore[arg-type]
                    symbol=sym,
                )
                for sym in q.leg_symbols
            ]
            return build_pm_fast_path_bundle(
                q=q,
                book_glob_for=self._recorded_book_glob,
                trades=trades,
                reference_events=ref_events,
                settlement_events=settle_events,
            )

        from ._event_array_cache import cached_bundle

        return cached_bundle(
            self._cache_root / "_event_array_cache",
            q.question_id,
            self._fastpath_source_files(q),
            _build,
            force_rebuild=force_rebuild,
            config_sig=config_sig,
        )

    def _bundle_config_sig(self) -> str:
        """Cache-key signature: every non-source-file input that changes the
        built bundle (resample period + reference/book source mode + liquidity
        profile), so a bundle built under one config never aliases to a request
        under another. Keep in sync with the params that flow into ``_build``."""
        lp = self._liquidity_profile
        if lp is None:
            lp_sig = "none"
        else:
            lp_sig = (
                f"{lp.bucket_width}:{lp.global_half_spread}:{lp.global_depth}"
                f":{hash(tuple(lp.half_spread))}:{hash(tuple(lp.depth))}"
            )
        return (
            f"rrs={self._reference_resample_ns}"
            f"|refsrc={self._reference_source}"
            f"|book={self._book_source}"
            f"|bbo={self._binance_bbo_product_type}"
            f"|lp={lp_sig}"
            f"|k1m={self._klines_subdir}|k1s={self._klines_1s_subdir}"
        )

    def _fastpath_source_files(self, q: QuestionDescriptor) -> list[Path]:
        """Concrete parquet/json files feeding ``events_arrays(q)`` — the
        event-array cache keys on their (size, mtime) so any re-record or
        kline refresh invalidates the cached bundle."""
        from glob import glob as _glob

        files: list[str] = []
        for sym in q.leg_symbols:
            files += _glob(self._recorded_book_glob(sym), recursive=True)
            files += _glob(
                str(self._pm_book_root / _PM_SETTLEMENT_DATA_SUBPATH
                    / f"symbol={sym}" / "**" / "*.parquet"),
                recursive=True,
            )
        entry = self._load_manifest().get(q.question_id) or {}
        cond_ids = [q.question_id]
        if entry.get("kind") == "bucket":
            cond_ids = (entry.get("bucket") or {}).get("leg_condition_ids", []) or cond_ids
        for c in cond_ids:
            p = self._cache_root / "pm_trades" / f"{c}.parquet"
            if p.exists():
                files.append(str(p))
        # Reference events derive from klines — include them in the key.
        files += _glob(str(self._cache_root / self._klines_subdir / "*.json"))
        if self._reference_source == "klines_1s":
            files += _glob(str(self._cache_root / self._klines_1s_subdir / "*.json"))
        return [Path(f) for f in files]

    def _recorded_book_glob(self, token_id: str) -> str:
        """Glob pattern for the recorded ``book_snapshot`` parquet of one leg.

        Shared by ``_load_recorded_book`` and the fast path
        ``events_arrays``.
        """
        return str(
            self._pm_book_root / _PM_BOOK_DATA_SUBPATH
            / f"symbol={token_id}" / "**" / "*.parquet"
        )

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
            return self._binary_outcome(q, entry)
        # Bucket: outcome aggregates the per-leg resolutions; we report
        # "unknown" because there's no single bucket outcome. Per-leg outcomes
        # are emitted via `SettlementEvent.symbol` in events().
        return "unknown"

    def leg_payoff(self, q: QuestionDescriptor, leg_symbol: str) -> float:
        """Per-leg payoff at settlement: 1.0 if the held leg won, else 0.0.

        Binary: mirrors the runner's binary fallback (yes=leg_symbols[0],
        no=leg_symbols[1]) via ``resolved_outcome`` — kept bit-identical so the
        binary settlement path is unchanged.

        Bucket: PM multi-strike is a ladder of independent 'above X' binaries.
        Each manifest leg pair is (yes_token, no_token) with a per-leg resolution.
        The YES token wins iff its leg resolved 'yes'; the NO token wins iff 'no'.
        """
        manifest = self._load_manifest()
        entry = manifest.get(q.question_id) or {}
        kind = entry.get("kind", "binary")
        if kind != "bucket":
            outcome = self.resolved_outcome(q)
            if outcome == "yes" and q.leg_symbols and leg_symbol == q.leg_symbols[0]:
                return 1.0
            if outcome == "no" and len(q.leg_symbols) > 1 and leg_symbol == q.leg_symbols[1]:
                return 1.0
            return 0.0
        b = entry.get("bucket") or {}
        for toks, res in zip(b.get("leg_tokens", []), b.get("leg_resolutions", [])):
            yes_tok, no_tok = str(toks[0]), str(toks[1])
            if leg_symbol == yes_tok:
                return 1.0 if res == "yes" else 0.0
            if leg_symbol == no_tok:
                return 1.0 if res == "no" else 0.0
        return 0.0

    def _binary_outcome(
        self, q: QuestionDescriptor, entry: dict,
    ) -> Literal["yes", "no", "unknown"]:
        """Resolved winner for a binary market.

        The recorder's on-chain `settlement` event is authoritative when
        present (the winning leg token redeems at settle_price≈1.0); we fall
        back to the manifest's Gamma-derived ``resolved_outcome`` for markets
        without recorder coverage (the whole pre-2026-05-27 corpus), so legacy
        backtests stay bit-identical. Independent of ``book_source`` — this
        governs settlement payoff, not fills.
        """
        rec = self._recorded_outcome(q)
        if rec != "unknown":
            return rec
        mk = entry.get("market") or {}
        fallback = mk.get("resolved_outcome", "unknown")
        if q.question_id not in self._settlement_fallback_warned:
            self._settlement_fallback_warned.add(q.question_id)
            logger.warning(
                f"PM winner: no recorder settlement for {q.question_id}; falling "
                f"back to manifest resolved_outcome={fallback!r} (Gamma-derived, "
                f"less authoritative than on-chain settlement)"
            )
        return fallback  # type: ignore[return-value]

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
        # SHR-54: keep the kline series advancing in lockstep with the PM market
        # cache. Without this the klines froze behind newly-cached markets and
        # their strikes silently resolved to a stale close.
        starts: list[int] = []
        ends: list[int] = []
        for e in manifest.values():
            if e.get("kind") == "binary":
                mk = e.get("market") or {}
                if mk.get("start_ts_ns"):
                    starts.append(int(mk["start_ts_ns"]))
                if mk.get("end_ts_ns"):
                    ends.append(int(mk["end_ts_ns"]))
            elif e.get("kind") == "bucket":
                b = e.get("bucket") or {}
                if b.get("start_ts_ns"):
                    starts.append(int(b["start_ts_ns"]))
                if b.get("end_ts_ns"):
                    ends.append(int(b["end_ts_ns"]))
        if starts and ends:
            self._ensure_kline_coverage(min(starts), max(ends))
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
            underlying=self._reference_symbol,
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
            underlying=self._reference_symbol,
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
        strike_ts_ns = self._strike_ts_ns(q)
        klines = self._load_all_klines()
        if not klines:
            raise StrikeCoverageError(
                f"{q.question_id}: no cached klines in {self._klines_subdir}/ — "
                f"cannot resolve strike at ts={strike_ts_ns}"
            )
        import bisect
        ts_list = [int(k["ts_ns"]) for k in klines]
        # Coverage guard (SHR-54): the strike ts must fall inside the cached
        # series. Past the last candle (+1 bar tolerance) bisect would silently
        # return the final close as a frozen, arbitrarily-wrong strike; before
        # the first candle there is no preceding close at all. Either way the
        # cache does not cover this market — refuse rather than fabricate.
        # Interior gaps (a missing minute with later candles, e.g. weekend) are
        # still resolved to the nearest-preceding close below.
        if strike_ts_ns < ts_list[0] or strike_ts_ns > ts_list[-1] + _BAR_NS:
            raise StrikeCoverageError(
                f"{q.question_id}: strike ts={strike_ts_ns} outside cached kline "
                f"coverage [{ts_list[0]}, {ts_list[-1]}] in {self._klines_subdir}/"
            )
        idx = bisect.bisect_right(ts_list, strike_ts_ns) - 1
        return float(klines[idx]["close"])

    def _strike_ts_ns(self, q: QuestionDescriptor) -> int:
        """Resolve the strike-reference ts for a binary market: the parsed
        ``strike_ref_ts_ns`` from the manifest, else the ``end_ts - 24h``
        fallback (every BTC Up/Down market in the corpus uses that offset)."""
        entry = self._load_manifest().get(q.question_id, {})
        strike_ts_ns = entry.get("strike_ref_ts_ns")
        # Non-BTC underlyings (e.g. WTI) hit this fallback because the strike-rule
        # regex is BTC-specific. The nearest-preceding-1m-candle lookup correctly
        # resolves to the prior trading session's close on weekend gaps, matching
        # PM's "most recent prior trading day" resolution semantics.
        if strike_ts_ns is None:
            strike_ts_ns = q.end_ts_ns - 24 * 3600 * 1_000_000_000
        return int(strike_ts_ns)

    def _question_view_bucket(
        self, q: QuestionDescriptor, entry: dict, *, now_ns: int, settled: bool,
    ) -> QuestionView:
        b = entry.get("bucket") or {}
        thresholds: list[float] = b.get("thresholds") or []
        kv = (
            ("priceThresholds", ",".join(f"{t:.0f}" for t in thresholds)),
            ("bucketLayout", "above_ladder"),
        )
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
        # Settlement payoff prefers the recorder's on-chain settlement event
        # (authoritative); falls back to the manifest outcome when uncovered.
        outcome = self._binary_outcome(q, entry)
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

        # In "recorded" mode the real multi-level book replaces the synthetic
        # 1-level book + the `1−p` parity synthesis — we have the recorded book
        # for BOTH legs. Real PM TradeEvents are still emitted in both modes.
        recorded = self._book_source == "recorded"
        leg_events: list[MarketEvent] = []
        for pair, trades in trades_by_pair.items():
            for t in sorted(trades, key=lambda r: r.ts_ns):
                if not recorded:
                    snap = trade_to_l2(
                        ts_ns=t.ts_ns, token_id=t.token_id, price=t.price,
                        half_spread=cfg.half_spread, depth=cfg.depth,
                        profile=self._liquidity_profile,
                    )
                    leg_events.append(_book_from_l2(snap))
                leg_events.append(TradeEvent(
                    ts_ns=t.ts_ns, symbol=t.token_id,
                    side=t.side, price=t.price, size=t.size,
                ))
                if not recorded:
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
                            profile=self._liquidity_profile,
                        )
                        leg_events.append(_book_from_l2(comp_snap))
        leg_event_streams.append(iter(leg_events))

        # Recorded multi-level book stream (one read per leg token). Empty when
        # a leg has no recorded coverage (logged) — the market degrades to
        # trade/reference-only and the runner simply gets no fills there.
        if recorded:
            recorded_books: list[BookSnapshot] = []
            for sym in q.leg_symbols:
                recorded_books.extend(
                    self._load_recorded_book(sym, q.start_ts_ns, q.end_ts_ns)
                )
            recorded_books.sort(key=lambda b: b.ts_ns)
            leg_event_streams.append(iter(recorded_books))

        # Reference stream. Two sources:
        # - "klines": cached 1m Binance klines (legacy 12-month-corpus path).
        # - "binance_bbo": recorded Binance perp BBO ticks, bucketed to
        #   `reference_resample_seconds` OHLC bars. Lets us probe cadence on
        #   the BBO-tick overlap window symmetrically with HL HIP-4.
        # Strike resolution (_binary_strike) ALWAYS uses 1m klines because
        # PM's resolution rule is defined by Binance spot 1m closes — that
        # contract is not negotiable. Only σ-feeding ReferenceEvents change.
        if self._reference_source == "binance_bbo":
            ref_events = self._load_binance_bbo_reference(q.start_ts_ns, q.end_ts_ns)
        elif self._reference_source == "klines_1s":
            ref_events = self._load_klines_1s_reference(q.start_ts_ns, q.end_ts_ns)
        else:
            # Populate `open` so binary-market runners can use the first bar's
            # open as the canonical strike — matching the legacy `day_open_btc`
            # convention.
            ref_events = [
                ReferenceEvent(
                    ts_ns=int(k["ts_ns"]), symbol=self._reference_symbol,
                    high=float(k["high"]), low=float(k["low"]),
                    close=float(k["close"]), open=float(k["open"]),
                )
                for k in sorted(klines, key=lambda k: int(k["ts_ns"]))
            ]
        leg_event_streams.append(iter(ref_events))

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
        klines_dir = self._cache_root / self._klines_subdir
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

    def _kline_coverage(self) -> tuple[int, int] | None:
        """(first_ts_ns, last_ts_ns) spanned by the cached kline series, or
        None when the cache is empty."""
        rows = self._load_all_klines()
        if not rows:
            return None
        return int(rows[0]["ts_ns"]), int(rows[-1]["ts_ns"])

    def _ensure_kline_coverage(self, start_ns: int, end_ns: int) -> None:
        """Extend the cached Binance spot kline series forward to cover
        ``[start_ns, end_ns]`` (SHR-54 coupled fetch).

        Keeps the kline cache advancing in lockstep with the PM market cache:
        the manual-klines split was what let the cache freeze at 05-09 while
        ``fetch_and_cache`` kept pulling newer markets. No-op when the window is
        already covered, or for non-Binance underlyings (WTI uses Pyth klines).
        Only forward gaps are fetched — backfilling before the cache start is
        out of scope (markets that old aren't in the corpus).
        """
        symbol = _BINANCE_SPOT_KLINE_SYMBOL.get(self._reference_symbol)
        if symbol is None:
            return  # non-Binance underlying (e.g. WTI/Pyth) — not our series
        cov = self._kline_coverage()
        last_ns = cov[1] if cov is not None else (start_ns - _BAR_NS)
        if end_ns <= last_ns:
            return  # already covered
        fetch_start_ms = (last_ns + _BAR_NS) // 1_000_000
        fetch_end_ms = (end_ns // 1_000_000) + 60_000  # inclusive of end bar
        rows = fetch_klines(fetch_start_ms, fetch_end_ms, symbol=symbol)
        if not rows:
            logger.warning(
                f"kline coverage fetch for {symbol} returned no rows "
                f"({fetch_start_ms}..{fetch_end_ms}ms); cache unchanged"
            )
            return
        klines_dir = self._cache_root / self._klines_subdir
        klines_dir.mkdir(parents=True, exist_ok=True)
        out = [
            {"ts_ns": k.ts_ns, "open": k.open, "high": k.high,
             "low": k.low, "close": k.close, "volume": k.volume}
            for k in rows
        ]
        # Name by covered range. Fetch starts one bar past the cache end, so the
        # new file never overlaps existing ones; _load_all_klines globs + ts-sorts.
        fname = f"fetch_{out[0]['ts_ns']}_{out[-1]['ts_ns']}.json"
        (klines_dir / fname).write_text(json.dumps(out))
        self._klines_cache = None  # invalidate so the next load picks up the extension
        logger.info(
            f"extended {self._klines_subdir}/ coverage with {len(out)} {symbol} "
            f"bars up to ts={out[-1]['ts_ns']} (SHR-54 coupled fetch)"
        )

    def _load_all_klines_1s(self) -> list[dict]:
        """Load + cache all genuine Binance 1s klines from the 1s subdir.

        Separate from ``_load_all_klines`` (1m) so the canonical strike series is
        never polluted by the 1s feed. Populated on demand by
        scripts/fetch_binance_1s_klines.py."""
        if self._klines_1s_cache is not None:
            return self._klines_1s_cache
        d = self._cache_root / self._klines_1s_subdir
        rows: list[dict] = []
        if d.exists():
            for f in sorted(d.glob("*.json")):
                rows.extend(json.loads(f.read_text()))
        rows.sort(key=lambda r: int(r["ts_ns"]))
        self._klines_1s_cache = rows
        return rows

    def _load_klines_1s_reference(
        self, start_ns: int, end_ns: int,
    ) -> list[ReferenceEvent]:
        """Genuine Binance 1s klines for [start_ns, end_ns), bucketed to
        ``reference_resample_seconds`` OHLC ReferenceEvents.

        The on-demand-klines counterpart to ``_load_binance_bbo_reference``: same
        OHLC bucketing contract (high/low = bucket extremes, close = last bar's
        close, ts = last bar's ts) so a dt=5 1s-kline σ series is directly
        comparable to the recorded-BBO σ series. The H/L here are TRADE-OHLC
        extremes (vs the BBO path's quote-mid extremes) — that difference is the
        whole point of the equivalence experiment, and it flows through the
        Parkinson estimator. Half-open window [start, end) matches the BBO path.
        Returns [] when the 1s cache is empty/uncovered (gates degrade gracefully).
        """
        resample_ns = self._reference_resample_ns
        out: list[ReferenceEvent] = []
        cur_bucket: int | None = None
        o = h = l = c = 0.0
        last_ts = 0
        for k in self._load_all_klines_1s():
            ts = int(k["ts_ns"])
            if ts < start_ns or ts >= end_ns:
                continue
            kh, kl, kc, ko = float(k["high"]), float(k["low"]), float(k["close"]), float(k["open"])
            bucket = ts // resample_ns
            if cur_bucket is None:
                cur_bucket = bucket
                o, h, l, c = ko, kh, kl, kc
                last_ts = ts
            elif bucket != cur_bucket:
                out.append(ReferenceEvent(
                    ts_ns=last_ts, symbol=self._reference_symbol,
                    high=h, low=l, close=c, open=o,
                ))
                cur_bucket = bucket
                o, h, l, c = ko, kh, kl, kc
                last_ts = ts
            else:
                if kh > h:
                    h = kh
                if kl < l:
                    l = kl
                c = kc
                last_ts = ts
        if cur_bucket is not None:
            out.append(ReferenceEvent(
                ts_ns=last_ts, symbol=self._reference_symbol,
                high=h, low=l, close=c, open=o,
            ))
        return out

    def _load_binance_bbo_reference(
        self, start_ns: int, end_ns: int,
    ) -> list[ReferenceEvent]:
        """Read recorded Binance BBO ticks for [start_ns, end_ns) and
        bucket to ``reference_resample_seconds`` OHLC ReferenceEvents.

        The product_type (perp or spot) is controlled by
        ``self._binance_bbo_product_type``:

        - ``"perp"`` (default): reads ``product_type=perp`` partitions. Binance
          PERP BBO ticks carry a valid ``exchange_ts`` (nanoseconds); filtering
          and ordering use ``exchange_ts`` directly.
        - ``"spot"``: reads ``product_type=spot`` partitions. Binance SPOT
          bookTicker does NOT provide an exchange timestamp — the adapter records
          ``exchange_ts=0`` for every tick. Filtering and ordering use
          ``local_recv_ts`` instead, which is always valid.

        Mirrors the HL HIP-4 `_resample_reference` contract: each bar's high/low
        track bucket extremes, close is the bucket's last tick, ts is the
        bucket's last-tick timestamp so monotone ordering is preserved.
        Returns an empty list (caller proceeds with klines-only or no ref) if
        no partitions match the window — strategy gates degrade gracefully.
        """
        import duckdb
        sym = _BINANCE_PERP_REF_SYMBOL.get(self._reference_symbol)
        if sym is None:
            logger.warning(
                f"binance_bbo reference unsupported for symbol {self._reference_symbol!r}"
            )
            return []
        # Hive partitions are `date=YYYY-MM-DD`. Cover the question window
        # with one-day padding on each side to catch tick-boundary spills.
        start_d = datetime.fromtimestamp(start_ns / 1e9, tz=timezone.utc).date()
        end_d = datetime.fromtimestamp(end_ns / 1e9, tz=timezone.utc).date()
        date_list: list[str] = []
        d = start_d
        from datetime import timedelta
        while d <= end_d + timedelta(days=1):
            date_list.append(d.isoformat())
            d += timedelta(days=1)
        product_subpath = (
            f"venue=binance/product_type={self._binance_bbo_product_type}/mechanism=clob"
        )
        glob = str(
            self._binance_data_root
            / product_subpath
            / "event=bbo" / f"symbol={sym}"
            / "**" / "*.parquet"
        )
        from glob import glob as _glob
        if not _glob(glob, recursive=True):
            logger.warning(f"binance_bbo: no parquet at {glob}")
            return []
        con = duckdb.connect()
        # Spot exchange_ts is always 0 — use local_recv_ts for filter + order.
        # Perp exchange_ts is a valid nanosecond timestamp — use it directly.
        is_spot = self._binance_bbo_product_type == "spot"
        ts_col = "local_recv_ts" if is_spot else "exchange_ts"
        try:
            rows = con.sql(
                f"""
                SELECT {ts_col} AS ts, bid_px, ask_px
                FROM read_parquet('{glob}', hive_partitioning=1)
                WHERE date IN ({','.join(repr(d) for d in date_list)})
                  AND {ts_col} >= {start_ns} AND {ts_col} < {end_ns}
                ORDER BY {ts_col}
                """
            ).fetchall()
        finally:
            con.close()
        if not rows:
            return []
        # Bucket to OHLC. Same algorithm as HL HIP-4's `_resample_reference`.
        resample_ns = self._reference_resample_ns
        out: list[ReferenceEvent] = []
        cur_bucket: int | None = None
        h = l = c = o = 0.0
        last_ts = 0
        for ts, bid, ask in rows:
            mid = (float(bid) + float(ask)) / 2.0
            bucket = int(ts) // resample_ns
            if cur_bucket is None:
                cur_bucket = bucket
                h = l = c = o = mid
                last_ts = int(ts)
            elif bucket != cur_bucket:
                out.append(ReferenceEvent(
                    ts_ns=last_ts, symbol=self._reference_symbol,
                    high=h, low=l, close=c, open=o,
                ))
                cur_bucket = bucket
                h = l = c = o = mid
                last_ts = int(ts)
            else:
                if mid > h: h = mid
                if mid < l: l = mid
                c = mid
                last_ts = int(ts)
        out.append(ReferenceEvent(
            ts_ns=last_ts, symbol=self._reference_symbol,
            high=h, low=l, close=c, open=o,
        ))
        return out

    def _load_recorded_book(
        self, token_id: str, start_ns: int, end_ns: int,
    ) -> list[BookSnapshot]:
        """Read the recorded multi-level L2 `book_snapshot` parquet for one PM
        token leg over ``[start_ns, end_ns)`` and emit ``BookSnapshot`` events.

        Level ordering in the recorded arrays is NOT guaranteed best-first
        (observed: ``bid_px`` ascending, ``ask_px`` descending). We normalize:
        bids sorted px DESC (best = max), asks sorted px ASC (best = min), with
        each level's size carried alongside its price. Returns an empty list
        (logged) when the token has no recorded coverage — the caller then
        proceeds with no book for that leg rather than crashing.

        Mirrors the ``_load_binance_bbo_reference`` duckdb reader pattern.
        """
        import duckdb
        glob = self._recorded_book_glob(token_id)
        from glob import glob as _glob
        if not _glob(glob, recursive=True):
            logger.info(f"recorded PM book: no coverage for token {token_id}")
            return []
        con = duckdb.connect()
        try:
            rows = con.sql(
                f"""
                SELECT exchange_ts, bid_px, bid_sz, ask_px, ask_sz
                FROM read_parquet('{glob}', hive_partitioning=1)
                WHERE exchange_ts >= {start_ns} AND exchange_ts < {end_ns}
                ORDER BY exchange_ts
                """
            ).fetchall()
        finally:
            con.close()
        out: list[BookSnapshot] = []
        for ts, bid_px, bid_sz, ask_px, ask_sz in rows:
            bids = _normalize_levels(bid_px, bid_sz, descending=True)
            asks = _normalize_levels(ask_px, ask_sz, descending=False)
            out.append(BookSnapshot(
                ts_ns=int(ts), symbol=token_id, bids=bids, asks=asks,
            ))
        return out

    def _recorded_outcome(
        self, q: QuestionDescriptor,
    ) -> Literal["yes", "no", "unknown"]:
        """Resolve a binary market's winner from the recorder `settlement`
        event. The winning leg token redeems at ``settle_price≈1.0`` and only
        the winning side is published, so the leg with the higher recorded
        settle price (and ≥0.5) is the winner. Returns "unknown" when neither
        leg has settlement coverage.
        """
        if len(q.leg_symbols) < 2:
            return "unknown"
        yes_t, no_t = q.leg_symbols[0], q.leg_symbols[1]
        prices: dict[Literal["yes", "no"], float] = {}
        yp = self._leg_settle_price(yes_t)
        if yp is not None:
            prices["yes"] = yp
        np_ = self._leg_settle_price(no_t)
        if np_ is not None:
            prices["no"] = np_
        if not prices:
            return "unknown"
        winner = max(prices, key=lambda k: prices[k])
        return winner if prices[winner] >= 0.5 else "unknown"

    def _leg_settle_price(self, token_id: str) -> float | None:
        """Recorded on-chain settle price for one PM token leg, or None when
        the token has no recorder settlement coverage. token_id is unique per
        market leg, so no time window filter is needed.
        """
        import duckdb
        glob = str(
            self._pm_book_root / _PM_SETTLEMENT_DATA_SUBPATH
            / f"symbol={token_id}" / "**" / "*.parquet"
        )
        from glob import glob as _glob
        if not _glob(glob, recursive=True):
            return None
        con = duckdb.connect()
        try:
            row = con.sql(
                f"""
                SELECT settle_price
                FROM read_parquet('{glob}', hive_partitioning=1)
                WHERE settle_price IS NOT NULL
                ORDER BY settle_ts DESC
                LIMIT 1
                """
            ).fetchone()
        finally:
            con.close()
        return float(row[0]) if row and row[0] is not None else None

    # -- internals: live fetch (used by fetch_and_cache only) --------------

    def _fetch_and_cache_binary(
        self, manifest: dict, *, start_iso: str, end_iso: str,
        min_trades: int, min_volume_usd: float, refresh: bool,
    ) -> None:
        raw = _fetch_series_events(self._series_slug)
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
        raw = _fetch_series_events(self._bucket_series_slug)
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


def _normalize_levels(
    px: list[float] | None, sz: list[float] | None, *, descending: bool,
) -> tuple[tuple[float, float], ...]:
    """Pair recorded (px, sz) levels and sort by price.

    ``descending=True`` for bids (best = max), ``False`` for asks (best = min).
    Tolerates ``None``/empty/ragged arrays (one side may be empty in a recorded
    snapshot). Sizes travel with their price level.
    """
    if not px:
        return ()
    sz = sz or []
    levels = [
        (float(px[i]), float(sz[i]) if i < len(sz) else 0.0)
        for i in range(len(px))
    ]
    levels.sort(key=lambda lv: lv[0], reverse=descending)
    return tuple(levels)


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
