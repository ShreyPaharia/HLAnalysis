"""Kalshi daily multi-bucket BTC `DataSource`.

Implements the §3.2 ``DataSource`` protocol against a local cache populated
by ``fetch_and_cache(...)``. Cache layout is a single ``manifest.json`` keyed
by Kalshi ``event_ticker`` plus per-market trade parquet files under
``kalshi_trades/``. BTC 1m klines are reused from the existing
``data/binance_klines/`` path (the same one the PM adapter consumes).

Design doc: ``docs/superpowers/specs/2026-05-18-kalshi-buckets-design.md``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Literal

from hlanalysis.strategy.types import QuestionView

from ..core.data_source import QuestionDescriptor
from ..core.events import MarketEvent

_HALF_SPREAD_DEFAULT = 0.005
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
        # Implemented in Task 6. For now, an empty iterator keeps `discover`
        # smoke-callable.
        return iter(())

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
