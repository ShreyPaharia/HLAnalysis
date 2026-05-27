"""Polymarket NBA single-game-winner data source.

Discovery: Gamma `/events?tag_slug=nba&closed=true`. We keep only 2-leg binary
"<TEAM A> vs <TEAM B>" markets; series-winner markets and 4+ way futures are
filtered. Each market is joined to an ESPN game by (resolution_date, team_pair).

PBP + WP series are pre-computed at fetch time and stored as parquet so the
backtester reads pure parquet (no network) at run time. The data source emits
`ReferenceEvent` where `close = p_yes_home` — the strategy interprets
`reference_price` as the model probability for the **home** team's leg directly.

This module currently contains only the discovery helpers. The
`PolymarketNBADataSource` class (Tasks 7-8) lives in the same file.
"""
from __future__ import annotations

import heapq
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Literal

import pyarrow as pa
import pyarrow.parquet as pq
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
from ._nba_teams import normalize_team
from ._synthetic_l2 import L2Snapshot, trade_to_l2

_VS_SPLIT = re.compile(r"\s+vs\.?\s+", flags=re.IGNORECASE)


def parse_nba_market_title(title: str) -> tuple[str, str] | None:
    """Return the two normalized 3-letter team abbreviations in title order.

    PM titles don't reliably signal which side is home vs away, so we just
    return (left, right) and let the ESPN join (in `match_pm_to_espn`) decide
    home/away by checking which ESPN home team matches which side.

    Returns ``None`` when either side fails normalization or the title isn't
    a two-team "A vs B" form.
    """
    if not title:
        return None
    parts = _VS_SPLIT.split(title)
    if len(parts) != 2:
        return None
    left = normalize_team(parts[0].strip())
    right = normalize_team(parts[1].strip())
    if left is None or right is None:
        return None
    return (left, right)


def is_series_market(title: str) -> bool:
    """Filter out playoff-series markets. PM uses 'Series Winner', 'win series',
    'X wins series 4-0', etc. for series, vs plain '<A> vs <B>' for single games."""
    if not title:
        return False
    t = title.lower()
    if "series" in t:
        return True
    if re.search(r"\bwins?\s+\d-\d\b", t):
        return True
    if re.search(r"\bin\s+\d\b", t):
        return True
    return False


def _date_prefix(iso: str) -> str:
    return (iso or "")[:10]


def match_pm_to_espn(pm_event: dict, espn_games: list[dict]) -> dict | None:
    """Match a PM event dict (with `title` + `endDate`) to one ESPN game dict.

    Strategy: same date (PM.endDate[:10] == ESPN.start_iso[:10] OR adjacent
    day for late-night games), team pair matches in either orientation.
    """
    title = pm_event.get("title") or ""
    if is_series_market(title):
        return None
    parsed = parse_nba_market_title(title)
    if parsed is None:
        return None
    pm_set = set(parsed)
    pm_date = _date_prefix(pm_event.get("endDate") or "")
    for g in espn_games:
        g_date = _date_prefix(g.get("start_iso") or "")
        if not g_date or not pm_date:
            continue
        # Accept exact-date OR one day off (late-night games span UTC midnight).
        try:
            dpm = datetime.fromisoformat(pm_date).date()
            dg = datetime.fromisoformat(g_date).date()
            if abs((dpm - dg).days) > 1:
                continue
        except ValueError:
            continue
        a = normalize_team(g.get("away") or "")
        h = normalize_team(g.get("home") or "")
        if a is None or h is None:
            continue
        if {a, h} == pm_set:
            return g
    return None


__all__ = [
    "parse_nba_market_title",
    "is_series_market",
    "match_pm_to_espn",
]

# ---------------------------------------------------------------------------
# PolymarketNBADataSource
# ---------------------------------------------------------------------------

_HALF_SPREAD_DEFAULT = 0.005
_DEPTH_DEFAULT = 10_000.0
_P_CLIP_LO = 1e-6
_P_CLIP_HI = 1.0 - 1e-6


def _question_idx(question_id: str) -> int:
    return hash(question_id) & 0x7FFFFFFF


def _ts_ns_in_iso_window(ts_ns: int, start_iso: str, end_iso: str) -> bool:
    iso_date = datetime.fromtimestamp(ts_ns / 1e9, tz=timezone.utc).strftime("%Y-%m-%d")
    return start_iso <= iso_date < end_iso


@dataclass(frozen=True, slots=True)
class _RawTrade:
    ts_ns: int
    token_id: str
    side: str
    price: float
    size: float


def _book_from_l2(s: L2Snapshot) -> BookSnapshot:
    return BookSnapshot(
        ts_ns=s.ts_ns,
        symbol=s.token_id,
        bids=((s.bid_px, s.bid_sz),),
        asks=((s.ask_px, s.ask_sz),),
    )


class PolymarketNBADataSource:
    """Cache-driven NBA-winner data source. Discovery returns descriptors from
    a manifest seeded by `fetch_and_cache(...)` (added in Task 8); `events()`
    reads cached PM trades + ESPN-derived WP series + emits per-leg settlement
    at end_ts_ns.

    Emits `ReferenceEvent` with `close = p_yes_home` — strategy reads
    `reference_price` as P(home team wins) directly. No GBM math.
    """

    name = "pm_nba"

    def __init__(
        self,
        *,
        cache_root: Path,
        half_spread: float = _HALF_SPREAD_DEFAULT,
        depth: float = _DEPTH_DEFAULT,
    ) -> None:
        self._cache_root = Path(cache_root)
        self._half_spread = half_spread
        self._depth = depth
        self._manifest_cache: dict | None = None

    # ---- DataSource protocol ---------------------------------------------

    def discover(
        self, *, start: str, end: str, **_filters: object,
    ) -> list[QuestionDescriptor]:
        manifest = self._load_manifest()
        out: list[QuestionDescriptor] = []
        for qid, entry in manifest.items():
            if entry.get("kind") != "nba_winner":
                continue
            mk = entry.get("market") or {}
            if not mk:
                continue
            descriptor = QuestionDescriptor(
                question_id=qid,
                question_idx=_question_idx(qid),
                start_ts_ns=int(mk["start_ts_ns"]),
                end_ts_ns=int(mk["end_ts_ns"]),
                leg_symbols=(str(mk["home_token_id"]), str(mk["away_token_id"])),
                klass="priceBinary",
                underlying="NBA",
            )
            if not _ts_ns_in_iso_window(descriptor.end_ts_ns, start, end):
                continue
            out.append(descriptor)
        out.sort(key=lambda d: d.start_ts_ns)
        return out

    def events(self, q: QuestionDescriptor) -> Iterator[MarketEvent]:
        manifest = self._load_manifest()
        entry = manifest.get(q.question_id) or {}
        mk = entry.get("market") or {}
        home_tok = q.leg_symbols[0]
        away_tok = q.leg_symbols[1]
        trades = self._read_trades(mk.get("condition_id") or q.question_id)
        wp_series = self._read_wp_series(mk.get("espn_game_id") or "")

        outcome = mk.get("resolved_outcome", "unknown")
        # YES = home; NO = away. So per-leg outcomes (in YES/NO terms):
        per_leg_outcomes: dict[str, Literal["yes", "no", "unknown"]] = {
            home_tok: "yes" if outcome == "home" else ("no" if outcome == "away" else "unknown"),
            away_tok: "yes" if outcome == "away" else ("no" if outcome == "home" else "unknown"),
        }

        leg_events: list[MarketEvent] = []
        for t in sorted(trades, key=lambda r: r.ts_ns):
            snap = trade_to_l2(
                ts_ns=t.ts_ns, token_id=t.token_id, price=t.price,
                half_spread=self._half_spread, depth=self._depth,
            )
            leg_events.append(_book_from_l2(snap))
            leg_events.append(TradeEvent(
                ts_ns=t.ts_ns, symbol=t.token_id,
                side=t.side, price=t.price, size=t.size,
            ))
            # Parity: emit complementary BookSnapshot at 1−p on the other leg.
            other = away_tok if t.token_id == home_tok else (
                home_tok if t.token_id == away_tok else None
            )
            if other is not None:
                comp_price = max(_P_CLIP_LO, min(_P_CLIP_HI, 1.0 - t.price))
                comp_snap = trade_to_l2(
                    ts_ns=t.ts_ns, token_id=other, price=comp_price,
                    half_spread=self._half_spread, depth=self._depth,
                )
                leg_events.append(_book_from_l2(comp_snap))

        ref_events = [
            ReferenceEvent(
                ts_ns=int(r["ts_ns"]), symbol="NBA_WP",
                # `close` carries p_yes_home; high/low/open mirror it.
                high=float(r["p_yes_home"]), low=float(r["p_yes_home"]),
                close=float(r["p_yes_home"]), open=float(r["p_yes_home"]),
            )
            for r in wp_series
        ]

        settle_events = [
            SettlementEvent(
                ts_ns=q.end_ts_ns,
                question_idx=q.question_idx,
                outcome=per_leg_outcomes.get(sym, "unknown"),
                symbol=sym,
            )
            for sym in q.leg_symbols
        ]

        yield from heapq.merge(
            iter(leg_events), iter(ref_events), iter(settle_events),
            key=lambda e: e.ts_ns,
        )

    def question_view(
        self, q: QuestionDescriptor, *, now_ns: int, settled: bool,
    ) -> QuestionView:
        manifest = self._load_manifest()
        entry = manifest.get(q.question_id) or {}
        mk = entry.get("market") or {}
        is_settled = settled or (now_ns > q.end_ts_ns)
        if is_settled:
            res = mk.get("resolved_outcome")
            side: Literal["yes", "no", "unknown"] | None = (
                "yes" if res == "home" else ("no" if res == "away" else "unknown")
            )
        else:
            side = None
        return QuestionView(
            question_idx=q.question_idx,
            yes_symbol=q.leg_symbols[0],
            no_symbol=q.leg_symbols[1],
            # No scalar strike for NBA; 0.5 is a sentinel that lets the legacy
            # binary "near-strike" diagnostic compute without div-by-zero.
            # v31_pm_nba strategy disables the gate that consumes this.
            strike=0.5,
            expiry_ns=q.end_ts_ns,
            underlying=q.underlying,
            klass=q.klass,
            period="game",
            settled=is_settled,
            settled_side=side,
            leg_symbols=q.leg_symbols,
            kv=(("home_team", str(mk.get("home_team", ""))),
                ("away_team", str(mk.get("away_team", "")))),
        )

    def resolved_outcome(self, q: QuestionDescriptor) -> Literal["yes", "no", "unknown"]:
        manifest = self._load_manifest()
        entry = manifest.get(q.question_id) or {}
        mk = entry.get("market") or {}
        res = mk.get("resolved_outcome", "unknown")
        # YES = home leg.
        if res == "home":
            return "yes"
        if res == "away":
            return "no"
        return "unknown"

    # ---- cache IO ---------------------------------------------------------

    def _manifest_path(self) -> Path:
        return self._cache_root / "manifest.json"

    def _load_manifest(self) -> dict:
        if self._manifest_cache is not None:
            return self._manifest_cache
        path = self._manifest_path()
        self._manifest_cache = json.loads(path.read_text()) if path.exists() else {}
        return self._manifest_cache

    def _read_trades(self, condition_id: str) -> list[_RawTrade]:
        path = self._cache_root / "pm_trades" / f"{condition_id}.parquet"
        if not path.exists():
            return []
        rows = pq.read_table(path).to_pylist()
        return [_RawTrade(
            ts_ns=int(r["ts_ns"]), token_id=str(r["token_id"]),
            side=str(r["side"]), price=float(r["price"]), size=float(r["size"]),
        ) for r in rows]

    def _read_wp_series(self, espn_game_id: str) -> list[dict]:
        if not espn_game_id:
            return []
        path = self._cache_root / "wp_series" / f"{espn_game_id}.parquet"
        if not path.exists():
            return []
        return pq.read_table(path).to_pylist()


__all__ += ["PolymarketNBADataSource"]
