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

# ---------------------------------------------------------------------------
# fetch_and_cache — network-driven cache-population path (Task 8)
# ---------------------------------------------------------------------------

import numpy as np
import requests

from ._espn_pbp import (
    fetch_scoreboard,
    fetch_summary,
    pbp_to_rows,
    write_pbp_parquet,
    read_pbp_parquet,
    total_regulation_seconds_remaining,
    wp_features,
)

_GAMMA_BASE = "https://gamma-api.polymarket.com"
_CLOB_DATA_BASE = "https://data-api.polymarket.com"
_PAGE_LIMIT = 100
_TRADES_PAGE_SIZE = 500


def _http_get_json(url: str, params: dict | None = None) -> dict | list:
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def _fetch_nba_gamma_events(start_iso: str, end_iso: str) -> list[dict]:
    """Paginate Gamma `/events?tag_slug=nba&closed=true` and filter to events
    whose `endDate` date-prefix lies in [start_iso, end_iso)."""
    out: list[dict] = []
    offset = 0
    while True:
        try:
            page = _http_get_json(
                f"{_GAMMA_BASE}/events",
                params={
                    "tag_slug": "nba",
                    "closed": "true",
                    "limit": _PAGE_LIMIT,
                    "offset": offset,
                },
            )
        except Exception as e:
            logger.warning(f"PM NBA gamma fetch failed offset={offset}: {e}")
            break
        if not isinstance(page, list) or not page:
            break
        out.extend(page)
        if len(page) < _PAGE_LIMIT:
            break
        offset += len(page)
    return [
        ev for ev in out
        if start_iso <= (ev.get("endDate") or "")[:10] < end_iso
    ]


def _fetch_pm_trades_raw(condition_id: str) -> list[dict]:
    out: list[dict] = []
    offset = 0
    while True:
        try:
            page = _http_get_json(
                f"{_CLOB_DATA_BASE}/trades",
                params={"market": condition_id, "limit": _TRADES_PAGE_SIZE, "offset": offset},
            )
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 400 and offset > 0:
                break
            # 408 Request Timeout — PM data API times out on large markets.
            # Return what we have so far rather than crashing the entire run.
            if e.response is not None and e.response.status_code in (408, 429, 503):
                logger.warning(
                    f"PM trades fetch timeout/throttle for {condition_id} "
                    f"at offset={offset} (HTTP {e.response.status_code}); "
                    f"returning {len(out)} trades fetched so far"
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


def _load_wp_model(path: Path):
    import joblib
    return joblib.load(path)


def _build_wp_series_rows(pbp_rows: list[dict], model) -> list[dict]:
    """Run the WP model against each regulation PBP row. OT rows are excluded
    (the strategy gates OT off at evaluation time too)."""
    if not pbp_rows:
        return []
    feature_matrix: list[tuple[float, float, float]] = []
    keep_idx: list[int] = []
    keep_total: list[int] = []
    for i, r in enumerate(pbp_rows):
        if r.get("is_overtime"):
            continue
        total = total_regulation_seconds_remaining(
            period=int(r["period"]),
            seconds_remaining_in_period=int(r["seconds_remaining_in_period"]),
        )
        feature_matrix.append(wp_features(
            score_diff_home=int(r["score_diff_home"]),
            total_seconds_remaining=total,
            period=int(r["period"]),
        ))
        keep_idx.append(i)
        keep_total.append(total)
    if not feature_matrix:
        return []
    X = np.asarray(feature_matrix, dtype=np.float64)
    probs = model.predict_proba(X)[:, 1]
    out: list[dict] = []
    for p, i, total in zip(probs, keep_idx, keep_total):
        r = pbp_rows[i]
        out.append({
            "ts_ns": int(r["ts_ns"]),
            "p_yes_home": float(p),
            "score_diff_home": int(r["score_diff_home"]),
            "total_seconds_remaining": int(total),
            "period": int(r["period"]),
            "is_overtime": False,
        })
    return out


def _write_wp_series_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    pq.write_table(pa.table({
        "ts_ns": [int(r["ts_ns"]) for r in rows],
        "p_yes_home": [float(r["p_yes_home"]) for r in rows],
        "score_diff_home": [int(r["score_diff_home"]) for r in rows],
        "total_seconds_remaining": [int(r["total_seconds_remaining"]) for r in rows],
        "period": [int(r["period"]) for r in rows],
        "is_overtime": [bool(r["is_overtime"]) for r in rows],
    }), path)


def _write_pm_trades_parquet(path: Path, raw_trades: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ts, toks, sides, prices, sizes = [], [], [], [], []
    for row in raw_trades:
        try:
            ts.append(int(float(row["timestamp"]) * 1e9))
            toks.append(str(row["asset"]))
            sides.append("buy" if str(row.get("side", "")).upper() == "BUY" else "sell")
            prices.append(float(row["price"]))
            sizes.append(float(row["size"]))
        except (KeyError, ValueError, TypeError):
            continue
    pq.write_table(pa.table({
        "ts_ns": ts, "token_id": toks, "side": sides, "price": prices, "size": sizes,
    }), path)


def _parse_iso_ns_local(iso: str) -> int:
    dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1e9)


def _parse_outcomes(market: dict) -> tuple[str, str] | None:
    """Return (left_team_abbr, right_team_abbr) matching the conditional
    `outcomes`/`clobTokenIds` ordering. Both must normalize to a known team."""
    raw = market.get("outcomes")
    if not raw:
        return None
    outs = json.loads(raw) if isinstance(raw, str) else raw
    if len(outs) != 2:
        return None
    a = normalize_team(outs[0])
    b = normalize_team(outs[1])
    if a is None or b is None:
        return None
    return (a, b)


def _fetch_and_cache(
    self: "PolymarketNBADataSource",
    *,
    start: str,
    end: str,
    wp_model_path: Path,
    refresh: bool = False,
) -> list[QuestionDescriptor]:
    """Discover via Gamma, join to ESPN, persist trades + PBP + WP series."""
    manifest = self._load_manifest()
    model = _load_wp_model(wp_model_path)

    gamma_events = _fetch_nba_gamma_events(start_iso=start, end_iso=end)
    # Per-date ESPN scoreboard cache to avoid repeated hits.
    scoreboard_cache: dict[str, list[dict]] = {}

    for ev in gamma_events:
        title = ev.get("title") or ""
        if is_series_market(title):
            continue
        markets = ev.get("markets") or []
        if len(markets) != 1:
            continue
        mk_raw = markets[0]
        cond_id = mk_raw.get("conditionId") or mk_raw.get("id")
        if not cond_id:
            continue
        if str(cond_id) in manifest and not refresh:
            continue
        outs = _parse_outcomes(mk_raw)
        if outs is None:
            continue
        tok_raw = mk_raw.get("clobTokenIds")
        if not tok_raw:
            continue
        tokens = json.loads(tok_raw) if isinstance(tok_raw, str) else tok_raw
        if len(tokens) != 2:
            continue
        left_team, right_team = outs
        left_tok, right_tok = str(tokens[0]), str(tokens[1])

        # Use the EVENT-level endDate for game-date lookup. PM sometimes recycles
        # market shells from prior weeks, so mk_raw.endDate can be +7 days vs the
        # actual game date. ev.endDate is always the game's resolution date.
        # Fall back to mk_raw.endDate only when ev.endDate is absent.
        end_iso = ev.get("endDate") or mk_raw.get("endDate") or ""
        start_iso_mkt = mk_raw.get("startDate") or ev.get("startDate") or ""
        if not (end_iso and start_iso_mkt):
            continue

        date_yyyymmdd = end_iso[:10].replace("-", "")
        if date_yyyymmdd not in scoreboard_cache:
            try:
                scoreboard_cache[date_yyyymmdd] = fetch_scoreboard(date_yyyymmdd)
            except Exception as e:
                logger.warning(f"ESPN scoreboard {date_yyyymmdd} failed: {e}")
                scoreboard_cache[date_yyyymmdd] = []
        # PM endDate is when the market resolves, not game start. NBA games tip
        # off in US evening so the PM date can be ±1 UTC day vs the ESPN date.
        # Fetch prior and next day too so match_pm_to_espn's ±1-day window is
        # fully populated.
        from datetime import timedelta
        pm_dt = datetime.strptime(date_yyyymmdd, "%Y%m%d")
        prior_date = (pm_dt - timedelta(days=1)).strftime("%Y%m%d")
        next_date = (pm_dt + timedelta(days=1)).strftime("%Y%m%d")
        for adj_date in (prior_date, next_date):
            if adj_date not in scoreboard_cache:
                try:
                    scoreboard_cache[adj_date] = fetch_scoreboard(adj_date)
                except Exception as e:
                    logger.warning(f"ESPN scoreboard {adj_date} failed: {e}")
                    scoreboard_cache[adj_date] = []
        candidate_games = (
            scoreboard_cache.get(date_yyyymmdd, [])
            + scoreboard_cache.get(prior_date, [])
            + scoreboard_cache.get(next_date, [])
        )

        pm_event_for_match = {"title": title, "endDate": end_iso}
        espn_game = match_pm_to_espn(pm_event_for_match, candidate_games)
        if espn_game is None:
            logger.info(f"No ESPN match for PM event '{title}' on {date_yyyymmdd}; skipping")
            continue

        # home_token_id = the leg whose `outcomes` slot equals ESPN home team.
        espn_home_abbr = normalize_team(espn_game.get("home") or "")
        if espn_home_abbr == left_team:
            home_tok, away_tok = left_tok, right_tok
            home_team, away_team = left_team, right_team
        else:
            home_tok, away_tok = right_tok, left_tok
            home_team, away_team = right_team, left_team

        # Resolution: outcomePrices is "[<left>, <right>]" — winner is whichever side ≈ 1.0
        op = mk_raw.get("outcomePrices")
        resolved = "unknown"
        if op:
            prices = json.loads(op) if isinstance(op, str) else op
            if len(prices) == 2:
                lp, rp = float(prices[0]), float(prices[1])
                if lp >= 0.99:
                    resolved = "home" if left_team == home_team else "away"
                elif rp >= 0.99:
                    resolved = "home" if right_team == home_team else "away"

        # PM trades
        pm_trades = _fetch_pm_trades_raw(cond_id)
        _write_pm_trades_parquet(
            self._cache_root / "pm_trades" / f"{cond_id}.parquet", pm_trades
        )

        # ESPN PBP
        try:
            summary = fetch_summary(espn_game["id"])
        except Exception as e:
            logger.warning(f"ESPN summary {espn_game['id']} failed: {e}")
            continue
        pbp_rows = pbp_to_rows(summary)
        write_pbp_parquet(
            self._cache_root / "pbp" / f"{espn_game['id']}.parquet", pbp_rows
        )

        # WP series
        wp_rows = _build_wp_series_rows(pbp_rows, model)
        _write_wp_series_parquet(
            self._cache_root / "wp_series" / f"{espn_game['id']}.parquet", wp_rows
        )

        # end_ts_ns = actual game-end time (last WP row), not PM resolution
        # cutoff. PM's endDate is when the market stops accepting bets; the game
        # tips off ~10-15 min after that and ends ~2.5 h later. The strategy's
        # TTE gate computes tau_s = (expiry_ns - now_ns) and must be positive
        # during in-game play. Store pm_resolution_ts_ns separately for reference.
        pm_resolution_ns = _parse_iso_ns_local(end_iso)
        game_end_ns = int(wp_rows[-1]["ts_ns"]) if wp_rows else pm_resolution_ns

        manifest[str(cond_id)] = {
            "kind": "nba_winner",
            "n_rows": len(pm_trades),
            "market": {
                "condition_id": str(cond_id),
                "home_token_id": home_tok,
                "away_token_id": away_tok,
                "start_ts_ns": _parse_iso_ns_local(start_iso_mkt),
                "end_ts_ns": game_end_ns,
                "pm_resolution_ts_ns": pm_resolution_ns,
                "resolved_outcome": resolved,
                "total_volume_usd": float(mk_raw.get("volume") or 0.0),
                "n_trades": len(pm_trades),
                "espn_game_id": str(espn_game["id"]),
                "home_team": home_team,
                "away_team": away_team,
                "title": title,
            },
        }
        self._write_manifest(manifest)
    return self.discover(start=start, end=end)


def _write_manifest(self: "PolymarketNBADataSource", manifest: dict) -> None:
    self._cache_root.mkdir(parents=True, exist_ok=True)
    (self._cache_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    self._manifest_cache = manifest


# Attach as methods to the class. (Defined as free functions above so that
# patching their internals via `hlanalysis.backtest.data.pm_nba.X` works.)
PolymarketNBADataSource.fetch_and_cache = _fetch_and_cache  # type: ignore[attr-defined]
PolymarketNBADataSource._write_manifest = _write_manifest  # type: ignore[attr-defined]
