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

import re
from datetime import datetime, timezone

from ._nba_teams import normalize_team

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
