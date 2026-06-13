"""NBA team-name normalization for joining PM market titles to ESPN game IDs.

PM uses full names ("Boston Celtics"). ESPN's scoreboard payload exposes
`displayName` (full), `shortDisplayName` (mascot only: "Celtics"), and
`abbreviation` (three-letter: "BOS"). Each market title may use any form,
so we map them all to a single canonical 3-letter key.
"""

from __future__ import annotations

# (abbreviation, full_name, mascot_only)
NBA_TEAMS: tuple[tuple[str, str, str], ...] = (
    ("ATL", "Atlanta Hawks", "Hawks"),
    ("BOS", "Boston Celtics", "Celtics"),
    ("BKN", "Brooklyn Nets", "Nets"),
    ("CHA", "Charlotte Hornets", "Hornets"),
    ("CHI", "Chicago Bulls", "Bulls"),
    ("CLE", "Cleveland Cavaliers", "Cavaliers"),
    ("DAL", "Dallas Mavericks", "Mavericks"),
    ("DEN", "Denver Nuggets", "Nuggets"),
    ("DET", "Detroit Pistons", "Pistons"),
    ("GSW", "Golden State Warriors", "Warriors"),
    ("HOU", "Houston Rockets", "Rockets"),
    ("IND", "Indiana Pacers", "Pacers"),
    ("LAC", "LA Clippers", "Clippers"),
    ("LAL", "Los Angeles Lakers", "Lakers"),
    ("MEM", "Memphis Grizzlies", "Grizzlies"),
    ("MIA", "Miami Heat", "Heat"),
    ("MIL", "Milwaukee Bucks", "Bucks"),
    ("MIN", "Minnesota Timberwolves", "Timberwolves"),
    ("NOP", "New Orleans Pelicans", "Pelicans"),
    ("NYK", "New York Knicks", "Knicks"),
    ("OKC", "Oklahoma City Thunder", "Thunder"),
    ("ORL", "Orlando Magic", "Magic"),
    ("PHI", "Philadelphia 76ers", "76ers"),
    ("PHX", "Phoenix Suns", "Suns"),
    ("POR", "Portland Trail Blazers", "Trail Blazers"),
    ("SAC", "Sacramento Kings", "Kings"),
    ("SAS", "San Antonio Spurs", "Spurs"),
    ("TOR", "Toronto Raptors", "Raptors"),
    ("UTA", "Utah Jazz", "Jazz"),
    ("WAS", "Washington Wizards", "Wizards"),
)


def _build_index() -> dict[str, str]:
    idx: dict[str, str] = {}
    for abbr, full, mascot in NBA_TEAMS:
        for s in (abbr, full, mascot):
            idx[s.lower()] = abbr
    # Common variants the canonical names miss.
    idx["la clippers"] = "LAC"
    idx["los angeles clippers"] = "LAC"
    return idx


_INDEX = _build_index()


def normalize_team(name: str) -> str | None:
    """Return the 3-letter NBA abbreviation for any of: full name, abbreviation,
    or mascot-only short name. Case-insensitive, whitespace-tolerant. Returns
    ``None`` when the name doesn't resolve.
    """
    if not name:
        return None
    return _INDEX.get(name.strip().lower())


def team_key(away: str, home: str) -> tuple[str, str] | None:
    """Build a normalized (away_abbr, home_abbr) key for a single-game match.
    Returns ``None`` if either side can't be normalized — caller skips."""
    a = normalize_team(away)
    h = normalize_team(home)
    if a is None or h is None:
        return None
    return (a, h)


__all__ = ["NBA_TEAMS", "normalize_team", "team_key"]
