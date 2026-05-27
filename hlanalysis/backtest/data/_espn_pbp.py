"""ESPN play-by-play helpers for the PM NBA data source.

Two endpoints, both keyless:

    GET site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates=YYYYMMDD
    GET site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event=<gameId>

We persist normalized PBP as parquet so backtest runs don't re-hit ESPN. Each
row carries the absolute wall-clock ts (ns), period number, seconds remaining
in the current period, signed score diff (home − away), and an `is_overtime`
flag (period >= 5).
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq
import requests
from loguru import logger

_ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
_HTTP_TIMEOUT = 30


def _http_get(url: str, params: dict | None = None) -> dict:
    r = requests.get(url, params=params, timeout=_HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()


def parse_clock_to_seconds(s: str) -> int | None:
    """ESPN clock display: 'M:SS' or 'MM:SS.s'. Returns floor(seconds).
    None for empty / malformed inputs."""
    if not s or ":" not in s:
        return None
    try:
        mm, rest = s.split(":", 1)
        # Strip optional ".s" decimal tail and clip to int seconds.
        ss = rest.split(".")[0]
        return int(mm) * 60 + int(ss)
    except (ValueError, IndexError):
        return None


def _ts_ns(iso: str) -> int | None:
    if not iso:
        return None
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1e9)
    except ValueError:
        return None


def pbp_to_rows(summary: dict) -> list[dict]:
    """Normalize an ESPN summary payload's `plays[]` array into row dicts."""
    plays = summary.get("plays") or []
    out: list[dict] = []
    for p in plays:
        wc = p.get("wallclock")
        ts_ns = _ts_ns(wc)
        if ts_ns is None:
            continue
        clock_disp = ((p.get("clock") or {}).get("displayValue")) or ""
        secs = parse_clock_to_seconds(clock_disp)
        if secs is None:
            continue
        period = int((p.get("period") or {}).get("number") or 0)
        if period <= 0:
            continue
        home_score = int(p.get("homeScore") or 0)
        away_score = int(p.get("awayScore") or 0)
        out.append({
            "ts_ns": ts_ns,
            "period": period,
            "seconds_remaining_in_period": secs,
            "score_diff_home": home_score - away_score,
            "home_score": home_score,
            "away_score": away_score,
            "is_overtime": period >= 5,
        })
    # Ensure monotone ts (ESPN occasionally interleaves).
    out.sort(key=lambda r: r["ts_ns"])
    return out


def write_pbp_parquet(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        return
    table = pa.table({
        "ts_ns": [int(r["ts_ns"]) for r in rows],
        "period": [int(r["period"]) for r in rows],
        "seconds_remaining_in_period": [int(r["seconds_remaining_in_period"]) for r in rows],
        "score_diff_home": [int(r["score_diff_home"]) for r in rows],
        "home_score": [int(r["home_score"]) for r in rows],
        "away_score": [int(r["away_score"]) for r in rows],
        "is_overtime": [bool(r["is_overtime"]) for r in rows],
    })
    pq.write_table(table, path)


def read_pbp_parquet(path: Path) -> list[dict]:
    table = pq.read_table(path)
    return table.to_pylist()


def fetch_scoreboard(date_yyyymmdd: str) -> list[dict]:
    """Return a list of game dicts with keys: id, away, home, status, start_iso."""
    data = _http_get(
        f"{_ESPN_BASE}/scoreboard", params={"dates": date_yyyymmdd}
    )
    events = data.get("events") or []
    out: list[dict] = []
    for ev in events:
        try:
            comp = (ev.get("competitions") or [{}])[0]
            comps = comp.get("competitors") or []
            home = next(c for c in comps if c.get("homeAway") == "home")
            away = next(c for c in comps if c.get("homeAway") == "away")
            out.append({
                "id": str(ev["id"]),
                "away": (away.get("team") or {}).get("displayName") or "",
                "home": (home.get("team") or {}).get("displayName") or "",
                "status": ((ev.get("status") or {}).get("type") or {}).get("name") or "",
                "start_iso": ev.get("date") or "",
            })
        except (StopIteration, KeyError, TypeError):
            continue
    return out


def fetch_summary(game_id: str) -> dict:
    return _http_get(f"{_ESPN_BASE}/summary", params={"event": game_id})


_REGULATION_PERIODS = 4
_PERIOD_SECONDS = 12 * 60  # NBA quarter length


def total_regulation_seconds_remaining(*, period: int, seconds_remaining_in_period: int) -> int:
    """Total seconds left in regulation given (period, seconds_left_in_period).

    Returns 0 if we are at/past the end of regulation (period 4 with 0s left)
    OR in overtime (period >= 5). Callers gate OT off — see report §Gotchas.
    """
    if period >= _REGULATION_PERIODS + 1:
        return 0
    if period < 1 or period > _REGULATION_PERIODS:
        return 0
    return (_REGULATION_PERIODS - period) * _PERIOD_SECONDS + max(0, int(seconds_remaining_in_period))


def wp_features(
    *,
    score_diff_home: int,
    total_seconds_remaining: int,
    period: int,
) -> tuple[float, float, float]:
    """Three-feature WP input vector in canonical training order:
    (score_diff_home, log(total_seconds_remaining + 1), period_indicator).
    `period_indicator` is 1 for OT (period >= 5), 0 otherwise.
    """
    return (
        float(score_diff_home),
        math.log(max(0, total_seconds_remaining) + 1.0),
        1.0 if period >= _REGULATION_PERIODS + 1 else 0.0,
    )


__all__ = [
    "parse_clock_to_seconds",
    "pbp_to_rows",
    "write_pbp_parquet",
    "read_pbp_parquet",
    "fetch_scoreboard",
    "fetch_summary",
    "total_regulation_seconds_remaining",
    "wp_features",
]
