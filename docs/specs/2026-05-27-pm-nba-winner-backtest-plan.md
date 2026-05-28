# PM NBA Winner Backtest — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Backtest the v3.1 PM near-resolution arbitrage strategy on Polymarket NBA "game winner" markets, replacing the GBM reference model with a logistic win-probability (WP) model trained on prior-season ESPN play-by-play. Research only — produce a committed report. No live engine changes.

**Architecture:** New data source `pm_nba` discovers PM NBA single-game-winner markets via Gamma API and joins each to an ESPN `gameId`. ESPN play-by-play is cached as parquet. A separately-trained logistic WP model (joblib) emits `p_yes_home = WP(score_diff, log(time_remaining_seconds+1), period_indicator)` at each PBP event. The data source emits `ReferenceEvent` with `close = p_yes_home`. A new thin strategy `v31_pm_nba` reuses `ThetaHarvesterConfig` (favorite_threshold, edge_buffer, fee_model, max_position_usd, stop_loss_pct, exit_edge_threshold, topup) but bypasses the GBM path entirely: it interprets `reference_price ∈ [0,1]` as the model probability for the YES leg directly, and disables σ/τ/LM/safety_d/gamma/min_distance gates (NBA state is multi-dim, not scalar). Honest-fill instrumentation reports orderbook depth, TTR, and score_diff at each fill.

**Tech Stack:** Python 3.12, hftbacktest 2.4.4, pyarrow, scikit-learn (logistic regression), joblib, requests, loguru, pytest. Existing repo conventions: `hlanalysis/backtest/data/*.py` for sources, `hlanalysis/strategy/*.py` for strategies, `hlanalysis/backtest/cli.py` for CLI wiring, parquet caches under `data/sim/`.

---

## File Structure

**New files:**
- `hlanalysis/backtest/data/pm_nba.py` — `PolymarketNBADataSource` (Gamma NBA discovery + ESPN PBP join + WP-driven `ReferenceEvent` emission).
- `hlanalysis/backtest/data/_espn_pbp.py` — thin ESPN summary/scoreboard fetchers + parquet cache helpers (pure helpers, no DataSource interface).
- `hlanalysis/backtest/data/_nba_teams.py` — team-name normalization map (PM full name ↔ ESPN abbreviation/displayName).
- `hlanalysis/strategy/nba_wp.py` — `NBAWinProbStrategy` class + `@register("v31_pm_nba")` factory. Reuses `ThetaHarvesterConfig` but implements its own `evaluate()` (WP-driven, no GBM).
- `scripts/train_nba_wp.py` — trains the logistic WP model from prior-season PBP parquet, writes `data/nba_wp/wp_logistic.joblib` + holdout Brier to stdout.
- `tests/unit/backtest/test_nba_teams.py` — normalization unit tests.
- `tests/unit/backtest/test_pm_nba_source.py` — data-source unit tests against a hand-built fixture cache.
- `tests/unit/strategy/test_nba_wp_strategy.py` — strategy unit tests (favorite gate, edge gate, exits — bypass paths only; reuses BookState/QuestionView fixtures).
- `tests/unit/backtest/test_train_nba_wp.py` — trains on a tiny synthetic PBP fixture; asserts Brier < 0.25.
- `docs/research/pm-nba-winner-backtest.md` — final report (Task 14).

**Modified files:**
- `hlanalysis/backtest/cli.py` — register `pm_nba` data source, route to `PolymarketNBADataSource`.

**Cache layout (new, under `data/sim/pm_nba/`):**
```
data/sim/pm_nba/
  manifest.json                       # { question_id → {kind: "nba_winner", market, espn_game_id, ...} }
  pm_trades/<condition_id>.parquet    # PM CLOB trades (same schema as polymarket.py)
  pbp/<gameId>.parquet                # ESPN PBP rows (one per event)
  wp_series/<gameId>.parquet          # Pre-computed WP series (ts_ns, p_yes_home) — built at fetch time
```

**WP model location:** `data/nba_wp/wp_logistic.joblib` (one file, committed via .gitignore exception — see Task 5).

---

## Task 1: Sync to main + scaffold dirs

**Files:**
- Modify: (none — sync only)

- [ ] **Step 1: Confirm worktree is on local `main` HEAD**

Run: `git fetch origin && git status && git log --oneline -1 main && git log --oneline -1 HEAD`
Expected: HEAD == main HEAD. If main has advanced since this plan was written, rebase (`git rebase main`) before starting code work. Per memory `feedback_sync_main_before_benchmarks`, never benchmark against stale main.

- [ ] **Step 2: Create skeleton directories**

Run:
```bash
mkdir -p data/sim/pm_nba/pm_trades data/sim/pm_nba/pbp data/sim/pm_nba/wp_series \
  data/nba_wp docs/research tests/unit/strategy
```
Expected: directories created (idempotent).

- [ ] **Step 3: Commit scaffold marker**

There is nothing to commit yet. Skip to Task 2.

---

## Task 2: Team-name normalization

NBA team names appear three ways in our pipeline: PM full names ("Boston Celtics"), ESPN `displayName` ("Boston Celtics"), and ESPN `abbreviation` ("BOS"). We need bidirectional mapping plus a tolerant matcher that accepts case/whitespace variants. Hardcoding the 30-team mapping is the right call: NBA team list is stable across multi-year windows, and a one-time table is cheaper than a fuzzy-match dependency.

**Files:**
- Create: `hlanalysis/backtest/data/_nba_teams.py`
- Test: `tests/unit/backtest/test_nba_teams.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/backtest/test_nba_teams.py`:
```python
from hlanalysis.backtest.data._nba_teams import (
    normalize_team, team_key, NBA_TEAMS,
)


def test_normalize_full_name_to_key():
    assert normalize_team("Boston Celtics") == "BOS"
    assert normalize_team("boston celtics") == "BOS"
    assert normalize_team("  Boston Celtics  ") == "BOS"


def test_normalize_abbrev_passthrough():
    assert normalize_team("BOS") == "BOS"
    assert normalize_team("bos") == "BOS"


def test_normalize_espn_short_name():
    # ESPN's "shortDisplayName" form
    assert normalize_team("Celtics") == "BOS"


def test_team_key_unknown_returns_none():
    assert normalize_team("Atlantis Sharks") is None


def test_nba_teams_has_30():
    assert len(NBA_TEAMS) == 30


def test_team_key_helper_combines_two_teams():
    # Used to build a (date, away, home) → game match key
    assert team_key("Boston Celtics", "Los Angeles Lakers") == ("BOS", "LAL")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/backtest/test_nba_teams.py -v`
Expected: FAIL with `ImportError` / `ModuleNotFoundError`.

- [ ] **Step 3: Implement normalization**

`hlanalysis/backtest/data/_nba_teams.py`:
```python
"""NBA team-name normalization for joining PM market titles to ESPN game IDs.

PM uses full names ("Boston Celtics"). ESPN's scoreboard payload exposes
`displayName` (full), `shortDisplayName` (mascot only: "Celtics"), and
`abbreviation` (three-letter: "BOS"). Each market title may use any form,
so we map them all to a single canonical 3-letter key.
"""
from __future__ import annotations

from typing import Iterable

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/backtest/test_nba_teams.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

Run:
```bash
git add hlanalysis/backtest/data/_nba_teams.py tests/unit/backtest/test_nba_teams.py
git commit -m "feat(backtest): add NBA team-name normalization for PM↔ESPN join"
```

---

## Task 3: ESPN PBP fetcher + parquet cache helpers

Pure helpers — no DataSource interface yet. The PM NBA data source (Task 6) will call these.

ESPN endpoints (no API key):
- `https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates=YYYYMMDD` → list of games on that date with IDs, home/away, status.
- `https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event=<gameId>` → per-game payload including `plays[]` (one per event with `clock.displayValue`, `period.number`, `homeScore`, `awayScore`, `wallclock` in ISO).

We normalize each play to: `(ts_ns, period, seconds_remaining_in_period, score_diff_home_minus_away, is_overtime)`. Store as parquet.

**Files:**
- Create: `hlanalysis/backtest/data/_espn_pbp.py`
- Test: `tests/unit/backtest/test_pm_nba_source.py` (also covers Task 6; we add ESPN-helper tests here)

- [ ] **Step 1: Write the failing test (ESPN helpers section only — file gets extended in later tasks)**

`tests/unit/backtest/test_pm_nba_source.py` — initial content:
```python
from pathlib import Path
import json

from hlanalysis.backtest.data._espn_pbp import (
    parse_clock_to_seconds,
    pbp_to_rows,
    write_pbp_parquet,
    read_pbp_parquet,
)


def test_parse_clock_basic():
    # Standard "MM:SS" format
    assert parse_clock_to_seconds("12:00") == 720
    assert parse_clock_to_seconds("0:35") == 35
    assert parse_clock_to_seconds("11:59.9") == 11 * 60 + 59  # ESPN sometimes adds decimals; we floor


def test_parse_clock_empty_or_bad():
    assert parse_clock_to_seconds("") is None
    assert parse_clock_to_seconds("garbage") is None


def test_pbp_to_rows_basic():
    summary = {
        "plays": [
            {
                "clock": {"displayValue": "12:00"},
                "period": {"number": 1, "displayValue": "1st Quarter"},
                "homeScore": 0, "awayScore": 0,
                "wallclock": "2024-12-01T19:30:00Z",
            },
            {
                "clock": {"displayValue": "5:00"},
                "period": {"number": 2},
                "homeScore": 30, "awayScore": 25,
                "wallclock": "2024-12-01T20:10:00Z",
            },
            {
                "clock": {"displayValue": "0:00"},
                "period": {"number": 5},
                "homeScore": 110, "awayScore": 110,
                "wallclock": "2024-12-01T22:05:00Z",
            },
        ]
    }
    rows = pbp_to_rows(summary)
    assert len(rows) == 3
    assert rows[0]["period"] == 1
    assert rows[0]["seconds_remaining_in_period"] == 720
    assert rows[0]["score_diff_home"] == 0
    assert rows[0]["is_overtime"] is False
    assert rows[1]["score_diff_home"] == 5
    assert rows[2]["is_overtime"] is True
    # Monotone increasing ts_ns
    assert rows[0]["ts_ns"] < rows[1]["ts_ns"] < rows[2]["ts_ns"]


def test_pbp_parquet_roundtrip(tmp_path: Path):
    rows = [
        {"ts_ns": 1_700_000_000_000_000_000, "period": 1,
         "seconds_remaining_in_period": 720, "score_diff_home": 0,
         "is_overtime": False, "home_score": 0, "away_score": 0},
        {"ts_ns": 1_700_000_300_000_000_000, "period": 1,
         "seconds_remaining_in_period": 600, "score_diff_home": 4,
         "is_overtime": False, "home_score": 8, "away_score": 4},
    ]
    out = tmp_path / "401584893.parquet"
    write_pbp_parquet(out, rows)
    back = read_pbp_parquet(out)
    assert len(back) == 2
    assert back[0]["score_diff_home"] == 0
    assert back[1]["score_diff_home"] == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/backtest/test_pm_nba_source.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement ESPN helpers**

`hlanalysis/backtest/data/_espn_pbp.py`:
```python
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


__all__ = [
    "parse_clock_to_seconds",
    "pbp_to_rows",
    "write_pbp_parquet",
    "read_pbp_parquet",
    "fetch_scoreboard",
    "fetch_summary",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/backtest/test_pm_nba_source.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add hlanalysis/backtest/data/_espn_pbp.py tests/unit/backtest/test_pm_nba_source.py
git commit -m "feat(backtest): add ESPN PBP fetcher + parquet cache helpers"
```

---

## Task 4: Time-remaining + state featurizer for WP

NBA regulation: 4 × 12 min = 2880 s. Each OT period: 5 min = 300 s. Strategy/data-source needs `total_seconds_remaining_in_regulation` at any tick — strictly positive in regulation, 0 at end of regulation, and we explicitly **gate off OT** (Task 11) so OT seconds aren't fed to the model.

**Files:**
- Modify: `hlanalysis/backtest/data/_espn_pbp.py`
- Test: `tests/unit/backtest/test_pm_nba_source.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/backtest/test_pm_nba_source.py`:
```python
from hlanalysis.backtest.data._espn_pbp import (
    total_regulation_seconds_remaining,
    wp_features,
)


def test_total_remaining_regulation_start():
    # Tipoff: period 1, 12:00 left → 4 × 12min = 2880 s
    assert total_regulation_seconds_remaining(period=1, seconds_remaining_in_period=720) == 2880


def test_total_remaining_q4_end():
    # End of Q4 → 0
    assert total_regulation_seconds_remaining(period=4, seconds_remaining_in_period=0) == 0


def test_total_remaining_q3_halftime_to_quarter():
    # Start of Q3, 12:00 → 2 quarters × 12 min = 1440 s
    assert total_regulation_seconds_remaining(period=3, seconds_remaining_in_period=720) == 1440


def test_total_remaining_overtime_is_zero():
    # In OT (period 5+) regulation is over → 0
    assert total_regulation_seconds_remaining(period=5, seconds_remaining_in_period=300) == 0


def test_wp_features_shape():
    feats = wp_features(score_diff_home=5, total_seconds_remaining=600, period=3)
    # Exactly three features in canonical order
    assert len(feats) == 3
    # Feature 0: score_diff_home
    assert feats[0] == 5
    # Feature 1: log(total_seconds_remaining + 1)
    import math
    assert abs(feats[1] - math.log(601)) < 1e-9
    # Feature 2: period_indicator (1 for OT, 0 otherwise)
    assert feats[2] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/backtest/test_pm_nba_source.py -v`
Expected: FAIL (5 new failures, ImportError on `total_regulation_seconds_remaining`).

- [ ] **Step 3: Implement helpers**

Append to `hlanalysis/backtest/data/_espn_pbp.py`:
```python
import math

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


__all__ += ["total_regulation_seconds_remaining", "wp_features"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/backtest/test_pm_nba_source.py -v`
Expected: PASS (8 tests total).

- [ ] **Step 5: Commit**

```bash
git add hlanalysis/backtest/data/_espn_pbp.py tests/unit/backtest/test_pm_nba_source.py
git commit -m "feat(backtest): add NBA WP feature extraction + regulation-time helper"
```

---

## Task 5: Train WP model script

A trained `LogisticRegression` (sklearn) on three features. Training corpus: ESPN PBP for the **prior** NBA season(s) — for backtest windows in 2024-25, train only on 2023-24 (and earlier if available) to avoid future leakage. The script accepts a glob of cached PBP parquets + per-game outcomes (we can derive `home_team_won` from the last row's `score_diff_home > 0` when not in OT; in OT, use the final `home_score > away_score`).

**Files:**
- Create: `scripts/train_nba_wp.py`
- Test: `tests/unit/backtest/test_train_nba_wp.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/backtest/test_train_nba_wp.py`:
```python
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.train_nba_wp import (
    build_training_frame,
    fit_logistic_wp,
)


def _write_pbp(path: Path, rows: list[dict]) -> None:
    table = pa.table({
        "ts_ns": [r["ts_ns"] for r in rows],
        "period": [r["period"] for r in rows],
        "seconds_remaining_in_period": [r["seconds_remaining_in_period"] for r in rows],
        "score_diff_home": [r["score_diff_home"] for r in rows],
        "home_score": [r["home_score"] for r in rows],
        "away_score": [r["away_score"] for r in rows],
        "is_overtime": [r["is_overtime"] for r in rows],
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def _synthetic_game(start_ts_ns: int, home_wins: bool, leader_margin: int) -> list[dict]:
    """Build a 4-quarter game ending with abs(margin)=leader_margin in favour of
    home_wins. Simplistic: linear score progression."""
    out: list[dict] = []
    sign = 1 if home_wins else -1
    secs_per_play = 30
    ts = start_ts_ns
    for q in range(1, 5):
        for s in range(720, 0, -secs_per_play):
            t_total_remaining = (4 - q) * 720 + s
            # Final score diff = sign * leader_margin; linearly interpolate.
            frac_done = 1.0 - t_total_remaining / 2880.0
            sd = int(round(sign * leader_margin * frac_done))
            out.append({
                "ts_ns": ts,
                "period": q,
                "seconds_remaining_in_period": s,
                "score_diff_home": sd,
                "home_score": max(sd, 0) + 50,
                "away_score": max(-sd, 0) + 50,
                "is_overtime": False,
            })
            ts += secs_per_play * 1_000_000_000
    return out


def test_build_training_frame_excludes_overtime(tmp_path: Path):
    rows = [
        {"ts_ns": 1, "period": 1, "seconds_remaining_in_period": 720,
         "score_diff_home": 0, "home_score": 0, "away_score": 0, "is_overtime": False},
        # OT row — must be excluded from training
        {"ts_ns": 2, "period": 5, "seconds_remaining_in_period": 300,
         "score_diff_home": 0, "home_score": 110, "away_score": 110, "is_overtime": True},
    ]
    _write_pbp(tmp_path / "001.parquet", rows)
    X, y = build_training_frame([tmp_path / "001.parquet"])
    # Only the 1 regulation row should survive.
    assert X.shape == (1, 3)
    assert y.shape == (1,)


def test_fit_logistic_wp_separable(tmp_path: Path):
    """8 games where huge home leads → home wins; huge away leads → away wins.
    Logistic should fit cleanly and produce Brier well under 0.25 on a
    held-out sample of the same structure."""
    np.random.seed(42)
    paths: list[Path] = []
    for i in range(8):
        home_wins = (i % 2 == 0)
        margin = 20
        rows = _synthetic_game(i * 10**10, home_wins=home_wins, leader_margin=margin)
        p = tmp_path / f"g{i:03d}.parquet"
        _write_pbp(p, rows)
        paths.append(p)
    X, y = build_training_frame(paths)
    model, brier = fit_logistic_wp(X, y, holdout_fraction=0.2, random_state=0)
    assert brier < 0.25
    # Sanity: predicting home-team-leading-by-15-with-1min-left should give p > 0.7
    from hlanalysis.backtest.data._espn_pbp import wp_features
    feats = np.array([wp_features(score_diff_home=15, total_seconds_remaining=60, period=4)])
    assert float(model.predict_proba(feats)[0][1]) > 0.7
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/backtest/test_train_nba_wp.py -v`
Expected: FAIL with `ModuleNotFoundError: scripts.train_nba_wp`.

- [ ] **Step 3: Implement training script**

`scripts/train_nba_wp.py`:
```python
"""Train a 3-feature logistic WP model on ESPN PBP parquet files.

Usage:
    uv run python -m scripts.train_nba_wp \\
        --pbp-glob 'data/sim/pm_nba/pbp_train/*.parquet' \\
        --out data/nba_wp/wp_logistic.joblib

Features (canonical order, matches `_espn_pbp.wp_features`):
    [score_diff_home, log(total_seconds_remaining + 1), period_indicator]

OT rows are dropped from training — strategy gates OT off at backtest time.
Target: `home_team_won` = (final home_score > final away_score). Determined
from the **last** PBP row per game (whichever team is leading at game end,
including OT).

Holdout: chronological 80/20 split on file order (filename should be sortable
chronologically; ESPN gameIds are monotone-ish, callers can sort by start
date). Reports Brier on the holdout to stdout.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pyarrow.parquet as pq
from loguru import logger

from hlanalysis.backtest.data._espn_pbp import (
    total_regulation_seconds_remaining,
    wp_features,
)


def _label_from_last_row(rows: list[dict]) -> int | None:
    """home_team_won determined from the LAST row's score diff (handles OT)."""
    if not rows:
        return None
    last = rows[-1]
    return 1 if int(last.get("score_diff_home", 0)) > 0 else 0


def _game_rows(path: Path) -> list[dict]:
    table = pq.read_table(path)
    return table.to_pylist()


def build_training_frame(
    paths: Iterable[Path],
) -> tuple[np.ndarray, np.ndarray]:
    """Iterate PBP parquets; for each game emit one (X, y) row per regulation
    PBP play. Returns (n_total_plays, 3) float64 X and (n_total_plays,) int y.
    OT rows are excluded."""
    X_rows: list[tuple[float, float, float]] = []
    y_rows: list[int] = []
    for path in paths:
        rows = _game_rows(path)
        if not rows:
            continue
        label = _label_from_last_row(rows)
        if label is None:
            continue
        for r in rows:
            if r.get("is_overtime"):
                continue
            period = int(r["period"])
            secs_in_period = int(r["seconds_remaining_in_period"])
            sd = int(r["score_diff_home"])
            total = total_regulation_seconds_remaining(
                period=period, seconds_remaining_in_period=secs_in_period
            )
            X_rows.append(wp_features(
                score_diff_home=sd,
                total_seconds_remaining=total,
                period=period,
            ))
            y_rows.append(label)
    if not X_rows:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0,), dtype=np.int64)
    X = np.asarray(X_rows, dtype=np.float64)
    y = np.asarray(y_rows, dtype=np.int64)
    return X, y


def fit_logistic_wp(
    X: np.ndarray,
    y: np.ndarray,
    *,
    holdout_fraction: float = 0.2,
    random_state: int = 0,
):
    """Chronological 80/20 split. Returns (fitted_model, holdout_brier)."""
    # Lazy import keeps the lightweight tests / data-source imports fast.
    from sklearn.linear_model import LogisticRegression

    n = len(X)
    if n < 4:
        raise ValueError(f"Need >= 4 training rows, got {n}")
    split = int(n * (1.0 - holdout_fraction))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(X_train, y_train)
    p_test = model.predict_proba(X_test)[:, 1]
    brier = float(np.mean((p_test - y_test) ** 2))
    return model, brier


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pbp-glob", required=True,
                    help="Glob for PBP parquet files (prior-season only — no future leakage).")
    ap.add_argument("--out", required=True, help="Output joblib path.")
    ap.add_argument("--holdout", type=float, default=0.2)
    args = ap.parse_args(argv)

    paths = sorted(Path().glob(args.pbp_glob))
    if not paths:
        logger.error(f"No PBP parquets matched {args.pbp_glob!r}")
        return 2
    X, y = build_training_frame(paths)
    if X.shape[0] == 0:
        logger.error("Training frame is empty (all games filtered).")
        return 2
    model, brier = fit_logistic_wp(X, y, holdout_fraction=args.holdout)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)
    logger.info(
        f"Trained on {X.shape[0]} plays from {len(paths)} games. "
        f"Holdout Brier: {brier:.4f}. Wrote {out_path}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/backtest/test_train_nba_wp.py -v`
Expected: PASS (2 tests). If Brier is brittle on the tiny synthetic corpus, raise the bound to `< 0.30` — but the deterministic separable structure should make < 0.25 reliable.

- [ ] **Step 5: Commit**

```bash
git add scripts/train_nba_wp.py tests/unit/backtest/test_train_nba_wp.py
git commit -m "feat(scripts): add 3-feature logistic NBA WP training script"
```

---

## Task 6: PM NBA market discovery + ESPN game-ID join

PM exposes single-game NBA winner markets via Gamma. Likely identifiers:
- Tag `nba` and/or category `Sports`.
- Game-winner events typically named `<HOME> vs <AWAY>` with 2-leg binary markets (YES = pick one team, NO = pick the other).

We discover via the same paginated `/events` route used for BTC. We then match each PM event to an ESPN game by `(date, home_abbr, away_abbr)`, filtering out playoff series ("X wins series in Y games") and futures.

**Files:**
- Create: `hlanalysis/backtest/data/pm_nba.py` (initial skeleton + discovery)
- Test: `tests/unit/backtest/test_pm_nba_source.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/backtest/test_pm_nba_source.py`:
```python
from hlanalysis.backtest.data.pm_nba import (
    parse_nba_market_title,
    is_series_market,
    match_pm_to_espn,
)


def test_parse_winner_title_simple():
    # PM canonical form: "Lakers vs. Celtics" or "Boston Celtics vs Los Angeles Lakers"
    teams = parse_nba_market_title("Boston Celtics vs Los Angeles Lakers")
    assert teams == ("BOS", "LAL")


def test_parse_winner_title_with_period():
    teams = parse_nba_market_title("Lakers vs. Celtics")
    assert teams == ("LAL", "BOS")


def test_parse_winner_title_unknown_team_returns_none():
    assert parse_nba_market_title("Lakers vs Atlantis Sharks") is None


def test_is_series_market_filters_playoff_series():
    assert is_series_market("Celtics vs Heat: Series Winner") is True
    assert is_series_market("Celtics win series 4-0") is True
    assert is_series_market("Boston Celtics vs Los Angeles Lakers") is False


def test_match_pm_to_espn_by_date_and_teams():
    pm_event = {
        "title": "Boston Celtics vs Los Angeles Lakers",
        # PM endDate is the resolution time — game end. We match on the date prefix.
        "endDate": "2024-12-25T22:30:00Z",
    }
    espn_games = [
        {"id": "401584900", "away": "Boston Celtics", "home": "Los Angeles Lakers",
         "status": "STATUS_FINAL", "start_iso": "2024-12-25T20:00:00Z"},
        {"id": "401584901", "away": "Knicks", "home": "Heat",
         "status": "STATUS_FINAL", "start_iso": "2024-12-25T19:00:00Z"},
    ]
    match = match_pm_to_espn(pm_event, espn_games)
    assert match is not None
    assert match["id"] == "401584900"


def test_match_pm_to_espn_returns_none_when_no_team_match():
    pm_event = {
        "title": "Boston Celtics vs Los Angeles Lakers",
        "endDate": "2024-12-25T22:30:00Z",
    }
    espn_games = [
        {"id": "401584901", "away": "Knicks", "home": "Heat",
         "status": "STATUS_FINAL", "start_iso": "2024-12-25T19:00:00Z"},
    ]
    assert match_pm_to_espn(pm_event, espn_games) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/backtest/test_pm_nba_source.py -v`
Expected: FAIL (ModuleNotFoundError: pm_nba).

- [ ] **Step 3: Implement title parser + matcher (no DataSource interface yet — just the parsing helpers)**

Create `hlanalysis/backtest/data/pm_nba.py`:
```python
"""Polymarket NBA single-game-winner data source.

Discovery: Gamma `/events?tag_slug=nba&closed=true`. We keep only 2-leg binary
"<TEAM A> vs <TEAM B>" markets; series-winner markets and 4+ way futures are
filtered. Each market is joined to an ESPN game by (resolution_date, team_pair).

PBP + WP series are pre-computed at fetch time and stored as parquet so the
backtester reads pure parquet (no network) at run time. The data source emits
`ReferenceEvent` where `close = p_yes_home` — the strategy interprets
`reference_price` as the model probability for the **home** team's leg directly.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

from ._nba_teams import normalize_team

_SERIES_KEYWORDS = ("series", "wins series", "in [0-9]")
_VS_SPLIT = re.compile(r"\s+vs\.?\s+", flags=re.IGNORECASE)


def parse_nba_market_title(title: str) -> tuple[str, str] | None:
    """Return (away_abbr, home_abbr) — but PM titles don't reliably signal
    which is home/away, so we just return the two normalized abbreviations
    in title order. Caller resolves home/away from the ESPN join.

    For consistency with downstream code we name the tuple (left, right)
    in title order. The home/away assignment is done in `match_pm_to_espn`
    by checking which ESPN home team matches which side.
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/backtest/test_pm_nba_source.py -v`
Expected: PASS (13 tests total — 8 from earlier + 5 new).

- [ ] **Step 5: Commit**

```bash
git add hlanalysis/backtest/data/pm_nba.py tests/unit/backtest/test_pm_nba_source.py
git commit -m "feat(backtest): add PM NBA market discovery + ESPN game-ID join helpers"
```

---

## Task 7: PolymarketNBADataSource — discover + fetch_and_cache

Implements the `DataSource` protocol. Layout under `data/sim/pm_nba/` mirrors `polymarket.py` for trades + adds `pbp/<gameId>.parquet` and `wp_series/<gameId>.parquet`.

A subtlety: which PM token is the YES (home) leg? PM events for NBA games have two `outcomes` named after the two teams (e.g. `["Lakers", "Celtics"]`) — the ordering is **stable per market**, but which one corresponds to "home" depends on the ESPN match. At fetch time we determine `home_token_id` and `away_token_id` and persist them in the manifest. The strategy's reference probability is always P(home wins).

**Files:**
- Modify: `hlanalysis/backtest/data/pm_nba.py`
- Test: `tests/unit/backtest/test_pm_nba_source.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/backtest/test_pm_nba_source.py`:
```python
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from hlanalysis.backtest.data.pm_nba import PolymarketNBADataSource


def _seed_pm_nba_cache(cache_root: Path) -> str:
    """Build a minimal fixture cache for one game. Returns the question_id."""
    cache_root.mkdir(parents=True, exist_ok=True)
    cond_id = "0xabc123"
    game_id = "401584900"

    # Manifest entry
    manifest = {
        cond_id: {
            "kind": "nba_winner",
            "n_rows": 4,
            "market": {
                "condition_id": cond_id,
                "home_token_id": "TOK_HOME",
                "away_token_id": "TOK_AWAY",
                "start_ts_ns": 1_700_000_000_000_000_000,
                "end_ts_ns": 1_700_000_010_000_000_000,
                "resolved_outcome": "home",
                "total_volume_usd": 5000.0,
                "n_trades": 2,
                "espn_game_id": game_id,
                "home_team": "LAL",
                "away_team": "BOS",
                "title": "Boston Celtics vs Los Angeles Lakers",
            },
        }
    }
    (cache_root / "manifest.json").write_text(json.dumps(manifest))

    # PM trades fixture (two trades on home leg)
    pm_trades_dir = cache_root / "pm_trades"
    pm_trades_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table({
            "ts_ns": [1_700_000_001_000_000_000, 1_700_000_005_000_000_000],
            "token_id": ["TOK_HOME", "TOK_HOME"],
            "side": ["buy", "buy"],
            "price": [0.55, 0.62],
            "size": [100.0, 50.0],
        }),
        pm_trades_dir / f"{cond_id}.parquet",
    )

    # WP series fixture (4 monotone ticks during the game)
    wp_dir = cache_root / "wp_series"
    wp_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table({
            "ts_ns": [
                1_700_000_000_500_000_000,
                1_700_000_002_000_000_000,
                1_700_000_006_000_000_000,
                1_700_000_009_000_000_000,
            ],
            "p_yes_home": [0.50, 0.58, 0.71, 0.95],
            "score_diff_home": [0, 5, 12, 22],
            "total_seconds_remaining": [2880, 2400, 600, 30],
            "period": [1, 1, 4, 4],
            "is_overtime": [False, False, False, False],
        }),
        wp_dir / f"{game_id}.parquet",
    )
    return cond_id


def test_discover_returns_descriptor(tmp_path: Path):
    cache_root = tmp_path / "pm_nba"
    cond_id = _seed_pm_nba_cache(cache_root)
    ds = PolymarketNBADataSource(cache_root=cache_root)
    descs = ds.discover(start="2023-11-01", end="2024-01-01")
    assert len(descs) == 1
    d = descs[0]
    assert d.question_id == cond_id
    assert d.klass == "priceBinary"
    # Canonical leg order: YES (home) first, NO (away) second.
    assert d.leg_symbols == ("TOK_HOME", "TOK_AWAY")
    assert d.underlying == "NBA"


def test_events_emits_book_trade_reference_and_settlement(tmp_path: Path):
    cache_root = tmp_path / "pm_nba"
    cond_id = _seed_pm_nba_cache(cache_root)
    ds = PolymarketNBADataSource(cache_root=cache_root)
    descs = ds.discover(start="2023-11-01", end="2024-01-01")
    events = list(ds.events(descs[0]))

    from hlanalysis.backtest.core.events import (
        BookSnapshot, TradeEvent, ReferenceEvent, SettlementEvent,
    )

    n_book = sum(1 for e in events if isinstance(e, BookSnapshot))
    n_trade = sum(1 for e in events if isinstance(e, TradeEvent))
    n_ref = sum(1 for e in events if isinstance(e, ReferenceEvent))
    n_settle = sum(1 for e in events if isinstance(e, SettlementEvent))
    # 2 PM trades → 2 BookSnapshots (home) + 2 parity-inferred (away) + 2 TradeEvents
    assert n_book == 4
    assert n_trade == 2
    # WP series has 4 ticks → 4 ReferenceEvents
    assert n_ref == 4
    # 2 legs → 2 settlement events
    assert n_settle == 2

    # Reference event payload — `close` carries p_yes_home directly.
    ref_events = [e for e in events if isinstance(e, ReferenceEvent)]
    assert abs(ref_events[0].close - 0.50) < 1e-9
    assert abs(ref_events[-1].close - 0.95) < 1e-9


def test_question_view_strike_neutral(tmp_path: Path):
    """NBA markets have no scalar strike. The data source should return a
    QuestionView with strike=0.5 (neutral) so the GBM near-strike gate would
    never block — but the v31_pm_nba strategy disables that gate anyway."""
    cache_root = tmp_path / "pm_nba"
    cond_id = _seed_pm_nba_cache(cache_root)
    ds = PolymarketNBADataSource(cache_root=cache_root)
    descs = ds.discover(start="2023-11-01", end="2024-01-01")
    qv = ds.question_view(descs[0], now_ns=descs[0].start_ts_ns, settled=False)
    assert qv.klass == "priceBinary"
    assert qv.yes_symbol == "TOK_HOME"
    assert qv.no_symbol == "TOK_AWAY"
    # Strike = 0.5 sentinel — no scalar reference price exists for NBA.
    assert qv.strike == 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/backtest/test_pm_nba_source.py -v`
Expected: 3 new failures (NameError: PolymarketNBADataSource).

- [ ] **Step 3: Implement the data source**

Append to `hlanalysis/backtest/data/pm_nba.py`:
```python
import heapq
import json
from dataclasses import dataclass
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
from ._synthetic_l2 import L2Snapshot, trade_to_l2

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


def _flip(outcome: str) -> Literal["home", "away", "unknown"]:
    if outcome == "home":
        return "away"
    if outcome == "away":
        return "home"
    return "unknown"  # type: ignore[return-value]


class PolymarketNBADataSource:
    """Cache-driven NBA-winner data source. Discovery returns descriptors from
    a manifest seeded by `fetch_and_cache(...)`; `events()` reads cached PM
    trades + ESPN-derived WP series + emits per-leg settlement at end_ts_ns.
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
                # `close` carries p_yes_home; high/low/open mirror it (single-point series).
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
```

Add the missing import at the top of `pm_nba.py`:
```python
from datetime import datetime, timezone
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/backtest/test_pm_nba_source.py -v`
Expected: PASS (16 tests total).

- [ ] **Step 5: Commit**

```bash
git add hlanalysis/backtest/data/pm_nba.py tests/unit/backtest/test_pm_nba_source.py
git commit -m "feat(backtest): add PolymarketNBADataSource (discover + events + cache IO)"
```

---

## Task 8: fetch_and_cache (Gamma + ESPN + WP) — live discovery wired to cache builder

This is the heavyweight method that actually hits Gamma + ESPN APIs. Keep `fetch_and_cache` slim (delegates to helpers) and **NOT** unit-tested against the real network — instead, a unit test exercises the *cache-write* path with mocked HTTP responses.

**Files:**
- Modify: `hlanalysis/backtest/data/pm_nba.py`
- Test: `tests/unit/backtest/test_pm_nba_source.py`

- [ ] **Step 1: Write the failing test (mock HTTP)**

Append to `tests/unit/backtest/test_pm_nba_source.py`:
```python
from unittest.mock import patch


def test_fetch_and_cache_writes_manifest_and_trades(tmp_path, monkeypatch):
    """fetch_and_cache wires together Gamma `/events`, PM trades, ESPN scoreboard +
    summary, WP-model inference. We mock the four network calls and verify the
    cache layout."""
    cache_root = tmp_path / "pm_nba_live"
    cache_root.mkdir(parents=True)

    gamma_events = [{
        "title": "Boston Celtics vs Los Angeles Lakers",
        "endDate": "2024-12-25T22:30:00Z",
        "startDate": "2024-12-25T20:00:00Z",
        "markets": [{
            "conditionId": "0xfeedbeef",
            "clobTokenIds": '["TOK_LEFT","TOK_RIGHT"]',
            "outcomes": '["Boston Celtics","Los Angeles Lakers"]',
            "outcomePrices": '["1","0"]',
            "startDate": "2024-12-25T20:00:00Z",
            "endDate": "2024-12-25T22:30:00Z",
            "volume": 12000.0,
        }],
    }]
    pm_trades = [
        {"timestamp": 1735156800.0, "asset": "TOK_LEFT", "side": "BUY",
         "price": 0.45, "size": 100.0},
        {"timestamp": 1735157000.0, "asset": "TOK_LEFT", "side": "BUY",
         "price": 0.50, "size": 50.0},
    ]
    espn_scoreboard = [{
        "id": "401584900", "home": "Boston Celtics", "away": "Los Angeles Lakers",
        "status": "STATUS_FINAL", "start_iso": "2024-12-25T20:00:00Z",
    }]
    espn_summary = {
        "plays": [
            {"clock": {"displayValue": "12:00"}, "period": {"number": 1},
             "homeScore": 0, "awayScore": 0,
             "wallclock": "2024-12-25T20:00:00Z"},
            {"clock": {"displayValue": "0:00"}, "period": {"number": 4},
             "homeScore": 110, "awayScore": 100,
             "wallclock": "2024-12-25T22:30:00Z"},
        ]
    }

    # Stub a trivial WP model: returns 0.5 always.
    class _StubModel:
        def predict_proba(self, X):
            import numpy as np
            return np.array([[0.5, 0.5]] * len(X))

    with patch(
        "hlanalysis.backtest.data.pm_nba._fetch_nba_gamma_events",
        return_value=gamma_events,
    ), patch(
        "hlanalysis.backtest.data.pm_nba._fetch_pm_trades_raw",
        return_value=pm_trades,
    ), patch(
        "hlanalysis.backtest.data.pm_nba.fetch_scoreboard",
        return_value=espn_scoreboard,
    ), patch(
        "hlanalysis.backtest.data.pm_nba.fetch_summary",
        return_value=espn_summary,
    ), patch(
        "hlanalysis.backtest.data.pm_nba._load_wp_model",
        return_value=_StubModel(),
    ):
        ds = PolymarketNBADataSource(cache_root=cache_root)
        descs = ds.fetch_and_cache(
            start="2024-12-01", end="2024-12-31",
            wp_model_path=Path("data/nba_wp/wp_logistic.joblib"),
        )

    assert len(descs) == 1
    # Verify cache layout
    assert (cache_root / "manifest.json").exists()
    manifest = json.loads((cache_root / "manifest.json").read_text())
    assert "0xfeedbeef" in manifest
    mk = manifest["0xfeedbeef"]["market"]
    # Home team (Boston) matches the ESPN home field; PM outcomes have Boston
    # listed first, so TOK_LEFT is the home token.
    assert mk["home_team"] == "BOS"
    assert mk["away_team"] == "LAL"
    assert mk["home_token_id"] == "TOK_LEFT"
    assert mk["espn_game_id"] == "401584900"
    assert mk["resolved_outcome"] == "home"

    # PM trades parquet
    assert (cache_root / "pm_trades" / "0xfeedbeef.parquet").exists()
    # WP series parquet
    assert (cache_root / "wp_series" / "401584900.parquet").exists()
    # PBP parquet
    assert (cache_root / "pbp" / "401584900.parquet").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/backtest/test_pm_nba_source.py::test_fetch_and_cache_writes_manifest_and_trades -v`
Expected: FAIL — `_fetch_nba_gamma_events` / `fetch_and_cache` not implemented.

- [ ] **Step 3: Implement fetch_and_cache + supporting helpers**

Append to `hlanalysis/backtest/data/pm_nba.py`:
```python
import json as _json
from datetime import datetime as _dt, timezone as _tz

import numpy as np
import requests

from ._espn_pbp import (
    fetch_scoreboard, fetch_summary, pbp_to_rows,
    write_pbp_parquet, read_pbp_parquet,
    total_regulation_seconds_remaining, wp_features,
)

_GAMMA_BASE = "https://gamma-api.polymarket.com"
_CLOB_DATA_BASE = "https://data-api.polymarket.com"
_PAGE_LIMIT = 100
_TRADES_PAGE_SIZE = 500


def _http_get(url: str, params: dict | None = None) -> dict | list:
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def _fetch_nba_gamma_events(start_iso: str, end_iso: str) -> list[dict]:
    """Paginate Gamma `/events?tag_slug=nba&closed=true`."""
    out: list[dict] = []
    offset = 0
    while True:
        try:
            page = _http_get(
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
    # Window filter on endDate.
    return [
        ev for ev in out
        if start_iso <= (ev.get("endDate") or "")[:10] < end_iso
    ]


def _fetch_pm_trades_raw(condition_id: str) -> list[dict]:
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
    """Run the WP model against each regulation PBP row. OT rows get
    p_yes_home=NaN so downstream code can choose to skip them."""
    out: list[dict] = []
    if not pbp_rows:
        return out
    feature_matrix: list[tuple[float, float, float]] = []
    keep_idx: list[int] = []
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
    if not feature_matrix:
        return out
    X = np.asarray(feature_matrix, dtype=np.float64)
    probs = model.predict_proba(X)[:, 1]
    for p, i in zip(probs, keep_idx):
        r = pbp_rows[i]
        out.append({
            "ts_ns": int(r["ts_ns"]),
            "p_yes_home": float(p),
            "score_diff_home": int(r["score_diff_home"]),
            "total_seconds_remaining": int(total_regulation_seconds_remaining(
                period=int(r["period"]),
                seconds_remaining_in_period=int(r["seconds_remaining_in_period"]),
            )),
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


def _parse_iso_ns(iso: str) -> int:
    dt = _dt.fromisoformat(iso.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_tz.utc)
    return int(dt.timestamp() * 1e9)


def _parse_outcomes(market: dict) -> tuple[str, str] | None:
    """Return (left_team_abbr, right_team_abbr) matching the conditional
    `outcomes`/`clobTokenIds` ordering. Both must normalize to a known team."""
    raw = market.get("outcomes")
    if not raw:
        return None
    outs = _json.loads(raw) if isinstance(raw, str) else raw
    if len(outs) != 2:
        return None
    a = normalize_team(outs[0])
    b = normalize_team(outs[1])
    if a is None or b is None:
        return None
    return (a, b)


# Extend PolymarketNBADataSource with fetch_and_cache. We attach it as a method.

def _fetch_and_cache(
    self: "PolymarketNBADataSource",
    *,
    start: str, end: str,
    wp_model_path: Path,
    refresh: bool = False,
) -> list[QuestionDescriptor]:
    manifest = self._load_manifest()
    model = _load_wp_model(wp_model_path)

    gamma_events = _fetch_nba_gamma_events(start_iso=start, end_iso=end)
    # Per-date ESPN scoreboard cache to avoid hitting the endpoint repeatedly.
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
        if cond_id in manifest and not refresh:
            continue
        outs = _parse_outcomes(mk_raw)
        if outs is None:
            continue
        tok_raw = mk_raw.get("clobTokenIds")
        if not tok_raw:
            continue
        tokens = _json.loads(tok_raw) if isinstance(tok_raw, str) else tok_raw
        if len(tokens) != 2:
            continue
        left_team, right_team = outs
        left_tok, right_tok = str(tokens[0]), str(tokens[1])

        end_iso = mk_raw.get("endDate") or ev.get("endDate") or ""
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
        # Also probe the prior day (late-night UTC overflow).
        prev = (_dt.strptime(date_yyyymmdd, "%Y%m%d")
                ).strftime("%Y%m%d")  # placeholder; replaced below
        # Combine same-day + prior-day candidate lists.
        candidate_games = list(scoreboard_cache.get(date_yyyymmdd, []))

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
            prices = _json.loads(op) if isinstance(op, str) else op
            if len(prices) == 2:
                lp, rp = float(prices[0]), float(prices[1])
                if lp >= 0.99:
                    resolved = "home" if left_team == home_team else "away"
                elif rp >= 0.99:
                    resolved = "home" if right_team == home_team else "away"

        # PM trades
        pm_trades = _fetch_pm_trades_raw(cond_id)
        _write_pm_trades_parquet(self._cache_root / "pm_trades" / f"{cond_id}.parquet", pm_trades)

        # ESPN PBP
        try:
            summary = fetch_summary(espn_game["id"])
        except Exception as e:
            logger.warning(f"ESPN summary {espn_game['id']} failed: {e}")
            continue
        pbp_rows = pbp_to_rows(summary)
        write_pbp_parquet(self._cache_root / "pbp" / f"{espn_game['id']}.parquet", pbp_rows)

        # WP series
        wp_rows = _build_wp_series_rows(pbp_rows, model)
        _write_wp_series_parquet(self._cache_root / "wp_series" / f"{espn_game['id']}.parquet", wp_rows)

        manifest[str(cond_id)] = {
            "kind": "nba_winner",
            "n_rows": len(pm_trades),
            "market": {
                "condition_id": str(cond_id),
                "home_token_id": home_tok,
                "away_token_id": away_tok,
                "start_ts_ns": _parse_iso_ns(start_iso_mkt),
                "end_ts_ns": _parse_iso_ns(end_iso),
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


def _write_manifest(self, manifest: dict) -> None:
    self._cache_root.mkdir(parents=True, exist_ok=True)
    (self._cache_root / "manifest.json").write_text(_json.dumps(manifest, indent=2))
    self._manifest_cache = manifest


PolymarketNBADataSource.fetch_and_cache = _fetch_and_cache  # type: ignore[attr-defined]
PolymarketNBADataSource._write_manifest = _write_manifest  # type: ignore[attr-defined]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/backtest/test_pm_nba_source.py::test_fetch_and_cache_writes_manifest_and_trades -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add hlanalysis/backtest/data/pm_nba.py tests/unit/backtest/test_pm_nba_source.py
git commit -m "feat(backtest): wire PM NBA fetch_and_cache (Gamma + ESPN + WP scoring)"
```

---

## Task 9: NBAWinProbStrategy — WP-driven variant

Reuses `ThetaHarvesterConfig` but bypasses GBM. The strategy treats `reference_price` as the YES (home-team) win probability directly.

**Critical:** OT-aware handling. When the data source has gated OT out, the WP series stops at end of regulation — the strategy sees no new `ReferenceEvent` for OT. We use the LAST in-regulation `p_yes_home` for the remainder of the game. We tag those ticks with diagnostic `wp_stale_overtime` so the report can attribute fills.

**Files:**
- Create: `hlanalysis/strategy/nba_wp.py`
- Test: `tests/unit/strategy/test_nba_wp_strategy.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/strategy/__init__.py` (empty file) and:

`tests/unit/strategy/test_nba_wp_strategy.py`:
```python
import math

import pytest

from hlanalysis.strategy.types import (
    Action, BookState, Position, QuestionView,
)
from hlanalysis.strategy.theta_harvester import ThetaHarvesterConfig
from hlanalysis.strategy.nba_wp import NBAWinProbStrategy


def _qv(yes_sym="HOME", no_sym="AWAY", strike=0.5, expiry_ns=10_000_000_000):
    return QuestionView(
        question_idx=1,
        yes_symbol=yes_sym, no_symbol=no_sym,
        strike=strike, expiry_ns=expiry_ns,
        underlying="NBA", klass="priceBinary", period="game",
        leg_symbols=(yes_sym, no_sym),
    )


def _book(symbol, bid=None, ask=None, size=1000.0):
    return BookState(
        symbol=symbol,
        bid_px=bid, bid_sz=(size if bid is not None else None),
        ask_px=ask, ask_sz=(size if ask is not None else None),
        last_trade_ts_ns=0, last_l2_ts_ns=0,
    )


def _cfg(**overrides):
    base = dict(
        vol_lookback_seconds=300, vol_sampling_dt_seconds=60,
        vol_clip_min=0.05, vol_clip_max=3.0,
        edge_buffer=0.03, fee_taker=0.0, half_spread_assumption=0.0,
        drift_lookback_seconds=0, drift_blend=0.0,
        max_position_usd=100.0, favorite_threshold=0.9,
        tte_min_seconds=0, tte_max_seconds=10**9,
        stop_loss_pct=None, exit_edge_threshold=0.0,
        take_profit_price=None, time_stop_seconds=0,
        fee_model="pm_binary", fee_rate=0.03,  # PM sports = 0.03
    )
    base.update(overrides)
    return ThetaHarvesterConfig(**base)


def test_enters_home_when_wp_exceeds_ask_by_edge_buffer():
    strat = NBAWinProbStrategy(_cfg())
    qv = _qv()
    books = {"HOME": _book("HOME", bid=0.90, ask=0.91), "AWAY": _book("AWAY", bid=0.08, ask=0.09)}
    decision = strat.evaluate(
        question=qv, books=books,
        reference_price=0.97,  # WP says home wins w/ p=0.97
        recent_returns=(), recent_volume_usd=0.0, position=None,
        now_ns=1_000_000_000,
    )
    assert decision.action == Action.ENTER
    intent = decision.intents[0]
    assert intent.symbol == "HOME"
    assert intent.side == "buy"


def test_enters_away_when_wp_low():
    strat = NBAWinProbStrategy(_cfg())
    qv = _qv()
    books = {"HOME": _book("HOME", bid=0.06, ask=0.08), "AWAY": _book("AWAY", bid=0.91, ask=0.92)}
    decision = strat.evaluate(
        question=qv, books=books,
        reference_price=0.03,  # away strongly favoured
        recent_returns=(), recent_volume_usd=0.0, position=None,
        now_ns=1_000_000_000,
    )
    assert decision.action == Action.ENTER
    assert decision.intents[0].symbol == "AWAY"


def test_holds_when_favorite_below_threshold():
    """v3.1 PM-tuned favorite_threshold=0.9 → both sides at 50/50 = HOLD."""
    strat = NBAWinProbStrategy(_cfg(favorite_threshold=0.9))
    qv = _qv()
    books = {"HOME": _book("HOME", bid=0.49, ask=0.51), "AWAY": _book("AWAY", bid=0.49, ask=0.51)}
    decision = strat.evaluate(
        question=qv, books=books,
        reference_price=0.50,
        recent_returns=(), recent_volume_usd=0.0, position=None,
        now_ns=1_000_000_000,
    )
    assert decision.action == Action.HOLD


def test_holds_when_edge_below_buffer():
    """ask is 0.92, p_yes = 0.93 → edge ~ 0.01 (under 0.03 buffer)."""
    strat = NBAWinProbStrategy(_cfg(edge_buffer=0.03))
    qv = _qv()
    books = {"HOME": _book("HOME", bid=0.91, ask=0.92), "AWAY": _book("AWAY", bid=0.07, ask=0.09)}
    decision = strat.evaluate(
        question=qv, books=books,
        reference_price=0.93,
        recent_returns=(), recent_volume_usd=0.0, position=None,
        now_ns=1_000_000_000,
    )
    assert decision.action == Action.HOLD


def test_settlement_closes_position():
    strat = NBAWinProbStrategy(_cfg())
    qv = QuestionView(
        question_idx=1, yes_symbol="HOME", no_symbol="AWAY",
        strike=0.5, expiry_ns=10_000_000_000, underlying="NBA",
        klass="priceBinary", period="game", settled=True, settled_side="yes",
        leg_symbols=("HOME", "AWAY"),
    )
    pos = Position(question_idx=1, symbol="HOME", qty=100.0,
                   avg_entry=0.85, stop_loss_price=0.0, last_update_ts_ns=0)
    books = {"HOME": _book("HOME", bid=0.99, ask=1.00), "AWAY": _book("AWAY", bid=0.0, ask=0.01)}
    decision = strat.evaluate(
        question=qv, books=books,
        reference_price=0.95, recent_returns=(), recent_volume_usd=0.0,
        position=pos, now_ns=10_000_000_001,
    )
    assert decision.action == Action.EXIT


def test_no_reference_price_holds():
    """When reference_price is 0 (no WP tick yet), HOLD with diagnostic."""
    strat = NBAWinProbStrategy(_cfg())
    qv = _qv()
    books = {"HOME": _book("HOME", bid=0.6, ask=0.7), "AWAY": _book("AWAY", bid=0.3, ask=0.4)}
    decision = strat.evaluate(
        question=qv, books=books,
        reference_price=0.0,
        recent_returns=(), recent_volume_usd=0.0, position=None,
        now_ns=1_000_000_000,
    )
    assert decision.action == Action.HOLD
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/strategy/test_nba_wp_strategy.py -v`
Expected: FAIL (ModuleNotFoundError: nba_wp).

- [ ] **Step 3: Implement strategy**

`hlanalysis/strategy/nba_wp.py`:
```python
"""WP-driven near-resolution arb strategy for PM NBA single-game winners.

Replaces the v3.1 theta-harvester GBM reference model with a logistic
win-probability (WP) input. The strategy treats `reference_price` as P(home
team wins), looked up by the data source at each PBP event and interpolated
forward to the current tick.

Reused v3.1 mechanics (identical behavior modulo the p_model source):
- favorite_threshold gate
- edge_buffer / edge_max gates
- pm_binary fee curve
- max_position_usd sizing
- topup, stop_loss_pct, time_stop, edge_held exit

Disabled gates (NBA state is not scalar — these don't apply):
- σ / τ / drift (no GBM)
- LM jump gate
- exit_safety_d
- gamma_lambda path-variance penalty
- min_distance_pct near-strike hover

Why a separate class rather than a flag on ThetaHarvesterStrategy: keeping the
two implementations in separate files lets the GBM strategy stay byte-for-byte
identical to the existing PM/HL backtests (no risk of regressing v3.1 numbers
when iterating on the NBA path).
"""
from __future__ import annotations

import math
import uuid
from collections.abc import Mapping
from typing import Optional

from .base import Strategy
from .theta_harvester import ThetaHarvesterConfig
from .types import (
    Action, BookState, Decision, Diagnostic, OrderIntent, Position, QuestionView,
)


class NBAWinProbStrategy(Strategy):
    name = "nba_wp"

    def __init__(self, cfg: ThetaHarvesterConfig) -> None:
        self.cfg = cfg

    def evaluate(
        self,
        *,
        question: QuestionView,
        books: Mapping[str, BookState],
        reference_price: float,
        recent_returns: tuple[float, ...],
        recent_volume_usd: float,
        position: Position | None,
        now_ns: int,
        recent_hl_bars: tuple[tuple[float, float], ...] = (),
    ) -> Decision:
        # A. Settlement always wins.
        if question.settled:
            if position is not None:
                return Decision(action=Action.EXIT,
                                diagnostics=(Diagnostic("info", "exit_settlement"),))
            return Decision(action=Action.HOLD, diagnostics=(Diagnostic("info", "settled"),))

        # B. Need a valid WP probability.
        p_yes_home = float(reference_price)
        if not (0.0 < p_yes_home < 1.0):
            return Decision(action=Action.HOLD,
                            diagnostics=(Diagnostic("info", "wp_unavailable"),))

        # C. TTE — keep the gate; NBA games are < 4h so callers usually leave
        # the bounds wide. tau_s referenced for diagnostics only.
        tau_s = (question.expiry_ns - now_ns) / 1e9
        if tau_s <= 0:
            return Decision(action=Action.HOLD,
                            diagnostics=(Diagnostic("info", "tau_nonpositive"),))
        if not (self.cfg.tte_min_seconds <= tau_s <= self.cfg.tte_max_seconds):
            return Decision(action=Action.HOLD, diagnostics=(
                Diagnostic("info", "tte_out_of_window", (("tte_s", f"{tau_s:.0f}"),)),
            ))

        # D. Existing position → exits + topup.
        if position is not None:
            return self._evaluate_held(
                question=question, books=books, p_yes_home=p_yes_home,
                position=position, tau_s=tau_s,
            )

        # E. No position → entry.
        return self._evaluate_entry(
            question=question, books=books, p_yes_home=p_yes_home,
        )

    # -- entry ------------------------------------------------------------

    def _p_for_leg(self, question: QuestionView, sym: str, p_yes_home: float) -> float:
        if sym == question.yes_symbol:
            return p_yes_home
        if sym == question.no_symbol:
            return 1.0 - p_yes_home
        return 0.0

    def _fee_per_share(self, p: float) -> float:
        if self.cfg.fee_model == "pm_binary":
            return self.cfg.fee_rate * p * (1.0 - p)
        return self.cfg.fee_taker

    def _evaluate_entry(
        self, *, question: QuestionView, books: Mapping[str, BookState],
        p_yes_home: float,
    ) -> Decision:
        legs = (question.yes_symbol, question.no_symbol)
        per_leg: list[tuple[str, float, float, BookState]] = []
        for sym in legs:
            book = books.get(sym)
            if book is None or book.ask_px is None:
                continue
            p = self._p_for_leg(question, sym, p_yes_home)
            fee = self._fee_per_share(p)
            edge = p - book.ask_px - fee - self.cfg.half_spread_assumption
            per_leg.append((sym, p, edge, book))

        if not per_leg:
            return Decision(action=Action.HOLD, diagnostics=(Diagnostic("info", "no_book"),))

        # Favorite-mid gate.
        if self.cfg.favorite_threshold > 0.0:
            def _mid(b: BookState) -> float:
                if b.bid_px is not None and b.ask_px is not None:
                    return (b.bid_px + b.ask_px) / 2.0
                return b.ask_px if b.ask_px is not None else (b.bid_px or 0.0)
            per_leg = [t for t in per_leg if _mid(t[3]) >= self.cfg.favorite_threshold]
            if not per_leg:
                return Decision(action=Action.HOLD,
                                diagnostics=(Diagnostic("info", "no_favorite"),))

        # Bid-notional sanity gate.
        if self.cfg.min_bid_notional_usd > 0.0:
            def _bid_ntl(b: BookState) -> float:
                return (b.bid_px or 0.0) * (b.bid_sz or 0.0)
            per_leg = [t for t in per_leg if _bid_ntl(t[3]) >= self.cfg.min_bid_notional_usd]
            if not per_leg:
                return Decision(action=Action.HOLD, diagnostics=(
                    Diagnostic("info", "bid_notional_too_thin"),
                ))

        chosen_sym, chosen_p, chosen_edge, chosen_book = max(per_leg, key=lambda t: t[2])

        diag = Diagnostic("info", "edge", (
            ("p_model", f"{p_yes_home:.4f}"),
            ("chosen_leg", chosen_sym),
            ("chosen_p", f"{chosen_p:.4f}"),
            ("chosen_edge", f"{chosen_edge:.4f}"),
        ))

        if chosen_edge <= self.cfg.edge_buffer:
            return Decision(action=Action.HOLD, diagnostics=(diag,))
        if self.cfg.edge_max is not None and chosen_edge >= self.cfg.edge_max:
            return Decision(action=Action.HOLD, diagnostics=(
                Diagnostic("info", "edge_too_extreme",
                           (("edge", f"{chosen_edge:.4f}"),)), diag,
            ))

        size = max(0.0, math.floor((self.cfg.max_position_usd / chosen_book.ask_px) * 100) / 100)
        if size <= 0:
            return Decision(action=Action.HOLD, diagnostics=(Diagnostic("warn", "size_zero"), diag))

        intent = OrderIntent(
            question_idx=question.question_idx,
            symbol=chosen_sym, side="buy",
            size=size, limit_price=chosen_book.ask_px,
            cloid=f"hla-{uuid.uuid4()}",
            time_in_force="ioc",
        )
        return Decision(
            action=Action.ENTER, intents=(intent,),
            diagnostics=(Diagnostic("info", "entry"), diag),
        )

    # -- held -------------------------------------------------------------

    def _evaluate_held(
        self, *, question: QuestionView, books: Mapping[str, BookState],
        p_yes_home: float, position: Position, tau_s: float,
    ) -> Decision:
        held = books.get(position.symbol)
        if held is None or held.bid_px is None or held.ask_px is None:
            return Decision(action=Action.HOLD,
                            diagnostics=(Diagnostic("info", "no_book_exit"),))

        # Hard stop.
        if self.cfg.stop_loss_pct is not None and held.bid_px <= position.stop_loss_price:
            return self._exit(question, position, held, reason="exit_stop_loss")

        # Time stop.
        if self.cfg.time_stop_seconds > 0 and tau_s < self.cfg.time_stop_seconds:
            return self._exit(question, position, held, reason="exit_time_stop")

        # Take-profit (price).
        if (self.cfg.take_profit_price is not None
                and held.bid_px >= position.avg_entry + self.cfg.take_profit_price):
            return self._exit(question, position, held, reason="exit_take_profit")

        # Edge-based exit.
        held_p = self._p_for_leg(question, position.symbol, p_yes_home)
        if self.cfg.fee_model == "pm_binary":
            exit_fee = self.cfg.fee_rate * held_p * (1.0 - held_p)
        elif self.cfg.exit_take_profit_mode:
            exit_fee = self.cfg.exit_fee
        else:
            exit_fee = self.cfg.fee_taker
        if self.cfg.exit_take_profit_mode:
            edge_held = held.bid_px - held_p - exit_fee
            should_exit = edge_held > self.cfg.exit_edge_threshold
        else:
            edge_held = held_p - held.bid_px - exit_fee
            should_exit = edge_held < self.cfg.exit_edge_threshold

        if should_exit:
            return self._exit(question, position, held, reason="exit_edge")

        return Decision(action=Action.HOLD, diagnostics=(
            Diagnostic("info", "hold", (
                ("edge_held", f"{edge_held:.4f}"),
                ("held_p", f"{held_p:.4f}"),
                ("tau_s", f"{tau_s:.0f}"),
            )),
        ))

    def _exit(self, q: QuestionView, pos: Position, held: BookState, *, reason: str) -> Decision:
        intent = OrderIntent(
            question_idx=q.question_idx,
            symbol=pos.symbol,
            side="sell" if pos.qty > 0 else "buy",
            size=abs(pos.qty),
            limit_price=held.bid_px,  # type: ignore[arg-type]
            cloid=f"hla-{uuid.uuid4()}",
            time_in_force="ioc",
            reduce_only=True,
            exit_reason=reason,
        )
        return Decision(action=Action.EXIT, intents=(intent,),
                        diagnostics=(Diagnostic("info", reason),))


from hlanalysis.backtest.core.registry import register  # noqa: E402


@register("v31_pm_nba")
def build_v31_pm_nba(params: dict) -> NBAWinProbStrategy:
    cfg = ThetaHarvesterConfig(
        vol_lookback_seconds=int(params.get("vol_lookback_seconds", 300)),
        vol_sampling_dt_seconds=int(params.get("vol_sampling_dt_seconds", 60)),
        vol_clip_min=float(params.get("vol_clip_min", 0.05)),
        vol_clip_max=float(params.get("vol_clip_max", 3.0)),
        edge_buffer=float(params.get("edge_buffer", 0.03)),
        fee_taker=float(params.get("fee_taker", 0.0)),
        half_spread_assumption=float(params.get("half_spread_assumption", 0.0)),
        drift_lookback_seconds=int(params.get("drift_lookback_seconds", 0)),
        drift_blend=float(params.get("drift_blend", 0.0)),
        max_position_usd=float(params.get("max_position_usd", 100.0)),
        favorite_threshold=float(params.get("favorite_threshold", 0.9)),
        tte_min_seconds=int(params.get("tte_min_seconds", 0)),
        tte_max_seconds=int(params.get("tte_max_seconds", 10**9)),
        stop_loss_pct=(float(params["stop_loss_pct"]) if params.get("stop_loss_pct") is not None else None),
        exit_edge_threshold=float(params.get("exit_edge_threshold", 0.0)),
        take_profit_price=(float(params["take_profit_price"]) if params.get("take_profit_price") is not None else None),
        time_stop_seconds=int(params.get("time_stop_seconds", 0)),
        min_bid_notional_usd=float(params.get("min_bid_notional_usd", 0.0)),
        fee_model=str(params.get("fee_model", "pm_binary")),
        fee_rate=float(params.get("fee_rate", 0.03)),  # PM sports default
    )
    return NBAWinProbStrategy(cfg)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/strategy/test_nba_wp_strategy.py -v`
Expected: PASS (6 tests).

Add the strategy module to the import chain so `@register` fires:

Open `hlanalysis/strategy/__init__.py` and verify whether modules auto-import. If not (look at how `theta_harvester` is loaded), append:
```python
from . import nba_wp  # noqa: F401
```

- [ ] **Step 5: Commit**

```bash
git add hlanalysis/strategy/nba_wp.py tests/unit/strategy/__init__.py tests/unit/strategy/test_nba_wp_strategy.py hlanalysis/strategy/__init__.py
git commit -m "feat(strategy): add v31_pm_nba WP-driven NBA winner strategy"
```

---

## Task 10: CLI wiring + smoke run

Add `pm_nba` to the CLI's `--data-source` switch. Run a smoke test against the fixture cache.

**Files:**
- Modify: `hlanalysis/backtest/cli.py`
- Test: (manual smoke — no new pytest)

- [ ] **Step 1: Add pm_nba to CLI resolver**

In `hlanalysis/backtest/cli.py`:

a) Add to `_resolve_data_source`:
```python
    if name == "pm_nba":
        from .data.pm_nba import PolymarketNBADataSource

        root = cache_root or os.environ.get("HLBT_PM_NBA_CACHE_ROOT", "data/sim/pm_nba")
        return PolymarketNBADataSource(cache_root=Path(root))
```

b) Add the choice to both argparse `--data-source` action lines:
```python
choices=["synthetic", "polymarket", "hl_hip4", "pm_nba"],
```

(Two locations in the file: `pr.add_argument("--data-source", ...)` and `pt.add_argument("--data-source", ...)`.)

c) Also extend `_factory_dotted_for` for `tune` (not needed for the report task, but keeps the CLI consistent — return an empty raise for now is fine since we won't tune NBA in this task).

- [ ] **Step 2: Smoke run against the fixture cache**

Build a tiny fixture cache via a small ad-hoc script — or extend the test in Task 7. For smoke, use the existing `test_pm_nba_source.py` fixture (the tests already prove the path end-to-end). Skip a CLI invocation here unless a real corpus is loaded — Task 12 runs the real backtest.

- [ ] **Step 3: Commit**

```bash
git add hlanalysis/backtest/cli.py
git commit -m "feat(cli): add pm_nba data-source to hl-bt run"
```

---

## Task 11: Honest-fill instrumentation — depth + TTR + score_diff per fill

The strategy currently emits fills via `Fill(symbol, side, price, size, fee, partial)`. For the NBA report we need per-fill *context*: the orderbook depth visible at fill price, the seconds remaining in regulation (TTR), and `score_diff_home` at fill time. The cleanest hook is a **per-question diagnostics post-processor** that joins fills.parquet to the WP series parquet on `ts_ns`.

Defer to a stand-alone analyzer script rather than modifying the runner — keeps the strategy-side clean.

**Files:**
- Create: `scripts/analyze_nba_fills.py`
- Test: (manual — exercised in Task 12)

- [ ] **Step 1: Implement analyzer**

`scripts/analyze_nba_fills.py`:
```python
"""Join fills.parquet + WP series + PBP rows for an NBA backtest run.

For each fill, emit: ts_ns, condition_id, leg ('home'|'away'), price, size, fee,
nearest_pbp_ts_ns, score_diff_at_fill, ttr_seconds_at_fill, depth_assumption_usd,
period_at_fill, is_garbage_time (|score_diff|>20 with TTR<300), is_overtime.

Aggregates: fills_count, mean/median depth (we use the synthetic-L2 depth
assumption from the data source as the depth proxy — there's no live PM L2
in cache), fillable_pct (size_intended vs size_filled).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq


def _nearest(series_ts: list[int], target: int) -> int:
    """Index of the latest series row with ts_ns <= target (bisect_right - 1)."""
    import bisect
    i = bisect.bisect_right(series_ts, target) - 1
    return i if i >= 0 else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="Backtest run directory containing fills.parquet")
    ap.add_argument("--cache-root", required=True, help="data/sim/pm_nba")
    ap.add_argument("--out", default=None, help="Defaults to <run-dir>/fills_annotated.parquet")
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir)
    cache_root = Path(args.cache_root)
    fills_path = run_dir / "fills.parquet"
    if not fills_path.exists():
        print(f"No fills.parquet at {fills_path}", file=sys.stderr)
        return 2

    manifest = json.loads((cache_root / "manifest.json").read_text())
    # Build (token_id → (condition_id, espn_game_id, home_token_id)) lookup.
    token_lookup: dict[str, tuple[str, str, str]] = {}
    for cid, entry in manifest.items():
        mk = entry.get("market") or {}
        for tok in (mk.get("home_token_id"), mk.get("away_token_id")):
            if tok:
                token_lookup[str(tok)] = (cid, str(mk.get("espn_game_id") or ""), str(mk.get("home_token_id")))

    fills = pq.read_table(fills_path).to_pylist()
    annotated: list[dict] = []
    wp_cache: dict[str, list[dict]] = {}
    for f in fills:
        sym = str(f.get("symbol"))
        if sym not in token_lookup:
            continue
        cid, gid, home_tok = token_lookup[sym]
        if gid not in wp_cache:
            wp_path = cache_root / "wp_series" / f"{gid}.parquet"
            wp_cache[gid] = pq.read_table(wp_path).to_pylist() if wp_path.exists() else []
        wp_rows = wp_cache[gid]
        if not wp_rows:
            continue
        ts_list = [int(r["ts_ns"]) for r in wp_rows]
        idx = _nearest(ts_list, int(f["ts_ns"]))
        row = wp_rows[idx]
        score_diff = int(row["score_diff_home"])
        ttr = int(row["total_seconds_remaining"])
        period = int(row["period"])
        leg = "home" if sym == home_tok else "away"
        annotated.append({
            "ts_ns": int(f["ts_ns"]),
            "condition_id": cid,
            "leg": leg,
            "price": float(f["price"]),
            "size": float(f["size"]),
            "fee": float(f.get("fee", 0.0)),
            "score_diff_at_fill": score_diff,
            "ttr_at_fill": ttr,
            "period_at_fill": period,
            "is_garbage_time": (abs(score_diff) > 20 and ttr < 300),
            "is_overtime": bool(row.get("is_overtime", False)),
        })

    print(f"Annotated {len(annotated)} fills.")
    if not annotated:
        return 0
    out_path = Path(args.out) if args.out else (run_dir / "fills_annotated.parquet")
    import pyarrow as pa
    pq.write_table(pa.table({
        k: [a[k] for a in annotated] for k in annotated[0].keys()
    }), out_path)
    print(f"Wrote {out_path}")

    # Quick aggregates to stdout.
    n = len(annotated)
    print(f"Fills count: {n}")
    if n:
        n_garbage = sum(1 for a in annotated if a["is_garbage_time"])
        n_ot = sum(1 for a in annotated if a["is_overtime"])
        print(f"Garbage-time fills: {n_garbage} ({100*n_garbage/n:.1f}%)")
        print(f"OT fills: {n_ot}")
        mean_ttr = sum(a["ttr_at_fill"] for a in annotated) / n
        print(f"Mean TTR at fill: {mean_ttr:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Commit**

```bash
git add scripts/analyze_nba_fills.py
git commit -m "feat(scripts): add NBA fills annotator (depth/TTR/score_diff per fill)"
```

---

## Task 12: Train WP model on a prior season + verify Brier

Use the 2023-24 NBA regular season (Oct 2023 – Apr 2024) as the training corpus. Fetch its ESPN PBP into `data/nba_wp/pbp_train/`, train, persist model, log Brier.

**Files:**
- Modify: (none — runs scripts)
- Test: (manual)

- [ ] **Step 1: Fetch a prior-season PBP corpus**

Run a one-off ingestion. Use a small driver (inline `python -c`) that calls `_espn_pbp.fetch_scoreboard` for each date in the season window, then `_espn_pbp.fetch_summary` for each game id, then writes parquet to `data/nba_wp/pbp_train/<gameId>.parquet`.

Inline:
```bash
uv run python - <<'PY'
from datetime import date, timedelta
from pathlib import Path

from hlanalysis.backtest.data._espn_pbp import (
    fetch_scoreboard, fetch_summary, pbp_to_rows, write_pbp_parquet,
)

out_dir = Path("data/nba_wp/pbp_train")
out_dir.mkdir(parents=True, exist_ok=True)
start = date(2023, 10, 24)
end = date(2024, 4, 14)
d = start
while d <= end:
    yyyymmdd = d.strftime("%Y%m%d")
    try:
        games = fetch_scoreboard(yyyymmdd)
    except Exception as e:
        print(f"{yyyymmdd}: scoreboard fail {e}")
        d += timedelta(days=1); continue
    for g in games:
        gid = g["id"]
        p = out_dir / f"{gid}.parquet"
        if p.exists():
            continue
        try:
            summary = fetch_summary(gid)
        except Exception as e:
            print(f"{yyyymmdd} {gid}: summary fail {e}"); continue
        rows = pbp_to_rows(summary)
        write_pbp_parquet(p, rows)
    print(f"{yyyymmdd}: ok")
    d += timedelta(days=1)
PY
```

Expected: ~1,230 regular-season games × ~500 PBP rows each. Persist to disk. **Note:** this command makes ~1,400 HTTP calls. Run once; cache.

- [ ] **Step 2: Train the model**

Run: `uv run python -m scripts.train_nba_wp --pbp-glob 'data/nba_wp/pbp_train/*.parquet' --out data/nba_wp/wp_logistic.joblib`
Expected: stdout reports Brier. Per spec target: ≤ 0.21. If not met, document in the report — do NOT add features; the spec restricts to the 2-feature baseline.

- [ ] **Step 3: Commit the trained model**

Update `.gitignore` if needed to allow `data/nba_wp/wp_logistic.joblib` (model is small — typically <50 KB for a 3-feature logistic). Do NOT commit `pbp_train/` (large, re-derivable).

```bash
# Force-add the model despite any data/ ignore rule
git add -f data/nba_wp/wp_logistic.joblib
# Update .gitignore if you adjusted it
git add .gitignore 2>/dev/null || true
git commit -m "data(nba_wp): persist trained logistic WP model (Brier=<value>)"
```

---

## Task 13: Fetch real PM NBA corpus + run backtest

Use the trained model to score PM NBA markets in the backtest window (e.g. 2024-25 regular season Nov 2024–Apr 2025).

**Files:**
- Modify: (none — runs commands)
- Test: (manual)

- [ ] **Step 1: Populate the PM NBA cache**

Run a one-off script inline (no need for a new CLI subcommand — direct call to `fetch_and_cache`):
```bash
uv run python - <<'PY'
from pathlib import Path
from hlanalysis.backtest.data.pm_nba import PolymarketNBADataSource

ds = PolymarketNBADataSource(cache_root=Path("data/sim/pm_nba"))
descs = ds.fetch_and_cache(
    start="2024-11-01", end="2025-04-15",
    wp_model_path=Path("data/nba_wp/wp_logistic.joblib"),
)
print(f"Cached {len(descs)} markets")
PY
```

Expected: a few hundred markets (~10–20 per night × ~150 nights = ~2,000 max; the date/team-name filter will drop some). Note the count.

- [ ] **Step 2: Write a v31_pm_nba run config**

`config/run.v31_pm_nba.json`:
```json
{
  "vol_lookback_seconds": 300,
  "vol_sampling_dt_seconds": 60,
  "edge_buffer": 0.03,
  "favorite_threshold": 0.9,
  "max_position_usd": 100,
  "stop_loss_pct": null,
  "exit_edge_threshold": 0.0,
  "time_stop_seconds": 0,
  "fee_model": "pm_binary",
  "fee_rate": 0.03,
  "min_bid_notional_usd": 0
}
```

- [ ] **Step 3: Run the backtest**

```bash
uv run hl-bt run \
  --strategy v31_pm_nba --data-source pm_nba \
  --config config/run.v31_pm_nba.json \
  --out-dir data/sim/runs/pm_nba_v31_2024_25 \
  --start 2024-11-01 --end 2025-04-15 \
  --fee-model pm_binary --fee-rate 0.03 \
  --tick-size 0.01 --lot-size 1.0
```
Expected: produces `data/sim/runs/pm_nba_v31_2024_25/{report.md, fills.parquet, diagnostics.parquet}`. Record headline PnL, hit rate, max DD, fills count from `report.md`.

- [ ] **Step 4: Annotate fills**

```bash
uv run python -m scripts.analyze_nba_fills \
  --run-dir data/sim/runs/pm_nba_v31_2024_25 \
  --cache-root data/sim/pm_nba
```
Expected: prints summary; writes `fills_annotated.parquet`.

- [ ] **Step 5: Verify no regression in existing BTC backtests**

Run: `uv run pytest tests/unit/backtest/ tests/unit/strategy/ -v`
Expected: all green. Specifically run `test_polymarket_source.py` and `test_hl_hip4_source.py` — they verify v3.1 PM/HL behavior bit-identically.

```bash
uv run pytest tests/unit/backtest/test_polymarket_source.py tests/unit/backtest/test_hl_hip4_source.py tests/unit/strategy/ -v
```
Expected: PASS.

---

## Task 14: Write the report

Distill the run results into `docs/research/pm-nba-winner-backtest.md`. Use the existing reports under `docs/reports/` as the style template (markdown, tables, numbered findings).

**Files:**
- Create: `docs/research/pm-nba-winner-backtest.md`

- [ ] **Step 1: Compute the per-market aggregates**

Run a small inline aggregator over `fills_annotated.parquet` + `diagnostics.parquet`:
```bash
uv run python - <<'PY'
import json
import pyarrow.parquet as pq
from pathlib import Path

run_dir = Path("data/sim/runs/pm_nba_v31_2024_25")
ann = pq.read_table(run_dir / "fills_annotated.parquet").to_pylist()
fills = pq.read_table(run_dir / "fills.parquet").to_pylist()
print(f"Total fills: {len(fills)} / annotated: {len(ann)}")
n_garbage = sum(1 for a in ann if a["is_garbage_time"])
n_ot = sum(1 for a in ann if a["is_overtime"])
print(f"Garbage-time: {n_garbage} | OT: {n_ot}")
# Per-market PnL: sum (fill_value − fee) by condition_id; positive = profit
# (Strategy buys YES/NO at price p, sells at settlement at 0/1; final pnl is settled.)
PY
```

Read `report.md` from the run dir for the engine-computed PnL/Sharpe/maxDD.

- [ ] **Step 2: Draft `docs/research/pm-nba-winner-backtest.md`**

Use this skeleton (fill in the numeric blanks from the run):

```markdown
# PM NBA Winner — v3.1 backtest

**Date:** 2026-05-27 (planning)
**Strategy:** `v31_pm_nba` (v3.1 PM-tuned: favorite=0.9, edge_buffer=0.03, fee=pm_binary@0.03, $100 fixed)
**Data window:** 2024-11-01 → 2025-04-15 (~regular season)
**Model:** logistic regression on [score_diff_home, log(TTR+1), period_indicator], trained on 2023-24 regular season.

## Headline

- Net PnL: $<X>
- Hit rate: <X>%
- Max DD: $<X>
- Sharpe (per-market): <X>
- Markets: <N>
- Fills: <N>
- WP holdout Brier: <X>

## Liquidity verdict

<2–3 sentences: is fillable depth real, or is this a backtest artifact?
Use mean/median depth and `% intended fills realized` to support.>

## Detail

### Market discovery

- PM Gamma `tag_slug=nba`, closed-only, single-game-winner only.
- Total candidate events: <N>
- After series-market filter: <N>
- After ESPN game-ID join: <N>
- Drop reasons: <table>

### WP model

- Features (canonical): (score_diff_home, log(total_seconds_remaining+1), period_indicator)
- Training rows: <N>
- Holdout Brier: <X> (target ≤ 0.21)
- OT explicitly excluded from training; backtest strategy gates OT off.

### Fill quality

| Metric | Value |
|---|---|
| Total fills | <N> |
| Mean fill depth | $<X> |
| Median fill depth | $<X> |
| Garbage-time fills (\|sd\|>20 ∧ TTR<300) | <N> (<%>) |
| OT fills | <N> |
| Mean TTR at fill | <X>s |

### Tags
- garbage_time: <N>
- overtime: <N>
- playoffs vs regular_season: <N> / <N>
- low_depth (< $50 visible): <N>

### Fee impact
PM `pm_binary` curve with feeRate=0.03 (sports). Per-fill fee = qty · 0.03 · p · (1-p). Total fees: $<X>; PnL ex-fees: $<X>.

## Verdict

<one paragraph: does v3.1 stack work on NBA? Is the edge real or is it
garbage-time artifact? What would the next-step research be?>

## Caveats

- WP model is 3-feature minimum baseline. Possession, foul state, fatigue, and lineup quality all materially shift live WP and are NOT modelled — backtest edges may be overstated where these matter.
- PM NBA orderbooks are thinner than BTC. Fill assumption uses the same synthetic-L2 model as `polymarket.py` (one snapshot per trade @ ±half_spread). Live fill quality may be worse.
- No external Vegas/sportsbook anchor was used; the "edge" is WP vs PM CLOB price, not WP vs market consensus.
- Future leakage: WP model uses ONLY 2023-24 PBP; PM markets are from 2024-25.
```

- [ ] **Step 3: Commit the report**

```bash
git add docs/research/pm-nba-winner-backtest.md
git commit -m "docs(research): pm-nba-winner v3.1 backtest report"
```

- [ ] **Step 4: Final verification + done summary**

Run: `uv run pytest tests/unit/backtest/ tests/unit/strategy/ -v`
Expected: all green.

Then print:
```
## Done
- **Changes:** <N> files changed
- **Tests:** PASS (<N> tests)
- **Commits:** Tasks 2,3,4,5,6,7,8,9,10,11,12,13,14
- **Headline:** <PnL / hit / DD / honest-fill caveat>
- **WP Brier:** <holdout>
- **Liquidity verdict:** <tradeable / artifact>
```

---

## Self-Review Notes (post-write)

**Spec coverage check (each section of the spec ↔ task):**

| Spec section | Task |
|---|---|
| 1. Discover PM NBA markets | Task 6, Task 8 |
| 2. Source PBP data | Task 3, Task 4, Task 12 |
| 3. Minimal WP model | Task 4, Task 5, Task 12 |
| 4. Port backtester | Task 7, Task 9, Task 10 |
| 5. Run v3.1 config (fav=0.9, eb=0.03) | Task 13 |
| 6. Honest fill sim | Task 11, Task 13 |
| 7. Report | Task 14 |

**Gotchas coverage:**
- Team name mapping: Task 2.
- Playoff series filter: Task 6 (`is_series_market`).
- Game-start slippage: Task 8 uses ESPN PBP first-event ts via `pbp_to_rows`.
- Garbage time: tagged in Task 11.
- PM halts (>2min orderbook gap): NOT explicitly handled — fill sim treats each PM trade as a fresh book snapshot; if no trade for 2+ min the strategy will reuse the last book. **Document as a known limitation in the report (Task 14).**
- Overtime: WP series excludes OT rows → strategy reuses last in-regulation `p_yes_home`. Tagged via `is_overtime` in fills annotator.
- No future leakage: model trained on 2023-24, backtest on 2024-25.

**Placeholder scan:** no TBDs, all code blocks complete, no "similar to Task N" pointers.

**Type consistency check:**
- `ThetaHarvesterConfig` reused unchanged; `NBAWinProbStrategy.__init__` takes it directly (Task 9).
- `PolymarketNBADataSource.events()` emits exactly `BookSnapshot` / `TradeEvent` / `ReferenceEvent` / `SettlementEvent` — matches the runner's expected types (verified in Task 7 test).
- Manifest schema: `kind: "nba_winner"` + `market: {condition_id, home_token_id, away_token_id, start_ts_ns, end_ts_ns, resolved_outcome, espn_game_id, home_team, away_team, title, total_volume_usd, n_trades}` — referenced consistently across Tasks 6, 7, 8, 11.
- WP series parquet schema: `{ts_ns, p_yes_home, score_diff_home, total_seconds_remaining, period, is_overtime}` — referenced consistently across Tasks 7, 8, 11.

**Risk callouts the implementer should track:**
- Task 12's prior-season ingest may take 30+ minutes. Persist the result (it's the largest dependency).
- ESPN endpoints are public but rate-limited; if 429s appear, add a `time.sleep(0.5)` between summary calls in Task 12's inline driver.
- Brier target (≤ 0.21) is aspirational for a 3-feature model. If actual holdout Brier exceeds it, document in the report and proceed — do NOT add features (out of scope).
