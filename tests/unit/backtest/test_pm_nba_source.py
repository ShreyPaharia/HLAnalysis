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
