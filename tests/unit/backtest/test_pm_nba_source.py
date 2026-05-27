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
