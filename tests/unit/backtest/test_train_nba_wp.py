from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.train_nba_wp import (
    build_training_frame,
)


def _write_pbp(path: Path, rows: list[dict]) -> None:
    table = pa.table(
        {
            "ts_ns": [r["ts_ns"] for r in rows],
            "period": [r["period"] for r in rows],
            "seconds_remaining_in_period": [r["seconds_remaining_in_period"] for r in rows],
            "score_diff_home": [r["score_diff_home"] for r in rows],
            "home_score": [r["home_score"] for r in rows],
            "away_score": [r["away_score"] for r in rows],
            "is_overtime": [r["is_overtime"] for r in rows],
        }
    )
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
            out.append(
                {
                    "ts_ns": ts,
                    "period": q,
                    "seconds_remaining_in_period": s,
                    "score_diff_home": sd,
                    "home_score": max(sd, 0) + 50,
                    "away_score": max(-sd, 0) + 50,
                    "is_overtime": False,
                }
            )
            ts += secs_per_play * 1_000_000_000
    return out


def test_build_training_frame_excludes_overtime(tmp_path: Path):
    rows = [
        {
            "ts_ns": 1,
            "period": 1,
            "seconds_remaining_in_period": 720,
            "score_diff_home": 0,
            "home_score": 0,
            "away_score": 0,
            "is_overtime": False,
        },
        # OT row — must be excluded from training
        {
            "ts_ns": 2,
            "period": 5,
            "seconds_remaining_in_period": 300,
            "score_diff_home": 0,
            "home_score": 110,
            "away_score": 110,
            "is_overtime": True,
        },
    ]
    _write_pbp(tmp_path / "001.parquet", rows)
    X, y = build_training_frame([tmp_path / "001.parquet"])
    # Only the 1 regulation row should survive.
    assert X.shape == (1, 3)
    assert y.shape == (1,)
