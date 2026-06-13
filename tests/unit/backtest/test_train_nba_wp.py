from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.train_nba_wp import (
    build_training_frame,
    fit_logistic_wp,
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


def test_fit_logistic_wp_separable(tmp_path: Path):
    """8 games where huge home leads → home wins; huge away leads → away wins.
    Logistic should fit cleanly and produce Brier well under 0.25 on a
    held-out sample of the same structure."""
    np.random.seed(42)
    paths: list[Path] = []
    for i in range(8):
        home_wins = i % 2 == 0
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
