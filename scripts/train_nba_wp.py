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
