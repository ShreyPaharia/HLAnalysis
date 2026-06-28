#!/usr/bin/env python3
"""Loss-injection tail stress for the HL v1 BUCKET candidates (event cadence 0.2/2.0).

Why: v1 bucket has NO active exit gate (stop_loss=null, exit_safety_d=0,
exit_bid_floor=0) — it buy-and-holds the bucket favorite to settlement. So its
realized maxDD=$0 is TAIL-BLIND: the 2026-05..06 corpus simply had no favorite lose
at settlement. This stresses that: each entered-and-held WINNING favorite flips to a
loss with prob = its own implied rate (1 − avg entry price). With no exit, a flip
loses the FULL stake. Realized losers kept as-is. Monte-Carlo → EV / 5th-pct /
P(net loss). Same model as run_v31_hl_binary_axes_tail_stress.py.

Also reports the distribution SCALED to the proposed $800/position sizing. NOTE the
scaling is LINEAR (optimistic): at $800 vs the backtested $300 (v1) / $500 (v31)
per-position, real fills walk deeper into the thin HIP-4 book, so true $ are worse
than the linear scale. P(loss) is scale-invariant.

Reads per-(config,question) fills.parquet from the 0.2/2.0 validation run.
Analysis only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

VR = Path("/Users/shreypaharia/Documents/Projects/Trading/HLAnalysis/data/sim/runs/v1-hl-validate-0p2-2026-06-24")
N_MC = 20000
SEED = 12345

# (label, cell_dir, esd, backtest_per_position_usd)
CONFIGS = [
    ("v1 bucket TUNED (tte8h, esd0)", VR / "bucket" / "winner", 0.0, 300.0),
    ("v1 bucket LIVE (tte6h, esd0)", VR / "bucket" / "live", 0.0, 300.0),
    ("v31 bucket LIVE (esd1.0)", VR / "bucket_v31" / "v31_live", 1.0, 500.0),
]
TARGET_PER_POSITION = 800.0


def concat_fills(cell_dir: Path) -> pd.DataFrame:
    parts = [pd.read_parquet(p) for p in sorted(cell_dir.glob("q*/fills.parquet"))]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def per_question(fills: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for qid, g in fills.groupby("question_id"):
        g = g[~g["is_hedge"].fillna(False)]
        realized = (
            float(g["realized_pnl_at_settle"].dropna().iloc[0]) if g["realized_pnl_at_settle"].notna().any() else 0.0
        )
        buys = g[(g["side"] == "buy") & (g["cloid"] != "settle")]
        if len(buys) == 0 or buys["size"].sum() <= 0:
            continue
        avg_entry = float(np.average(buys["price"], weights=buys["size"]))
        settle = g[g["cloid"] == "settle"]
        held_qty = float(settle["size"].sum()) if len(settle) else 0.0
        rows.append(dict(qid=qid, realized=realized, avg_entry=avg_entry, stake=held_qty * avg_entry,
                         held_to_settle=held_qty > 0))
    return pd.DataFrame(rows)


def stress(label: str, cell_dir: Path, esd: float, bt_size: float, rng: np.random.Generator, scale: float) -> dict:
    fills = concat_fills(cell_dir)
    if fills.empty:
        return {}
    pq = per_question(fills)
    if pq.empty:
        return {}
    # scale all $ by `scale` (= target_per_position / backtest_per_position)
    pq = pq.copy()
    pq["realized"] *= scale
    pq["stake"] *= scale
    realized_total = pq["realized"].sum()
    losers = pq[pq["realized"] < 0]["realized"]
    emp_loss = float(-losers.mean()) if len(losers) else 50.0 * scale
    wins = pq[(pq["held_to_settle"]) & (pq["realized"] > 0)].copy()
    wins["p_flip"] = (1.0 - wins["avg_entry"]).clip(0.01, 0.5)
    wins["loss_on_flip"] = wins["stake"] if esd <= 0.0 else emp_loss
    fixed = realized_total - wins["realized"].sum()
    p = wins["p_flip"].to_numpy()
    win_pnl = wins["realized"].to_numpy()
    loss = wins["loss_on_flip"].to_numpy()
    flips = rng.random((N_MC, len(wins))) < p
    totals = fixed + np.where(flips, -loss, win_pnl).sum(axis=1)
    return dict(
        label=label, esd=esd, realized=realized_total, n_q=len(pq), n_win=len(wins),
        mean_flips=float(p.sum()), ev=float(totals.mean()),
        p5=float(np.percentile(totals, 5)), p_loss=float((totals < 0).mean()),
    )


def run_block(title: str, scale_to_800: bool):
    rng = np.random.default_rng(SEED)
    rows = []
    for lbl, d, esd, bt in CONFIGS:
        scale = (TARGET_PER_POSITION / bt) if scale_to_800 else 1.0
        r = stress(lbl, d, esd, bt, rng, scale)
        if r:
            rows.append(r)
    print(f"\n{title}")
    print("=" * 110)
    print(f"{'config':>34} {'esd':>4} {'realized':>10} {'wins':>5} {'~flips':>6} "
          f"{'EV':>10} {'5th-pct':>10} {'P(loss)':>8}")
    print("-" * 110)
    for r in rows:
        print(f"{r['label']:>34} {r['esd']:>4.1f} ${r['realized']:>9.0f} {r['n_win']:>5} {r['mean_flips']:>6.2f} "
              f"${r['ev']:>9.0f} ${r['p5']:>9.0f} {r['p_loss']:>7.1%}")
    print("=" * 110)


def main():
    print(f"Loss-injection tail stress — HL v1 BUCKET (N={N_MC}, corpus 2026-05-06..06-24, n=47)")
    print("flip prob = 1 − avg_entry_price; loss_on_flip = FULL stake (esd=0, no exit) | empirical adverse-exit (esd>0)")
    run_block("AT BACKTEST SIZE (v1 $300/pos, v31 $500/pos)", scale_to_800=False)
    run_block("SCALED TO $800/POSITION (LINEAR — optimistic; real fills worse on thin books)", scale_to_800=True)


if __name__ == "__main__":
    main()
