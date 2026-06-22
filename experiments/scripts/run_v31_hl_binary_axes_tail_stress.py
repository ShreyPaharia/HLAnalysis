#!/usr/bin/env python3
"""Loss-injection tail stress for the HL binary AXES-sweep candidates.

Same model as run_v31_hl_safetyd_tail_stress.py (which reproduces the 2026-06-18
methodology): the realized HL corpus is tail-deficient (favorites mostly resolve
ITM), so realized maxDD/worst-half understate tail risk. Each favorite the config
ENTERED and HELD to settlement as a WIN flips to a loss with probability = its own
implied loss rate (1 − weighted-avg entry price). A flipped favorite loses:
  * the FULL stake under buy-and-hold (exit_safety_d == 0, no protective exit), or
  * the config's EMPIRICAL adverse-exit loss (mean realized loss on entered-and-lost
    questions) when the soft exit is on (exit_safety_d > 0).
Realized losers are kept as-is. Monte-Carlo N runs -> EV, 5th-pct, P(net loss).

Reads per-config full-corpus fills.parquet. Candidates come from the axes sweep
(data/sim/runs/v31-hl-binary-axes-2026-06-22); the live anchor (prod_ref) is there
too; the buy-and-hold VALIDATION cell is reused from the safetyd sweep
(data/sim/runs/v31-hl-safetyd-sweep-2026-06-22/msd2.5_esd0.0).

VALIDATION: the BAH cell must reproduce the doc's collapse (EV near 0,
P(loss) high, deep-negative 5th-pct). If it does, the exits-on candidate numbers
are trustworthy.

Analysis only. No config change, no deploy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

AXES = Path("/Users/shreypaharia/Documents/Projects/Trading/HLAnalysis/data/sim/runs/v31-axes-driver/full")
SAFETYD = Path(
    "/Users/shreypaharia/Documents/Projects/Trading/HLAnalysis/data/sim/runs/v31-hl-safetyd-sweep-2026-06-22"
)
N_MC = 20000
SEED = 12345

# (display_label, fills_dir, esd) — esd==0 => buy-and-hold (full-stake flip).
# All candidates keep the live safety band (esd=1.5); only fav/eb/vlb differ, so
# the soft mid-hold exit caps each flip at the empirical adverse-exit loss.
CONFIGS = [
    ("prod_ref (LIVE fav.85/eb.02/vlb900)", AXES / "prod_ref", 1.5),
    ("cand1 fav.80/eb.01/vlb1800", AXES / "grid_fav0.8_eb0.01_vlb1800", 1.5),
    ("cand2 fav.80/eb.00/vlb1800", AXES / "grid_fav0.8_eb0.0_vlb1800", 1.5),
    ("cand3 fav.85/eb.01/vlb1800", AXES / "grid_fav0.85_eb0.01_vlb1800", 1.5),
    ("BAH validation (esd0.0)", SAFETYD / "msd2.5_esd0.0__full", 0.0),
]


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
        rows.append(
            dict(
                qid=qid, realized=realized, avg_entry=avg_entry, stake=held_qty * avg_entry, held_to_settle=held_qty > 0
            )
        )
    return pd.DataFrame(rows)


def stress(label: str, fills_dir: Path, esd: float, rng: np.random.Generator) -> dict:
    fills = pd.read_parquet(fills_dir / "fills.parquet")
    pq = per_question(fills)
    if pq.empty:
        return {}
    realized_total = pq["realized"].sum()
    losers = pq[pq["realized"] < 0]["realized"]
    emp_loss = float(-losers.mean()) if len(losers) else 50.0
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
        label=label,
        esd=esd,
        realized=realized_total,
        n_q=len(pq),
        n_win=len(wins),
        emp_loss=emp_loss,
        mean_flips=float(p.sum()),
        ev=float(totals.mean()),
        p5=float(np.percentile(totals, 5)),
        p_loss=float((totals < 0).mean()),
    )


def main():
    rng = np.random.default_rng(SEED)
    rows = [r for r in (stress(lbl, d, e, rng) for lbl, d, e in CONFIGS) if r]
    print(f"\nLoss-injection tail stress (N={N_MC}, HL binary full corpus n=42)")
    print("  flip prob = 1 - avg_entry_price; loss_on_flip = full stake (esd=0) or empirical adverse-exit loss (esd>0)")
    print("=" * 104)
    print(
        f"{'config':>34} {'esd':>4} {'realized':>9} {'wins':>5} {'~flips':>6} "
        f"{'flipLoss':>9} {'EV':>9} {'5th-pct':>9} {'P(loss)':>8}"
    )
    print("-" * 104)
    for r in rows:
        print(
            f"{r['label']:>34} {r['esd']:>4.1f} ${r['realized']:>8.0f} {r['n_win']:>5} "
            f"{r['mean_flips']:>6.2f} ${r['emp_loss']:>8.0f} ${r['ev']:>8.0f} "
            f"${r['p5']:>8.0f} {r['p_loss']:>7.1%}"
        )
    print("=" * 104)


if __name__ == "__main__":
    main()
