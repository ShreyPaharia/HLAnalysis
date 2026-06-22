#!/usr/bin/env python3
"""Loss-injection tail stress for FULL-GRID HYPE-binary candidates.

Same model as run_v31_hl_hype_binary_tail_stress.py (the 2026-06-18 methodology),
but reads the full-grid driver's per-question cell layout
(``<OUT>/<config_id>/qNNNN/fills.parquet``) and concatenates q0..q5 per config.

The full grid's literal winner is a buy-and-hold cell (exit_safety_d=0, maxDD $0) —
the known tail-blind trap. This stress flips each entered-and-held winning favorite
to a loss at its implied rate (full stake under buy-and-hold; empirical adverse-exit
loss when soft exits on) to expose that. VALIDATION = a buy-and-hold cell must show a
deep-negative 5th-pct. Analysis only — n=6, exploratory.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("/Users/shreypaharia/Documents/Projects/Trading/HLAnalysis"
           "/data/sim/runs/v31-hl-hype-binary-fullgrid-2026-06-22")
N_MC = 20000
SEED = 12345

# (config_id, exit_safety_d). esd==0 => buy-and-hold (full-stake flip).
CONFIGS = [
    ("f0.75_eb0.0_vl3600_dt5_tte43200_msd1.5_esd0.0", 0.0),  # grid winner = buy&hold (VALIDATION + reject)
    ("f0.75_eb0.0_vl3600_dt5_tte43200_msd2.5_esd1.0", 1.0),  # best TAIL-SAFE (exits on)
    ("f0.75_eb0.0_vl3600_dt5_tte43200_msd2.5_esd1.5", 1.5),  # more conservative exit
    ("f0.75_eb0.0_vl3600_dt5_tte43200_msd2.0_esd1.0", 1.0),  # lower floor, exits on
    ("f0.85_eb0.02_vl900_dt5_tte43200_msd2.5_esd1.5", 1.5),  # ANCHOR (live BTC config)
]


def load_fills(cid: str) -> pd.DataFrame:
    parts = sorted((OUT / cid).glob("q*/fills.parquet"))
    if not parts:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)


def per_question(fills: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for qid, g in fills.groupby("question_id"):
        g = g[~g["is_hedge"].fillna(False)]
        realized = float(g["realized_pnl_at_settle"].dropna().iloc[0]) if g["realized_pnl_at_settle"].notna().any() else 0.0
        buys = g[(g["side"] == "buy") & (g["cloid"] != "settle")]
        if len(buys) == 0 or buys["size"].sum() <= 0:
            continue
        avg_entry = float(np.average(buys["price"], weights=buys["size"]))
        settle = g[g["cloid"] == "settle"]
        held_qty = float(settle["size"].sum()) if len(settle) else 0.0
        rows.append(dict(qid=qid, realized=realized, avg_entry=avg_entry,
                         stake=held_qty * avg_entry, held_to_settle=held_qty > 0))
    return pd.DataFrame(rows)


def stress(cid: str, esd: float, rng: np.random.Generator) -> dict:
    pq = per_question(load_fills(cid))
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
    return dict(cid=cid, esd=esd, realized=realized_total, n_q=len(pq), n_win=len(wins),
                emp_loss=emp_loss, mean_flips=float(p.sum()),
                ev=float(totals.mean()), p5=float(np.percentile(totals, 5)),
                p_loss=float((totals < 0).mean()))


def main():
    rng = np.random.default_rng(SEED)
    rows = [r for r in (stress(c, e, rng) for c, e in CONFIGS) if r]
    print(f"\nFULL-GRID loss-injection tail stress (N={N_MC}, HYPE binary n=6)")
    print("=" * 118)
    print(f"{'config':>48} {'esd':>4} {'realized':>9} {'wins':>5} {'~flips':>6} {'flipLoss':>9} {'EV':>9} {'5th-pct':>9} {'P(loss)':>8}")
    print("-" * 118)
    for r in rows:
        tag = "  <- buy&hold(VALIDATION)" if r["esd"] == 0 else ("  <- anchor" if r["cid"].startswith("f0.85") else "")
        print(f"{r['cid']:>48} {r['esd']:>4.1f} ${r['realized']:>8.0f} {r['n_win']:>5} "
              f"{r['mean_flips']:>6.2f} ${r['emp_loss']:>8.0f} ${r['ev']:>8.0f} ${r['p5']:>8.0f} {r['p_loss']:>7.1%}{tag}")
    print("=" * 118)


if __name__ == "__main__":
    main()
