#!/usr/bin/env python3
"""Loss-injection tail stress for the HYPE-binary v31/theta candidates.

Mirrors run_v31_hl_safetyd_tail_stress.py / the 2026-06-18 methodology
(docs/research/2026-06-18-hl-buy-and-hold-tail-stress.md). The HYPE corpus is
TINY (n=6) and already mostly losers, so the realized numbers carry almost no tail
information — this stress is run for completeness + to validate the model against a
buy-and-hold collapse cell, NOT as a promotion gate.

Model: each favorite the config ENTERED and HELD to settlement as a WIN is, with
probability = its own implied loss rate (1 - weighted-avg entry price), flipped to
a loss. A flipped favorite loses the FULL stake under buy-and-hold (exit_safety_d
== 0) or the config's empirical adverse-exit loss when soft exits are on. Realized
losers are kept as-is. Monte-Carlo N runs -> EV, 5th-pct, P(net loss).

Reads the per-config full-corpus fills.parquet produced by
run_v31_hl_hype_binary_sweep.py.

VALIDATION: msd2.0_esd0.0 (buy-and-hold) should show the worst tail (full-stake
flips). If the flip model reproduces a relative collapse vs the exits-on configs,
the exits-on numbers are trustworthy.

Analysis only. No config change, no deploy.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("/Users/shreypaharia/Documents/Projects/Trading/HLAnalysis"
           "/data/sim/runs/v31-hl-hype-binary-2026-06-22")
N_MC = 20000
SEED = 12345

# (label, exit_safety_d). exit_safety_d==0 => buy-and-hold (full-stake flip).
# Anchor + top sweep candidates + the buy-and-hold validation cell.
CONFIGS = [
    ("prod_ref", 1.5),        # BTC-tuned anchor (msd2.5/esd1.5)
    ("msd3.0_esd1.5", 1.5),   # grid winner by worst-half
    ("msd3.0_esd2.0", 2.0),
    ("msd2.0_esd1.5", 1.5),   # anchor-adjacent (lower entry floor)
    ("msd2.0_esd0.0", 0.0),   # VALIDATION: buy-and-hold, full-stake flips
]


def per_question(fills: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for qid, g in fills.groupby("question_id"):
        g = g[~g["is_hedge"].fillna(False)]
        realized = float(g["realized_pnl_at_settle"].dropna().iloc[0]) if g["realized_pnl_at_settle"].notna().any() else 0.0
        buys = g[(g["side"] == "buy") & (g["cloid"] != "settle")]
        if len(buys) == 0 or buys["size"].sum() <= 0:
            continue  # never entered
        avg_entry = float(np.average(buys["price"], weights=buys["size"]))
        settle = g[g["cloid"] == "settle"]
        held_qty = float(settle["size"].sum()) if len(settle) else 0.0
        stake = held_qty * avg_entry
        rows.append(dict(qid=qid, realized=realized, avg_entry=avg_entry,
                         stake=stake, held_to_settle=held_qty > 0))
    return pd.DataFrame(rows)


def stress(label: str, esd: float, rng: np.random.Generator) -> dict:
    fp = OUT / f"{label}__full" / "fills.parquet"
    if not fp.exists():
        return {}
    pq = per_question(pd.read_parquet(fp))
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
    contrib = np.where(flips, -loss, win_pnl)
    totals = fixed + contrib.sum(axis=1)
    return dict(label=label, esd=esd, realized=realized_total,
                n_q=len(pq), n_win=len(wins), emp_loss=emp_loss,
                mean_flips=float(p.sum()),
                ev=float(totals.mean()), p5=float(np.percentile(totals, 5)),
                p_loss=float((totals < 0).mean()))


def main():
    rng = np.random.default_rng(SEED)
    rows = [r for r in (stress(l, e, rng) for l, e in CONFIGS) if r]
    print(f"\nLoss-injection tail stress (N={N_MC}, HYPE binary full corpus n=6)")
    print("  flip prob = 1 - avg_entry_price; loss_on_flip = full stake (esd=0) "
          "or empirical adverse-exit loss (esd>0)")
    print("=" * 96)
    h = (f"{'config':>16} {'esd':>4} {'realized':>9} {'wins':>5} {'~flips':>6} "
         f"{'flipLoss':>9} {'EV':>9} {'5th-pct':>9} {'P(loss)':>8}")
    print(h); print("-" * 96)
    for r in rows:
        tag = "  <- anchor" if r["label"] == "prod_ref" else ("  <- VALIDATION(BAH)" if r["esd"] == 0 else "")
        print(f"{r['label']:>16} {r['esd']:>4.1f} ${r['realized']:>8.0f} {r['n_win']:>5} "
              f"{r['mean_flips']:>6.2f} ${r['emp_loss']:>8.0f} ${r['ev']:>8.0f} "
              f"${r['p5']:>8.0f} {r['p_loss']:>7.1%}{tag}")
    print("=" * 96)


if __name__ == "__main__":
    main()
