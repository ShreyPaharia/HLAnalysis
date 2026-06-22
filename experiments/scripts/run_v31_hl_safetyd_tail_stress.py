#!/usr/bin/env python3
"""Loss-injection tail stress for the HL binary safety_d band candidates.

Reproduces the 2026-06-18 methodology (docs/research/2026-06-18-hl-buy-and-hold-
tail-stress.md): the realized HL corpus is tail-deficient (favorites mostly resolve
ITM), so realized maxDD/worst-half understate tail risk. Model: each favorite the
config ENTERED and HELD to settlement as a WIN is, with probability = its own
implied loss rate (1 - weighted-avg entry price), flipped to a loss. A flipped
favorite loses:
  * the FULL stake under buy-and-hold (exit_safety_d == 0, no protective exit), or
  * the config's EMPIRICAL adverse-exit loss (mean realized loss on the questions
    it entered and lost) when the soft exit is on (exit_safety_d > 0).
Realized losers are kept as-is. Monte-Carlo N runs -> EV, 5th-pct, P(net loss).

Reads the per-config full-corpus fills.parquet already produced by
run_v31_hl_safetyd_sweep.py (data/sim/runs/v31-hl-safetyd-sweep-2026-06-22).

VALIDATION: msd2.0_esd0.0 (buy-and-hold) must reproduce the doc's collapse
(EV near 0, P(loss) ~40-50%, deep-negative 5th-pct). If it does, the exits-on
candidate numbers are trustworthy.

Analysis only. No config change, no deploy.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("/Users/shreypaharia/Documents/Projects/Trading/HLAnalysis"
           "/data/sim/runs/v31-hl-safetyd-sweep-2026-06-22")
N_MC = 20000
SEED = 12345

# (label, exit_safety_d) — exit_safety_d==0 => buy-and-hold (full-stake flip).
CONFIGS = [
    ("msd2.0_esd1.0", 1.0),   # current live
    ("msd2.5_esd1.0", 1.0),   # raise entry floor only
    ("msd2.5_esd1.5", 1.5),
    ("msd2.0_esd2.0", 2.0),
    ("msd2.5_esd2.0", 2.0),   # headline candidate
    ("msd2.0_esd0.0", 0.0),   # VALIDATION: buy-and-hold, must collapse
]


def per_question(fills: pd.DataFrame) -> pd.DataFrame:
    """Collapse fills -> one row per question: realized PnL, held-to-settle win
    flag, weighted-avg entry price, stake (held qty * avg entry price)."""
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
        held_to_settle = held_qty > 0
        stake = held_qty * avg_entry
        rows.append(dict(qid=qid, realized=realized, avg_entry=avg_entry,
                         stake=stake, held_to_settle=held_to_settle))
    return pd.DataFrame(rows)


def stress(label: str, esd: float, rng: np.random.Generator) -> dict:
    fills = pd.read_parquet(OUT / f"{label}__full" / "fills.parquet")
    pq = per_question(fills)
    if pq.empty:
        return {}
    realized_total = pq["realized"].sum()

    # Empirical adverse-exit loss: mean realized loss on entered-and-lost questions.
    losers = pq[pq["realized"] < 0]["realized"]
    emp_loss = float(-losers.mean()) if len(losers) else 50.0

    # Flippable set: favorites held to settlement that WON (realized > 0).
    wins = pq[(pq["held_to_settle"]) & (pq["realized"] > 0)].copy()
    wins["p_flip"] = (1.0 - wins["avg_entry"]).clip(0.01, 0.5)
    if esd <= 0.0:
        wins["loss_on_flip"] = wins["stake"]          # full-stake blowup
    else:
        wins["loss_on_flip"] = emp_loss               # capped by soft exit
    fixed = realized_total - wins["realized"].sum()   # losers + soft-exit round-trips

    p = wins["p_flip"].to_numpy()
    win_pnl = wins["realized"].to_numpy()
    loss = wins["loss_on_flip"].to_numpy()
    flips = rng.random((N_MC, len(wins))) < p          # (N_MC, n_wins)
    # each win contributes win_pnl if not flipped, else -loss
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
    print(f"\nLoss-injection tail stress (N={N_MC}, HL binary full corpus n=42)")
    print(f"  flip prob = 1 - avg_entry_price; loss_on_flip = full stake (esd=0) "
          f"or empirical adverse-exit loss (esd>0)")
    print("=" * 94)
    h = f"{'config':>16} {'esd':>4} {'realized':>9} {'wins':>5} {'~flips':>6} {'flipLoss':>9} {'EV':>9} {'5th-pct':>9} {'P(loss)':>8}"
    print(h); print("-" * 94)
    for r in rows:
        tag = "  <- live" if r["label"] == "msd2.0_esd1.0" else ("  <- VALIDATION(BAH)" if r["esd"] == 0 else "")
        print(f"{r['label']:>16} {r['esd']:>4.1f} ${r['realized']:>8.0f} {r['n_win']:>5} "
              f"{r['mean_flips']:>6.2f} ${r['emp_loss']:>8.0f} ${r['ev']:>8.0f} "
              f"${r['p5']:>8.0f} {r['p_loss']:>7.1%}{tag}")
    print("=" * 94)


if __name__ == "__main__":
    main()
