"""HL v31 BINARY estimator A/B: sample_std (live) vs bipower, all settlement days.

The live HL v31 binary runs vol_estimator=sample_std. bipower is jump-robust and
flips knife-edge entry timing on vol-burst days (memory: 06-08 one market
-$4.72 vs +$58). This sweep settles the open question with a full walk over EVERY
recorded HL settlement day, holding ALL other params equal and varying ONLY the
estimator. Caps held equal (off) on both arms, so the delta isolates the estimator.

Usage: HLBT_HL_DATA_ROOT=<main>/data uv run python tools/_estimator_sweep.py
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(os.environ["HLBT_HL_DATA_ROOT"])
OUTROOT = Path("data")
BBO = ROOT / "venue=hyperliquid/product_type=prediction_binary/mechanism=clob/event=bbo"
# KIND=binary|bucket; BASE_JSON points at the dumped slot params for that cell.
KIND = os.environ.get("EST_KIND", "binary")
BASE = json.loads(Path(os.environ.get(
    "EST_BASE_JSON", "/tmp/v31_binary_base.json")).read_text())
assert BASE.get("vol_estimator", "sample_std") == "sample_std", BASE.get("vol_estimator")
BIPOWER = {**BASE, "vol_estimator": "bipower"}
CONCURRENCY = 6
_TAG = KIND[:3]


def settlement_days() -> list[str]:
    dates = sorted({m.group(1) for p in BBO.glob("symbol=*/date=*")
                    if (m := re.search(r"date=(\d{4}-\d{2}-\d{2})", str(p)))})
    have = set(dates)
    return [d for d in dates
            if (date.fromisoformat(d) - timedelta(days=1)).isoformat() in have]


def run_cell(day: str, name: str, cfg: dict) -> tuple[str, str, int, float, float]:
    end = (date.fromisoformat(day) + timedelta(days=1)).isoformat()
    cf = f"/tmp/est_{_TAG}_{name}.json"
    Path(cf).write_text(json.dumps(cfg))
    out = OUTROOT / "sim/runs" / f"est_{_TAG}_{name}_{day}"
    env = {**os.environ, "HLBT_HL_DATA_ROOT": str(ROOT), "LOGURU_LEVEL": "ERROR"}
    subprocess.run(
        ["hl-bt", "run", "--strategy", "v3_theta_harvester", "--config", cf,
         "--kind", KIND, "--data-source", "hl_hip4", "--ref-source", "hl_perp",
         "--ref-event", "mark", "--reference-ticks", "raw", "--scan-mode", "event",
         "--fee-taker", "0.0", "--slippage-bps", "0", "--no-cache",
         "--start", day, "--end", end, "--out-dir", str(out)],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
    )
    rep = out / "report.md"
    pnl = dd = 0.0
    tr = 0
    if rep.exists():
        for ln in rep.read_text().splitlines():
            if (m := re.search(r"total PnL:\s*\$(-?[0-9.]+)", ln)):
                pnl = float(m.group(1))
            if (m := re.search(r"trades:\s*(\d+)", ln)):
                tr = int(m.group(1))
            if (m := re.search(r"max drawdown:\s*\$(-?[0-9.]+)", ln)):
                dd = float(m.group(1))
    return day, name, tr, pnl, dd


def main() -> None:
    days = settlement_days()
    print(f"settlement days: {len(days)}  ({days[0]} .. {days[-1]})")
    jobs = [(d, "ss", BASE) for d in days] + [(d, "bp", BIPOWER) for d in days]
    res: dict[tuple[str, str], tuple[int, float, float]] = {}
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        for day, name, tr, pnl, dd in ex.map(lambda a: run_cell(*a), jobs):
            res[(day, name)] = (tr, pnl, dd)

    print(f"\n{'day':12} {'sample_std f/pnl':>18} {'bipower f/pnl':>18} {'Δpnl':>9} {'flag':>10}")
    tss = tbp = fss = fbp = 0.0
    worse = []
    better = []
    for d in days:
        st, sp, sdd = res.get((d, "ss"), (0, 0.0, 0.0))
        bt, bp, bdd = res.get((d, "bp"), (0, 0.0, 0.0))
        tss += sp; tbp += bp; fss += st; fbp += bt
        dp = bp - sp
        flag = ""
        if dp > 0.01:
            flag = "BETTER"; better.append((d, sp, bp))
        elif dp < -0.01:
            flag = "WORSE"; worse.append((d, sp, bp))
        print(f"{d:12} {st:6}/{sp:10.2f} {bt:6}/{bp:10.2f} {dp:+9.2f} {flag:>10}")
    print(f"\n{'TOTAL':12} {int(fss):6}/{tss:10.2f} {int(fbp):6}/{tbp:10.2f} {tbp-tss:+9.2f}")
    print(f"churn: sample_std {int(fss)} fills -> bipower {int(fbp)} fills ({int(fbp-fss):+d})")
    print(f"days bipower BETTER: {len(better)}  WORSE: {len(worse)}  "
          f"flat: {len(days)-len(better)-len(worse)}")
    if worse:
        print("\nbipower WORSE on:")
        for d, sp, bp in worse:
            print(f"  {d}: {sp:.2f} -> {bp:.2f} ({bp-sp:+.2f})")
    if better:
        print("\nbipower BETTER on:")
        for d, sp, bp in better:
            print(f"  {d}: {sp:.2f} -> {bp:.2f} ({bp-sp:+.2f})")


if __name__ == "__main__":
    main()
