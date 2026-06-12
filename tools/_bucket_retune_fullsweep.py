"""Full-history HL v31 bucket retune sweep: baseline vs exit_spread_hold=0.04.

Runs the v31 priceBucket cell (--no-cache) for EVERY recorded settlement day and
compares the current config against +exit_spread_hold=0.04. Flags any day where the
gate HURTS (the regression test the operator asked for). Parallelized across cores.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

# Corpus root (read-only) defaults to env HLBT_HL_DATA_ROOT — point it at the main
# worktree's full data/ so we don't duplicate 19 GB. Output run dirs stay local.
ROOT = Path(os.environ.get("HLBT_HL_DATA_ROOT", "data"))
OUTROOT = Path("data")
BBO = ROOT / "venue=hyperliquid/product_type=prediction_binary/mechanism=clob/event=bbo"
BASE = json.loads(Path("/tmp/bucket_base.json").read_text())
HOLD = {**BASE, "exit_spread_hold": 0.04}
CONCURRENCY = 8


def settlement_days() -> list[str]:
    dates = sorted({m.group(1) for p in BBO.glob("symbol=*/date=*")
                    if (m := re.search(r"date=(\d{4}-\d{2}-\d{2})", str(p)))})
    have = set(dates)
    out = []
    for d in dates:
        prev = (date.fromisoformat(d) - timedelta(days=1)).isoformat()
        if prev in have:
            out.append(d)
    return out


def run_cell(day: str, name: str, cfg: dict) -> tuple[str, str, int, float]:
    end = (date.fromisoformat(day) + timedelta(days=1)).isoformat()
    cf = f"/tmp/fs_{name}.json"
    Path(cf).write_text(json.dumps(cfg))
    out = OUTROOT / "sim/runs" / f"fs_{name}_{day}"
    env = {**os.environ, "HLBT_HL_DATA_ROOT": str(ROOT), "LOGURU_LEVEL": "ERROR"}
    subprocess.run(
        ["hl-bt", "run", "--strategy", "v3_theta_harvester", "--config", cf,
         "--kind", "bucket", "--data-source", "hl_hip4", "--ref-source", "hl_perp",
         "--ref-event", "mark", "--reference-ticks", "raw", "--scan-mode", "event",
         "--fee-taker", "0.0", "--slippage-bps", "0", "--no-cache",
         "--start", day, "--end", end, "--out-dir", str(out)],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
    )
    rep = out / "report.md"
    pnl = tr = 0
    if rep.exists():
        for ln in rep.read_text().splitlines():
            if (m := re.search(r"total PnL:\s*\$(-?[0-9.]+)", ln)):
                pnl = float(m.group(1))
            if (m := re.search(r"trades:\s*(\d+)", ln)):
                tr = int(m.group(1))
    return day, name, tr, pnl


def main() -> None:
    days = settlement_days()
    print(f"settlement days: {len(days)}  ({days[0]} .. {days[-1]})")
    jobs = [(d, "base", BASE) for d in days] + [(d, "hold", HOLD) for d in days]
    res: dict[tuple[str, str], tuple[int, float]] = {}
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        for day, name, tr, pnl in ex.map(lambda a: run_cell(*a), jobs):
            res[(day, name)] = (tr, pnl)

    print(f"\n{'day':12} {'base fills/pnl':>18} {'hold fills/pnl':>18} {'Δpnl':>9} {'flag':>6}")
    tb = th = fb = fh = 0
    regressions = []
    for d in days:
        bt, bp = res.get((d, "base"), (0, 0.0))
        ht, hp = res.get((d, "hold"), (0, 0.0))
        tb += bp; th += hp; fb += bt; fh += ht
        dp = hp - bp
        flag = ""
        if hp < bp - 0.01:
            flag = "WORSE"; regressions.append((d, bp, hp))
        elif ht < bt and abs(dp) < 0.01:
            flag = "less-churn"
        elif dp > 0.01:
            flag = "BETTER"
        print(f"{d:12} {bt:6}/{bp:10.2f} {ht:6}/{hp:10.2f} {dp:+9.2f} {flag:>10}")
    print(f"\n{'TOTAL':12} {fb:6}/{tb:10.2f} {fh:6}/{th:10.2f} {th-tb:+9.2f}")
    print(f"churn: base {fb} fills -> hold {fh} fills  ({fh-fb:+d})")
    if regressions:
        print(f"\nREGRESSIONS ({len(regressions)} days hold<base):")
        for d, bp, hp in regressions:
            print(f"  {d}: {bp:.2f} -> {hp:.2f} ({hp-bp:+.2f})")
    else:
        print("\nNO REGRESSIONS — exit_spread_hold never hurts on any recorded day.")


if __name__ == "__main__":
    main()
