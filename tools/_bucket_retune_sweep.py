import json, subprocess, re, sys, os
from pathlib import Path
base = json.loads(Path("/tmp/bucket_base.json").read_text())
def variant(**ov):
    d = dict(base); d.update(ov); return d
CONFIGS = {
    "base":          variant(),
    "esd0":          variant(exit_safety_d=0.0),
    "hold0.04":      variant(exit_spread_hold=0.04),
    "hold0.04_esd0": variant(exit_spread_hold=0.04, exit_safety_d=0.0),
    "hold0.06":      variant(exit_spread_hold=0.06),
    "entry":         variant(entry_spread_gate=True),
    "entry_hold0.04":variant(entry_spread_gate=True, exit_spread_hold=0.04),
}
days = sys.argv[1:] or ["2026-06-07"]
env = {**os.environ, "HLBT_HL_DATA_ROOT":"data", "LOGURU_LEVEL":"ERROR"}
for day in days:
    end = day[:-2] + f"{int(day[-2:])+1:02d}"
    for name, cfg in CONFIGS.items():
        cf = f"/tmp/cfg_{name}.json"; Path(cf).write_text(json.dumps(cfg))
        out = f"data/sim/runs/rt_{name}_{day[-2:]}"
        subprocess.run(["hl-bt","run","--strategy","v3_theta_harvester","--config",cf,
            "--kind","bucket","--data-source","hl_hip4","--ref-source","hl_perp",
            "--ref-event","mark","--reference-ticks","raw","--scan-mode","event",
            "--fee-taker","0.0","--slippage-bps","0","--no-cache",
            "--start",day,"--end",end,"--out-dir",out],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        rep = Path(out)/"report.md"
        pnl=tr=0
        if rep.exists():
            for ln in rep.read_text().splitlines():
                if (m:=re.search(r"total PnL:\s*\$(-?[0-9.]+)",ln)): pnl=float(m.group(1))
                if (m:=re.search(r"trades:\s*(\d+)",ln)): tr=int(m.group(1))
        print(f"{day} {name:16} fills={tr:3} pnl=${pnl:8.2f}")
