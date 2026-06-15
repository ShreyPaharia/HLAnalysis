"""SHR-102 A/B: quantify the `entry_spread_gate` lever on recorded HL data.

Backtest-only. Flips ONLY `entry_spread_gate` true/false on the v31 priceBucket
and priceBinary cells, per settlement day, and reports per-day + aggregate PnL,
fills (churn), hit-rate, max DD, plus how many scan-ticks the gate fired on and on
which days. Plus an isolation arm that zeroes the already-deployed
`exit_spread_hold` to test the entry gate standalone.

Live-faithful flags (mirror the SHR-102 sibling `tools/_bucket_retune_fullsweep.py`
and the engine cadence): raw `mark` reference ticks, event scan (0.2/2.0s), order
latency 50ms, fee 0 (HL HIP-4 is fee-free), slippage 0.

SPEED: each per-day worker runs IN ONE PROCESS and replays every arm against the
SAME DataSource instance, so the per-question event-array bundle is built ONCE
(DuckDB) and reused across arms via the in-process memo. Days run 8-wide.

Usage:
  HLBT_HL_DATA_ROOT=/path uv run python tools/_entry_spread_gate_ab.py [N_DAYS]
  (worker, internal):     uv run python tools/_entry_spread_gate_ab.py worker DAY KIND OUT
N_DAYS limits to the last N settlement days (smoke). Default: all.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import duckdb

ROOT = Path(os.environ.get("HLBT_HL_DATA_ROOT", "data"))
OUTROOT = Path(os.environ.get("HLBT_AB_OUTROOT", "/tmp/shr102/runs"))
BBO = ROOT / "venue=hyperliquid/product_type=prediction_binary/mechanism=clob/event=bbo"
CONCURRENCY = int(os.environ.get("HLBT_AB_CONCURRENCY", "8"))

BUCKET_BASE = json.loads(Path("/tmp/shr102/bucket_base.json").read_text())
BINARY_BASE = json.loads(Path("/tmp/shr102/binary_base.json").read_text())

# Arm name -> (kind, config). entry_spread_gate is the ONLY flipped knob within
# each {base, gate} pair. The *_nohold pair additionally zeroes the deployed
# exit_spread_hold to isolate the entry gate's standalone value.
ARMS = {
    "bk_base": ("bucket", BUCKET_BASE),
    "bk_gate": ("bucket", {**BUCKET_BASE, "entry_spread_gate": True}),
    "bk_nohold_base": ("bucket", {**BUCKET_BASE, "exit_spread_hold": 0.0}),
    "bk_nohold_gate": ("bucket", {**BUCKET_BASE, "exit_spread_hold": 0.0, "entry_spread_gate": True}),
    "bn_base": ("binary", BINARY_BASE),
    "bn_gate": ("binary", {**BINARY_BASE, "entry_spread_gate": True}),
}
ARMS_BY_KIND = {
    "bucket": ["bk_base", "bk_gate", "bk_nohold_base", "bk_nohold_gate"],
    "binary": ["bn_base", "bn_gate"],
}


def settlement_days() -> list[str]:
    dates = sorted({m.group(1) for p in BBO.glob("symbol=*/date=*")
                    if (m := re.search(r"date=(\d{4}-\d{2}-\d{2})", str(p)))})
    have = set(dates)
    return [d for d in dates
            if (date.fromisoformat(d) - timedelta(days=1)).isoformat() in have]


# --------------------------------------------------------------------------- #
# Worker: one process per (day, kind). Builds the source ONCE, replays arms.
# --------------------------------------------------------------------------- #
def _make_args(kind: str, day: str, end: str, cfg_path: str) -> argparse.Namespace:
    return argparse.Namespace(
        data_source="hl_hip4", cache_root=str(ROOT), config=cfg_path, slot=None,
        ref_source="hl_perp", ref_event="mark", reference_ticks="raw",
        scanner_interval_seconds=60, tick_size=0.001, lot_size=1.0,
        slippage_bps=0.0, fee_taker=0.0, fee_model="flat", fee_rate=0.0,
        depth=None, order_latency_ms=50.0, scan_mode="event",
        scan_min_interval_seconds=0.2, scan_max_interval_seconds=2.0,
        min_inter_order_seconds=0.0, ioc_fleeting_persistence_seconds=0.0,
        reference_warmup_seconds=None, kind=kind, start=day, end=end,
        skip_markets=0, max_markets=None, hedge_data_path=None,
        strategy="v3_theta_harvester",
    )


def run_worker(day: str, kind: str, out: Path) -> None:
    from hlanalysis.backtest._cli_plumbing import (
        _build_strategy_for_cli,
        _resolve_reference_warmup_seconds,
        _run_config_from_args,
        _source_config_from_args,
        _strike_for_data_source,
    )
    from hlanalysis.backtest.runner.parallel import run_questions_parallel
    from hlanalysis.marketdata.decision_input import from_backtest_params

    end = (date.fromisoformat(day) + timedelta(days=1)).isoformat()
    arms = ARMS_BY_KIND[kind]
    # Source config is identical across the arms of one kind (they share dt /
    # vol_lookback), so build the DataSource ONCE and reuse it for every arm —
    # the per-question bundle memo then makes arms 2..N skip the DuckDB build.
    base_cfg = ARMS[arms[0]][1]
    cfgp = out / "cfg_src.json"
    cfgp.parent.mkdir(parents=True, exist_ok=True)
    cfgp.write_text(json.dumps(base_cfg))
    args = _make_args(kind, day, end, str(cfgp))
    warmup = _resolve_reference_warmup_seconds(base_cfg, data_source="hl_hip4", cli_override=None)
    resolved = from_backtest_params(base_cfg, track_default_source="mark")
    source_config = _source_config_from_args(
        args, reference_resample_seconds=int(base_cfg.get("vol_sampling_dt_seconds", 60)),
        reference_warmup_seconds=warmup, resolved=resolved)
    data_source = source_config.build()
    discover_kwargs = {"kinds": ("priceBinary" if kind == "binary" else "priceBucket",)}
    descriptors = list(data_source.discover(start=day, end=end, **discover_kwargs))
    run_cfg = _run_config_from_args(args, None)
    strike_fn = _strike_for_data_source("hl_hip4")

    result = {"day": day, "kind": kind, "n_questions": len(descriptors), "arms": {}}
    for arm in arms:
        cfg = ARMS[arm][1]
        diag = out / f"{arm}_diag"
        fills = out / f"{arm}_fills"
        strategy = _build_strategy_for_cli("v3_theta_harvester", dict(cfg))
        qres = run_questions_parallel(
            descriptors=descriptors, strategy_id="v3_theta_harvester", params=dict(cfg),
            run_cfg=run_cfg, source_config=source_config, diagnostics_dir=diag,
            fills_dir=fills, strike_for=strike_fn, hedge_data_path=None,
            hedge_half_spread_bps=1.0, n_workers=1, data_source=data_source, strategy=strategy)
        pnls = [r.realized_pnl_usd for r in qres]
        fills_n = sum(r.n_fills for r in qres)
        total = sum(pnls)
        wins = sum(1 for p in pnls if p > 0)
        nz = sum(1 for p in pnls if abs(p) > 1e-9)
        # max drawdown over per-question cumulative
        cum = peak = mdd = 0.0
        for p in pnls:
            cum += p
            peak = max(peak, cum)
            mdd = max(mdd, peak - cum)
        fires = fireq = 0
        dglob = diag / "*.parquet"
        try:
            row = duckdb.sql(
                f"SELECT count(*), count(DISTINCT question_id) FROM read_parquet('{dglob}') "
                f"WHERE reason='entry_spread_too_wide'").fetchone()
            fires, fireq = int(row[0]), int(row[1])
        except Exception:
            pass
        result["arms"][arm] = {
            "pnl": round(total, 4), "fills": fills_n, "n_q": len(pnls),
            "n_traded": nz, "wins": wins,
            "hit": round(wins / nz, 4) if nz else 0.0,
            "maxdd": round(mdd, 4), "fires": fires, "fireq": fireq,
        }
    (out / "result.json").write_text(json.dumps(result, indent=2))


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run_cell(day: str, kind: str) -> tuple[str, str, dict]:
    out = OUTROOT / f"{kind}_{day}"
    env = {**os.environ, "LOGURU_LEVEL": "ERROR"}
    subprocess.run(
        [sys.executable, __file__, "worker", day, kind, str(out)],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    rj = out / "result.json"
    return day, kind, (json.loads(rj.read_text()) if rj.exists() else {"arms": {}})


def _fmt_block(days, res, base, gate, title):
    print(f"\n=== {title} ===")
    print(f"{'day':12} {'base f/pnl':>16} {'gate f/pnl':>16} {'Δpnl':>8} {'Δfills':>7} {'fires(q)':>9}")
    tb = tg = 0.0
    fb = fg = 0
    deltas = []
    for d in days:
        a = res.get(d, {}).get("arms", {})
        rb = a.get(base, {"pnl": 0.0, "fills": 0})
        rg = a.get(gate, {"pnl": 0.0, "fills": 0, "fires": 0, "fireq": 0})
        bp, bt = rb["pnl"], rb["fills"]
        gp, gt = rg["pnl"], rg["fills"]
        tb += bp
        tg += gp
        fb += bt
        fg += gt
        deltas.append((d, gp - bp, gt - bt))
        mark = " *" if abs(gp - bp) > 0.01 or gt != bt else ""
        print(f"{d:12} {bt:5d}/{bp:9.2f} {gt:5d}/{gp:9.2f} {gp-bp:+8.2f} {gt-bt:+7d} "
              f"{rg.get('fires',0):5d}({rg.get('fireq',0)}){mark}")
    print(f"{'TOTAL':12} {fb:5d}/{tb:9.2f} {fg:5d}/{tg:9.2f} {tg-tb:+8.2f} {fg-fb:+7d}")
    h = len(deltas) // 2
    s1 = sum(x[1] for x in deltas[:h])
    s2 = sum(x[1] for x in deltas[h:])
    c1 = sum(x[2] for x in deltas[:h])
    c2 = sum(x[2] for x in deltas[h:])
    sign = "trivial(both~0)" if abs(s1) < 0.01 and abs(s2) < 0.01 else (
        "YES" if s1 * s2 > 0 else "NO")
    print(f"split-half PnLΔ: H1={s1:+.2f} ({days[0]}..{days[h-1]})  "
          f"H2={s2:+.2f} ({days[h]}..{days[-1]})  sign-stable={sign}")
    print(f"split-half fillsΔ: H1={c1:+d}  H2={c2:+d}")


def main_driver() -> None:
    days = settlement_days()
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        days = days[-int(sys.argv[1]):]
    print(f"settlement days: {len(days)}  ({days[0]} .. {days[-1]})", flush=True)
    jobs = [(d, k) for d in days for k in ("bucket", "binary")]
    res: dict[str, dict] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        for day, kind, r in ex.map(lambda a: run_cell(*a), jobs):
            res.setdefault(day, {}).setdefault("arms", {}).update(r.get("arms", {}))
            res[day]["n_questions"] = res[day].get("n_questions", {})
            done += 1
            print(f"  [{done}/{len(jobs)}] {kind} {day} done", flush=True)
    Path("/tmp/shr102/ab_results.json").write_text(json.dumps(res, indent=2))
    _fmt_block(days, res, "bk_base", "bk_gate",
               "BUCKET — current live config (incl exit_spread_hold=0.04)")
    _fmt_block(days, res, "bk_nohold_base", "bk_nohold_gate",
               "BUCKET — exit_spread_hold=0.0 (isolate entry gate)")
    _fmt_block(days, res, "bn_base", "bn_gate", "BINARY control")


if __name__ == "__main__":
    if len(sys.argv) >= 5 and sys.argv[1] == "worker":
        run_worker(sys.argv[2], sys.argv[3], Path(sys.argv[4]))
    else:
        main_driver()
