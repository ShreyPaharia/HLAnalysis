"""Daily HL sim-vs-live fidelity report — one command, SSM-free.

Ties the whole fidelity loop into a single artifact for side-by-side analysis of
the last fully-settled HL HIP-4 day:

  1. (optional) pull recorded market data + the engine state.db/journal from S3
     (``scripts/pull-data.sh`` + ``scripts/pull-engine.sh``).
  2. discover the day's binary + bucket markets and run the live-faithful sim for
     each (slot, kind) cell (``hl-bt run --slot`` — SHR-99 single config source).
  3. read the LIVE side SSM-free from the pulled ``state.db`` (``fill`` table —
     ``closed_pnl`` includes the settlement fill; ``trade_journal`` for decisions).
  4. compare:
       • PnL (live vs sim, Δ, Σ|sim−live|, sign-match)
       • activity (fills / buy-notional — churn & over-entry)
       • decision fidelity (decision_replay harness: match-rate, input coverage,
         per-field σ/ref/p_model/edge skew, phantom & unmatched actions)
       • book regime (median spread of the traded leg — tight binary vs wide bucket)
       • health/confounds (halts / OOM / stale in the pulled engine logs)
  5. write a markdown report + a machine-readable JSON sidecar.

Run daily:  ``make daily-report``  (or ``uv run python tools/daily_report.py --pull``)
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb

from hlanalysis.backtest.data.hl_hip4 import HLHip4DataSource
from tools.decision_replay_report import load_live, load_sim
from hlanalysis.parity.decision_replay import replay

# Per-class slot config keys (the engine's per-question.klass resolution).
_KIND_KLASS = {"binary": "priceBinary", "bucket": "priceBucket"}
_HALT_KEYS = ("restart_blocked", "daily_loss", "stale", "OOM", "reject_breaker", "FEED STALE")


def _ns(dt: datetime) -> int:
    return int(dt.timestamp() * 1e9)


def _day_dt(day: str) -> datetime:
    return datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def latest_settled_day(data_root: Path) -> str:
    """Latest settlement day D with both corpus partitions D-1 and D present.

    The market settling D 06:00 UTC has book activity on D-1 (06:00→) and D
    (→06:00), so it is fully covered iff both daily partitions exist."""
    base = data_root / "venue=hyperliquid/product_type=prediction_binary/mechanism=clob/event=bbo"
    dates = sorted({m.group(1) for p in base.glob("symbol=*/date=*")
                    if (m := re.search(r"date=(\d{4}-\d{2}-\d{2})", str(p)))})
    if not dates:
        raise SystemExit(f"no recorded HL prediction_binary partitions under {base}")
    have = set(dates)
    for d in reversed(dates):
        prev = (_day_dt(d) - timedelta(days=1)).strftime("%Y-%m-%d")
        if prev in have:
            return d
    raise SystemExit("no day has both its D-1 and D partitions present")


@dataclass
class Leg:
    question_idx: int
    klass: str          # priceBinary | priceBucket
    symbols: tuple[str, ...]


def discover_legs(data_root: Path, day: str) -> dict[str, list[Leg]]:
    """Binary + bucket markets settling on ``day`` (expiry day 06:00 UTC)."""
    src = HLHip4DataSource(data_root=data_root, ref_event="mark", ref_source="hl_perp")
    end = (_day_dt(day) + timedelta(days=1)).strftime("%Y-%m-%d")
    out: dict[str, list[Leg]] = {"binary": [], "bucket": []}
    for q in src.discover(start=day, end=end):
        kind = "binary" if q.klass == "priceBinary" else "bucket"
        out[kind].append(Leg(q.question_idx, q.klass, tuple(q.leg_symbols)))
    return out


def run_sim(slot: str, kind: str, day: str, data_root: Path, out_root: Path, fresh: bool) -> Path:
    run_dir = out_root / f"daily_{slot}_{kind}_{day.replace('-', '')}"
    if run_dir.exists() and not fresh and (run_dir / "report.md").exists():
        return run_dir
    end = (_day_dt(day) + timedelta(days=1)).strftime("%Y-%m-%d")
    cmd = [
        "hl-bt", "run", "--slot", slot, "--slot-class", _KIND_KLASS[kind], "--kind", kind,
        "--data-source", "hl_hip4", "--ref-source", "hl_perp", "--ref-event", "mark",
        "--reference-ticks", "raw", "--scan-mode", "event",
        "--fee-taker", "0.0", "--slippage-bps", "0",
        # --no-cache is load-bearing: the default-ON event-array cache returns WRONG
        # results on HL hip4 (its key misses the reference resample dt → a dt=60 entry
        # is served for a dt=5 run, σ-inflated → far fewer trades; verified 3 vs 74
        # trades on v31 binary 06-10). 4 sims/day is cheap; never trust the cache here.
        "--no-cache",
        "--start", day, "--end", end, "--out-dir", str(run_dir),
    ]
    env_root = {"HLBT_HL_DATA_ROOT": str(data_root)}
    import os
    subprocess.run(cmd, check=True, env={**os.environ, **env_root},
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return run_dir


def live_cell(state_db: Path, legs: tuple[str, ...], w0: int, w1: int) -> tuple[int, float, float]:
    """(n_order_fills, buy_notional, pnl) from the state.db fill table over the window.

    PnL = Σ(closed_pnl − fee), settlement included (HL settlement is a venue fill).
    Order-fill count excludes the settlement row (closed_pnl with size at price 1.0)."""
    import sqlite3
    con = sqlite3.connect(f"file:{state_db}?mode=ro", uri=True)
    qm = ",".join("?" for _ in legs)
    # source='venue' ONLY: the fill table records each fill twice (a 'router' ack row
    # and a 'venue' reconcile row, SHR-72) — summing both double-counts PnL exactly
    # 2× (verified $30.51 vs venue-truth $15.25). venue is the authoritative leg.
    rows = con.execute(
        f"""SELECT side, price, size, fee, closed_pnl FROM fill
            WHERE symbol IN ({qm}) AND ts_ns >= ? AND ts_ns < ? AND source = 'venue'""",
        (*legs, w0, w1),
    ).fetchall()
    con.close()
    pnl = sum((cp or 0.0) - (fee or 0.0) for _, _, _, fee, cp in rows)
    # settlement fills price at ~1.0 with no buy notional; count orders as buys+sells with px<1
    orders = [r for r in rows if not (abs(r[1] - 1.0) < 1e-9 and (r[4] or 0.0) != 0.0)]
    buy_ntl = sum(p * s for side, p, s, _, _ in orders if side == "buy")
    return len(orders), float(buy_ntl), float(pnl)


def sim_cell(run_dir: Path) -> tuple[int, float, float]:
    """(n_trades, buy_notional, pnl) from a run dir."""
    pnl, n = 0.0, 0
    rep = run_dir / "report.md"
    if rep.exists():
        for ln in rep.read_text().splitlines():
            if (m := re.search(r"total PnL:\s*\$(-?[0-9.]+)", ln)):
                pnl = float(m.group(1))
            if (m := re.search(r"trades:\s*(\d+)", ln)):
                n = int(m.group(1))
    buy_ntl = 0.0
    fp = run_dir / "fills.parquet"
    if fp.exists():
        try:
            df = duckdb.connect().execute(
                f"SELECT side, price, size FROM read_parquet('{fp}')").df()
            buy_ntl = float((df[df.side == "buy"].price * df[df.side == "buy"].size).sum())
        except Exception:
            pass
    return n, buy_ntl, pnl


def book_median_spread(data_root: Path, leg: str, w0: int, w1: int) -> float | None:
    g = (f"{data_root}/venue=hyperliquid/product_type=prediction_binary/mechanism=clob/"
         f"event=bbo/symbol={leg}/date=2026-*/hour=all/*.parquet")
    try:
        row = duckdb.connect().execute(
            f"""SELECT median(ask_px - bid_px) m FROM read_parquet('{g}', hive_partitioning=1)
                WHERE exchange_ts >= {w0} AND exchange_ts < {w1}
                  AND bid_px > 0 AND ask_px > 0"""
        ).fetchone()
        return float(row[0]) if row and row[0] is not None else None
    except Exception:
        return None


def health_scan(data_root: Path) -> dict[str, int]:
    """Best-effort halt/OOM/stale counts from pulled engine logs (filtered)."""
    counts = {k: 0 for k in _HALT_KEYS}
    for f in glob.glob(str(data_root / "engine/date=*/engine/log-filtered.gz")) + \
             glob.glob(str(data_root / "engine/date=*/*/gate_decisions.jsonl.gz")):
        try:
            with gzip.open(f, "rt", errors="ignore") as fh:
                text = fh.read()
            for k in _HALT_KEYS:
                counts[k] += text.count(k)
        except Exception:
            continue
    return counts


def build(args) -> dict:
    data_root = Path(args.data_root)
    day = args.day or latest_settled_day(data_root)
    d = _day_dt(day)
    w0, w1 = _ns(d - timedelta(hours=17)), _ns(d + timedelta(hours=7))  # [D-1 07:00, D 07:00)
    legs = discover_legs(data_root, day)
    out_root = data_root / "sim" / "runs"
    out_root.mkdir(parents=True, exist_ok=True)
    slots = args.slots.split(",")

    cells = []
    for slot in slots:
        state_db = data_root / "engine" / f"date={args.engine_date}" / slot / "state.db"
        sdb = state_db if state_db.exists() else None
        for kind in ("binary", "bucket"):
            leg_syms = tuple(s for L in legs[kind] for s in L.symbols)
            qidxs = [L.question_idx for L in legs[kind]]
            run_dir = run_sim(slot, kind, day, data_root, out_root, args.fresh)
            sn, sntl, spnl = sim_cell(run_dir)
            ln, lntl, lpnl = (live_cell(sdb, leg_syms, w0, w1) if sdb else (0, 0.0, 0.0))
            # decision-replay harness (needs the live journal + sim diagnostics)
            harness = None
            if sdb is not None:
                sim_ticks = load_sim(run_dir)
                if sim_ticks:
                    qset = {t.question_idx for t in sim_ticks}
                    tlo = min(t.ts_ns for t in sim_ticks)
                    thi = max(t.ts_ns for t in sim_ticks)
                    live_dec = load_live(sdb, qset, tlo, thi)
                    harness = replay(live_dec, sim_ticks,
                                     ts_tol_ns=int(args.ts_tol_seconds * 1e9)).to_dict()
            spread = None
            if leg_syms:
                sp = [book_median_spread(data_root, s, w0, w1) for s in leg_syms]
                sp = [x for x in sp if x is not None]
                spread = round(min(sp), 4) if sp else None  # favorite leg ≈ tightest
            cells.append({
                "slot": slot, "kind": kind, "question_idxs": qidxs,
                "live": {"fills": ln, "buy_notional": round(lntl, 2), "pnl": round(lpnl, 2)},
                "sim": {"fills": sn, "buy_notional": round(sntl, 2), "pnl": round(spnl, 2)},
                "delta": round(spnl - lpnl, 2),
                "median_spread": spread,
                "harness": harness,
            })
    sum_abs = round(sum(abs(c["delta"]) for c in cells), 2)
    same_sign = sum(1 for c in cells
                    if (c["live"]["pnl"] >= 0) == (c["sim"]["pnl"] >= 0))
    return {
        "day": day, "window_ns": [w0, w1], "slots": slots,
        "cells": cells,
        "sum_abs_delta": sum_abs,
        "same_sign_cells": f"{same_sign}/{len(cells)}",
        "health": health_scan(data_root),
    }


def render_md(rep: dict) -> str:
    L = [f"# HL daily sim-vs-live fidelity — settlement {rep['day']}", ""]
    L.append(f"**Window:** D-1 07:00 → D 07:00 UTC · **slots:** {', '.join(rep['slots'])}")
    L.append(f"**Σ|sim−live|:** ${rep['sum_abs_delta']} · **same-sign cells:** {rep['same_sign_cells']}")
    h = rep["health"]
    flagged = {k: v for k, v in h.items() if v}
    L.append(f"**Health/confounds:** {flagged or 'clean (no halt/OOM/stale in pulled logs)'}")
    L += ["", "## PnL + activity", "",
          "| slot | kind | live $ | sim $ | Δ | live fills/ntl | sim fills/ntl | med spread |",
          "| :-- | :-- | --: | --: | --: | --: | --: | --: |"]
    for c in rep["cells"]:
        lv, sm = c["live"], c["sim"]
        L.append(f"| {c['slot']} | {c['kind']} | {lv['pnl']:.2f} | {sm['pnl']:.2f} | "
                 f"{c['delta']:+.2f} | {lv['fills']}/${lv['buy_notional']:.0f} | "
                 f"{sm['fills']}/${sm['buy_notional']:.0f} | "
                 f"{c['median_spread'] if c['median_spread'] is not None else '—'} |")
    L += ["", "## Decision fidelity (replay harness)", "",
          "| slot | kind | match | input-cov | σ rel | ref rel | p_model |Δ| | phantom | unmatched |",
          "| :-- | :-- | --: | --: | --: | --: | --: | --: | --: |"]
    for c in rep["cells"]:
        hn = c["harness"]
        if not hn:
            L.append(f"| {c['slot']} | {c['kind']} | — | — | — | — | — | — | — |")
            continue
        fs = hn["field_skews"]
        def rel(f):
            v = fs[f]["median_rel"]
            return f"{v:.2%}" if v is not None else "—"
        L.append(
            f"| {c['slot']} | {c['kind']} | "
            f"{hn['n_live_matched']}/{hn['n_live']} ({hn['decision_match_rate']:.0%}) | "
            f"{hn['n_input_comparable']}/{hn['n_live']} | {rel('sigma')} | {rel('reference_price')} | "
            f"{fs['p_model']['median_abs']:.4g} | {hn['n_sim_phantom']} | {len(hn['unmatched_live'])} |")
    L += ["", "## Read",
          "- **Inputs vs cadence:** where input-cov is high and σ/ref skew tiny, the decision "
          "*inputs* are faithful; a low match-rate then means the sim and live SCANNED at "
          "different instants (cadence/execution, SHR-89), not that inputs disagree.",
          "- **Phantom sim actions** = over-entry (SHR-91); **bucket Δ with same-sign** but large "
          "magnitude = doom-loop / spread crossing (SHR-102).",
          "- Treat any non-empty Health line as a live-side confound (the sim assumes 100% engine uptime)."]
    return "\n".join(L)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default="data")
    p.add_argument("--day", default=None, help="settlement day YYYY-MM-DD (default: latest settled)")
    p.add_argument("--engine-date", default=None,
                   help="engine snapshot date= partition to read state.db from (default: --day)")
    p.add_argument("--slots", default="v1,v31")
    p.add_argument("--ts-tol-seconds", type=float, default=2.0)
    p.add_argument("--pull", action="store_true", help="run pull-data.sh + pull-engine.sh first")
    p.add_argument("--fresh", action="store_true", help="re-run sims even if a run dir exists")
    p.add_argument("--out", default=None, help="markdown output path (default docs/research/<day>-hl-daily-report.md)")
    args = p.parse_args(argv)

    if args.pull:
        here = Path(__file__).resolve().parent.parent
        for sh in ("scripts/pull-data.sh", "scripts/pull-engine.sh"):
            print(f"==> {sh}")
            subprocess.run(["bash", str(here / sh)], check=True)

    if args.engine_date is None:
        # default: the engine snapshot taken the morning AFTER settlement holds the day's fills.
        args.engine_date = None  # resolved per available snapshot below
    if args.engine_date is None:
        snaps = sorted({m.group(1) for p in glob.glob(str(Path(args.data_root) / "engine/date=*"))
                        if (m := re.search(r"date=(\d{4}-\d{2}-\d{2})", p))})
        args.engine_date = snaps[-1] if snaps else (args.day or "")

    rep = build(args)
    day = rep["day"]
    out = Path(args.out) if args.out else Path(f"docs/research/{day}-hl-daily-report.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    md = render_md(rep)
    out.write_text(md)
    out.with_suffix(".json").write_text(json.dumps(rep, indent=2))
    print(md)
    print(f"\nwrote {out} (+ .json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
