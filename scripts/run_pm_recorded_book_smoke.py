#!/usr/bin/env python3
"""SYNTHETIC vs RECORDED PM book-fill smoke eval (HL-parity feature).

Runs the v1 PM prod slot (`v1_late_resolution`, pm_binary fees) on the small
set of BTC Up/Down binary markets that have native recorded L2 `book_snapshot`
coverage (recorder is new — coverage starts 2026-05-27, so only ~2 markets),
once with `--pm-book-source synthetic` and once with `recorded`, and reports
per-mode: n_markets, trades, total PnL, hit rate, max drawdown, and average
taker fill price vs leg mid (slippage in ¢).

This is a CAPABILITY + smoke-eval, NOT a strategy decision — the window is far
too small to be load-bearing.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[1]
# PM cache (manifest + trades + klines) and the recorded-book root both live in
# the main repo's data/ tree; the worktree shares them via these abs paths.
REPO_DATA = Path("/Users/shreypaharia/Documents/Projects/Trading/HLAnalysis/data")
PM_CACHE = REPO_DATA / "sim"
OUT_ROOT = WORKTREE / "data" / "sim" / "runs" / "pm-recorded-book-smoke-2026-05-31"

# Window covering the two book-covered BTC binary markets (settle 05-27, 05-28).
START, END = "2026-05-27", "2026-05-29"

# v1 PM prod slot (config/strategy.yaml `late_resolution` PM, priceBinary).
BASE = {
    "tte_min_seconds": 0,
    "tte_max_seconds": 86400,
    "price_extreme_threshold": 0.85,
    "price_extreme_max": 0.99,
    "distance_from_strike_usd_min": 0,
    "vol_max": 100,
    "stop_loss_pct": None,
    "max_position_usd": 300,
    "min_safety_d": 1.0,
    "vol_lookback_seconds": 3600,
    "exit_safety_d": 1.0,
    "vol_ewma_lambda": 0.85,
    "size_cap_near_strike_pct": 1.0,
    "size_cap_max_dist_pct": 1.5,
    "size_cap_min_ask": 0.88,
    "fee_model": "pm_binary",
    "fee_rate": 0.07,
    "vol_sampling_dt_seconds": 60,
}


def _parse_report(path: Path) -> dict:
    text = path.read_text()

    def find(pat: str, cast=float, default=0):
        m = re.search(pat, text)
        if not m:
            return default
        return cast(m.group(1).replace(",", ""))

    return {
        "n_markets": find(r"questions:\s+(\d+)", int),
        "n_trades": find(r"trades:\s+(\d+)", int),
        "total_pnl_usd": find(r"total PnL:\s+\$([-\d.,]+)"),
        "sharpe": find(r"Sharpe[^:]*:\s+([-\d.]+)"),
        "hit_rate": find(r"hit rate:\s+([\d.]+)%") / 100.0,
        "max_drawdown_usd": find(r"max drawdown:\s+\$([-\d.,]+)"),
    }


def _leg_map() -> dict[str, str]:
    """token_id → 'yes' | 'no' for the BTC binary markets in the cache."""
    from hlanalysis.backtest.data.polymarket import PolymarketDataSource

    ds = PolymarketDataSource(cache_root=PM_CACHE)
    out: dict[str, str] = {}
    for d in ds.discover(start=START, end=END, kind="binary"):
        out[d.leg_symbols[0]] = "yes"
        out[d.leg_symbols[1]] = "no"
    return out


def _avg_fill_vs_mid(out_dir: Path) -> dict:
    """Average taker fill price vs leg mid (signed adverse slippage, ¢).

    Joins each non-settle / non-hedge fill to the diagnostic row at its ts +
    question and reads the fill's leg mid. Positive = adverse (paid above mid on
    a buy / sold below mid). Returns {'n_taker_fills', 'avg_slippage_cents',
    'avg_fill_price', 'avg_mid'}.
    """
    fills_p = out_dir / "fills.parquet"
    diag_p = out_dir / "diagnostics.parquet"
    if not (fills_p.exists() and diag_p.exists()):
        return {"n_taker_fills": 0, "avg_slippage_cents": None,
                "avg_fill_price": None, "avg_mid": None}
    import pyarrow.parquet as pq

    leg = _leg_map()
    fills = pq.read_table(fills_p).to_pylist()
    diag = pq.read_table(diag_p).to_pylist()
    # Index diagnostics by (question_id, ts_ns).
    diag_idx = {(r["question_id"], r["ts_ns"]): r for r in diag}

    slips: list[float] = []
    fpx: list[float] = []
    mids: list[float] = []
    for f in fills:
        if f["cloid"] == "settle" or f.get("is_hedge"):
            continue
        side = f["side"]
        leg_side = leg.get(f["symbol"])
        d = diag_idx.get((f["question_id"], f["ts_ns"]))
        if d is None or leg_side is None:
            continue
        if leg_side == "yes":
            bid, ask = d["yes_bid"], d["yes_ask"]
        else:
            bid, ask = d["no_bid"], d["no_ask"]
        if bid is None or ask is None:
            continue
        mid = (float(bid) + float(ask)) / 2.0
        px = float(f["price"])
        # Adverse slippage: buy pays above mid, sell receives below mid.
        adverse = (px - mid) if side == "buy" else (mid - px)
        slips.append(adverse * 100.0)  # cents
        fpx.append(px)
        mids.append(mid)

    n = len(slips)
    return {
        "n_taker_fills": n,
        "avg_slippage_cents": (sum(slips) / n) if n else None,
        "avg_fill_price": (sum(fpx) / n) if n else None,
        "avg_mid": (sum(mids) / n) if n else None,
    }


def run_mode(book_source: str) -> dict:
    out_dir = OUT_ROOT / book_source
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = out_dir / "config.json"
    cfg_path.write_text(json.dumps(BASE, indent=2))

    cmd = [
        "uv", "run", "hl-bt", "run",
        "--strategy", "v1_late_resolution",
        "--data-source", "polymarket",
        "--config", str(cfg_path),
        "--out-dir", str(out_dir),
        "--start", START, "--end", END,
        "--kind", "binary",
        "--fee-model", "pm_binary", "--fee-rate", "0.07",
        "--pm-book-source", book_source,
    ]
    env = {**os.environ, "HLBT_PM_CACHE_ROOT": str(PM_CACHE)}
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=WORKTREE, capture_output=True, text=True, env=env)
    dt = time.time() - t0
    if proc.returncode != 0:
        print(f"[FAIL {book_source}] rc={proc.returncode}")
        print(proc.stderr[-3000:])
        raise SystemExit(1)
    r = _parse_report(out_dir / "report.md")
    r.update(_avg_fill_vs_mid(out_dir))
    print(
        f"[ok] {book_source:>9} ({dt:.1f}s) : PnL=${r.get('total_pnl_usd', 0):.2f} "
        f"trades={r.get('n_trades', 0)} hit={r.get('hit_rate', 0):.1%} "
        f"maxDD=${r.get('max_drawdown_usd', 0):.2f} "
        f"taker_fills={r.get('n_taker_fills', 0)} "
        f"slip={r.get('avg_slippage_cents')}"
    )
    return r


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = {mode: run_mode(mode) for mode in ("synthetic", "recorded")}

    print()
    hdr = (f"{'mode':>9} {'mkts':>5} {'trades':>7} {'PnL':>9} {'hit':>7} "
           f"{'maxDD':>9} {'takerN':>7} {'slip¢':>7} {'avgfill':>8} {'avgmid':>8}")
    print(hdr)
    for mode, r in results.items():
        slip = r.get("avg_slippage_cents")
        af = r.get("avg_fill_price")
        am = r.get("avg_mid")
        print(
            f"{mode:>9} {r.get('n_markets', 0):>5} {r.get('n_trades', 0):>7} "
            f"${r.get('total_pnl_usd', 0):>7.2f} {r.get('hit_rate', 0):>6.1%} "
            f"${r.get('max_drawdown_usd', 0):>7.2f} {r.get('n_taker_fills', 0):>7} "
            f"{(f'{slip:.3f}' if slip is not None else 'n/a'):>7} "
            f"{(f'{af:.4f}' if af is not None else 'n/a'):>8} "
            f"{(f'{am:.4f}' if am is not None else 'n/a'):>8}"
        )

    out_json = OUT_ROOT / "summary.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nfull results → {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
