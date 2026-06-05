#!/usr/bin/env python3
"""BTC reference equivalence: recorded Binance spot BBO vs pulled 1s klines, dt=5.

Question: if we stop RECORDING Binance spot and instead pull 1s klines on demand,
does the PM BTC backtest change materially? Both feeds are bucketed to dt=5
(vol_sampling_dt_seconds=5) so the only difference is the source of the σ-feeding
ReferenceEvents:
  (A) recorded  = recorded Binance SPOT BBO ticks  → 5s OHLC of quote-MID extremes
  (B) klines_1s = pulled Binance SPOT 1s klines    → 5s OHLC of trade-OHLC extremes
Strike resolution is identical in both (PM settles on the Binance 1m spot close).

Run on the LIVE-relevant corpus: recorded PM L2 book (book_source=recorded,
coverage from 2026-05-27) over the 9 BTC Up/Down binaries that resolve
2026-05-27 .. 2026-06-04. Small n → suggestive, not load-bearing.

Three layers of evidence (per the task — a matching PnL with a diverging σ is
luck, not equivalence):
  1. σ-series divergence  : per-market rolling Parkinson + bipower σ at the
                            strategy lookback, aligned by 5s bucket.
  2. gate/entry divergence: per-market entries (do they fire on the same ticks?).
  3. metric divergence    : PnL / hit-rate / trade-count deltas.

Both live PM strategies are run because they stress different parts of the feed:
  - v1_late_resolution → Parkinson σ (H/L based → sensitive to the H/L source)
  - v3_theta_harvester → bipower σ   (close-to-close → sensitive to closes)
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from hlanalysis.backtest.data.polymarket import PolymarketDataSource
from hlanalysis.strategy._numba.vol import (
    bipower_variation_sigma,
    parkinson_sigma_window,
)

WORKTREE = Path(__file__).resolve().parents[1]
PM_CACHE = Path("/Users/shreypaharia/Documents/Projects/Trading/HLAnalysis/data/sim")
OUT_ROOT = WORKTREE / "data" / "sim" / "runs" / "btc-ref-equivalence-2026-06-06"
START, END = "2026-05-27", "2026-06-05"
DT = 5
RES_NS = DT * 1_000_000_000
LOOKBACK_S = 3600
WIN_BARS = LOOKBACK_S // DT  # 720 trailing bars

# Live PM slot configs (config/strategy.yaml), dt=5.
V1_CFG = {
    "tte_min_seconds": 0, "tte_max_seconds": 7200,
    "price_extreme_threshold": 0.85, "price_extreme_max": 0.99,
    "min_safety_d": 3.0, "vol_lookback_seconds": 3600, "exit_safety_d": 1.0,
    "vol_ewma_lambda": 0.97, "vol_estimator": "parkinson",
    "vol_sampling_dt_seconds": 5, "distance_from_strike_usd_min": 0, "vol_max": 100,
    "size_cap_near_strike_pct": 1.0, "size_cap_max_dist_pct": 1.5,
    "size_cap_min_ask": 0.88, "use_bid_for_entry_gate": False,
    "min_bid_notional_usd": 10.0, "max_position_usd": 50, "stop_loss_pct": None,
    "fee_model": "pm_binary", "fee_rate": 0.07,
}
THETA_CFG = {
    "vol_lookback_seconds": 3600, "vol_sampling_dt_seconds": 5,
    "vol_clip_min": 0.0, "vol_clip_max": 5.0, "vol_estimator": "bipower",
    "edge_buffer": 0.02, "fee_taker": 0.0, "half_spread_assumption": 0.005,
    "drift_lookback_seconds": 3600, "drift_blend": 0.0, "favorite_threshold": 0.85,
    "edge_max": None, "exit_edge_threshold": 0.0, "time_stop_seconds": 0,
    "exit_take_profit_mode": True, "exit_fee": 0.0007, "min_distance_pct": None,
    "min_bid_notional_usd": 10.0, "topup_enabled": False, "exit_safety_d": 1.0,
    "max_position_usd": 50, "stop_loss_pct": None, "tte_min_seconds": 0,
    "tte_max_seconds": 86400, "price_extreme_threshold": 0.0, "price_extreme_max": 1.0,
    "fee_model": "pm_binary", "fee_rate": 0.07,
}
STRATS = {
    "v1": ("v1_late_resolution", V1_CFG),
    "theta": ("v3_theta_harvester", THETA_CFG),
}
# (A) recorded vs (B) pulled 1s klines.
REFS = {
    "recorded": ["--pm-reference-source", "binance_bbo",
                 "--pm-binance-bbo-product-type", "spot"],
    "klines_1s": ["--pm-reference-source", "klines_1s"],
}


# ----------------------------------------------------------------------------
# Layer 1: σ-series divergence (computed directly, no backtester)
# ----------------------------------------------------------------------------

def _mk(src: str) -> PolymarketDataSource:
    return PolymarketDataSource(
        cache_root=PM_CACHE, reference_source=src, reference_resample_seconds=DT,
        binance_bbo_product_type="spot", book_source="recorded",
    )


def _ref_series(ds: PolymarketDataSource, d) -> list:
    if ds._reference_source == "binance_bbo":
        return ds._load_binance_bbo_reference(d.start_ts_ns, d.end_ts_ns)
    return ds._load_klines_1s_reference(d.start_ts_ns, d.end_ts_ns)


def _rolling_sigma(events: list, lam: float) -> dict:
    """Rolling Parkinson + bipower σ over a ref series, keyed by 5s bucket idx."""
    buckets = np.array([e.ts_ns // RES_NS for e in events])
    highs = np.array([e.high for e in events])
    lows = np.array([e.low for e in events])
    closes = np.array([e.close for e in events])
    park: dict[int, float] = {}
    bipo: dict[int, float] = {}
    for i in range(len(events)):
        lo = max(0, i - WIN_BARS + 1)
        h, l, c = highs[lo:i + 1], lows[lo:i + 1], closes[lo:i + 1]
        if len(c) >= 2:
            park[int(buckets[i])] = parkinson_sigma_window(h, l, lam)
            rets = np.diff(np.log(c))
            bipo[int(buckets[i])] = bipower_variation_sigma(rets)
    return {"park": park, "bipo": bipo}


def _rel_delta_stats(a: dict, b: dict) -> dict:
    common = sorted(set(a) & set(b))
    if not common:
        return {"n": 0}
    av = np.array([a[k] for k in common])
    bv = np.array([b[k] for k in common])
    denom = np.where(av > 1e-12, av, np.nan)
    rel = np.abs(av - bv) / denom
    rel = rel[~np.isnan(rel)]
    return {
        "n": len(common),
        "mean_rel": float(np.nanmean(rel)),
        "median_rel": float(np.nanmedian(rel)),
        "p95_rel": float(np.nanpercentile(rel, 95)),
        # max_rel is dominated by near-zero-σ bars (a degenerate H==L bucket in
        # one source vs a tiny non-zero σ in the other inflates the ratio); the
        # frac_within_* fields are the honest "do they agree" signal.
        "max_rel": float(np.nanmax(rel)),
        "frac_within_1pct": float(np.mean(rel < 0.01)),
        "frac_within_5pct": float(np.mean(rel < 0.05)),
        "mean_a": float(av.mean()), "mean_b": float(bv.mean()),
    }


def sigma_divergence() -> dict:
    rec, k1s = _mk("binance_bbo"), _mk("klines_1s")
    descs = rec.discover(start=START, end=END, kind="binary")
    per_market = []
    agg = {"park": {"a": {}, "b": {}}, "bipo": {"a": {}, "b": {}}}
    for d in descs:
        ra, rb = _ref_series(rec, d), _ref_series(k1s, d)
        if len(ra) < WIN_BARS or len(rb) < WIN_BARS:
            continue
        sa = _rolling_sigma(ra, V1_CFG["vol_ewma_lambda"])
        sb = _rolling_sigma(rb, V1_CFG["vol_ewma_lambda"])
        park = _rel_delta_stats(sa["park"], sb["park"])
        bipo = _rel_delta_stats(sa["bipo"], sb["bipo"])
        per_market.append({
            "qid": d.question_id[:12], "n_buckets_rec": len(ra),
            "n_buckets_k1s": len(rb), "parkinson": park, "bipower": bipo,
        })
        # offset bucket keys per-market so they don't collide in the aggregate.
        # a = recorded (sa), b = klines_1s (sb).
        off = d.start_ts_ns // RES_NS
        for est in ("park", "bipo"):
            for k, v in sa[est].items():
                agg[est]["a"][(off, k)] = v
            for k, v in sb[est].items():
                agg[est]["b"][(off, k)] = v
    aggregate = {
        "parkinson": _rel_delta_stats(agg["park"]["a"], agg["park"]["b"]),
        "bipower": _rel_delta_stats(agg["bipo"]["a"], agg["bipo"]["b"]),
    }
    return {"per_market": per_market, "aggregate": aggregate}


# ----------------------------------------------------------------------------
# Layers 2 & 3: backtest entries + metrics
# ----------------------------------------------------------------------------

def _parse_report(path: Path) -> dict:
    text = path.read_text()

    def find(pat, cast=float, default=0):
        m = re.search(pat, text)
        return cast(m.group(1).replace(",", "")) if m else default

    metrics = {
        "n_markets": find(r"questions:\s+(\d+)", int),
        "n_trades": find(r"trades:\s+(\d+)", int),
        "total_pnl_usd": find(r"total PnL:\s+\$([-\d.,]+)"),
        "sharpe": find(r"Sharpe[^:]*:\s+([-\d.]+)"),
        "hit_rate": find(r"hit rate:\s+([\d.]+)%") / 100.0,
        "max_drawdown_usd": find(r"max drawdown:\s+\$([-\d.,]+)"),
    }
    # Per-question entries: qid → (n_trades, pnl). Table rows look like
    # | 0xabc... | no | 1 | ts | ts | 8.30 |
    entries: dict[str, tuple[int, float]] = {}
    for m in re.finditer(r"\|\s*(0x[0-9a-f]+)\s*\|[^|]*\|\s*(\d+)\s*\|[^|]*\|[^|]*\|\s*([-\d.]+)\s*\|", text):
        entries[m.group(1)[:12]] = (int(m.group(2)), float(m.group(3)))
    return {"metrics": metrics, "entries": entries}


def run_cell(strat_key: str, ref_key: str) -> dict:
    strategy_id, cfg = STRATS[strat_key]
    out_dir = OUT_ROOT / f"{strat_key}_{ref_key}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = out_dir / "config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2))
    report = out_dir / "report.md"
    if report.exists() and report.stat().st_size > 100:
        print(f"[skip] {strat_key}/{ref_key} (cached)")
        return _parse_report(report)
    cmd = [
        "uv", "run", "hl-bt", "run",
        "--strategy", strategy_id, "--data-source", "polymarket",
        "--config", str(cfg_path), "--out-dir", str(out_dir),
        "--start", START, "--end", END, "--kind", "binary",
        "--fee-model", "pm_binary", "--fee-rate", "0.07",
        "--pm-book-source", "recorded",
        *REFS[ref_key],
    ]
    env = {**os.environ, "HLBT_PM_CACHE_ROOT": str(PM_CACHE)}
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=WORKTREE, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        print(f"[FAIL {strat_key}/{ref_key}] rc={proc.returncode}\n{proc.stderr[-3000:]}")
        raise SystemExit(1)
    r = _parse_report(report)
    m = r["metrics"]
    print(f"[ok] {strat_key}/{ref_key} ({time.time()-t0:.0f}s): "
          f"PnL=${m['total_pnl_usd']:.2f} trades={m['n_trades']} hit={m['hit_rate']:.0%}")
    return r


def entry_divergence(rec: dict, k1s: dict) -> dict:
    """Compare per-market entry decisions between the two reference sources."""
    qids = sorted(set(rec["entries"]) | set(k1s["entries"]))
    diffs = []
    same = 0
    for q in qids:
        na = rec["entries"].get(q, (0, 0.0))[0]
        nb = k1s["entries"].get(q, (0, 0.0))[0]
        if na == nb:
            same += 1
        else:
            diffs.append({"qid": q, "n_rec": na, "n_k1s": nb})
    return {"markets": len(qids), "same_entry_count": same, "diffs": diffs}


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("=== Layer 1: σ-series divergence (recorded vs klines_1s, dt=5) ===")
    sig = sigma_divergence()
    for pm in sig["per_market"]:
        p, b = pm["parkinson"], pm["bipower"]
        print(f"  {pm['qid']}  buckets rec/k1s={pm['n_buckets_rec']}/{pm['n_buckets_k1s']}  "
              f"Parkinson Δσ median={p.get('median_rel',0):.3%} p95={p.get('p95_rel',0):.3%}  "
              f"bipower Δσ median={b.get('median_rel',0):.3%} p95={b.get('p95_rel',0):.3%}")
    ap, ab = sig["aggregate"]["parkinson"], sig["aggregate"]["bipower"]
    print(f"  AGGREGATE Parkinson: n={ap['n']} median={ap['median_rel']:.3%} "
          f"p95={ap['p95_rel']:.3%} within1%={ap['frac_within_1pct']:.1%} "
          f"within5%={ap['frac_within_5pct']:.1%} (mean σ rec={ap['mean_a']:.5f} k1s={ap['mean_b']:.5f})")
    print(f"  AGGREGATE bipower:   n={ab['n']} median={ab['median_rel']:.3%} "
          f"p95={ab['p95_rel']:.3%} within1%={ab['frac_within_1pct']:.1%} "
          f"within5%={ab['frac_within_5pct']:.1%} (mean σ rec={ab['mean_a']:.5f} k1s={ab['mean_b']:.5f})")

    print("\n=== Layers 2&3: backtest (recorded PM book, dt=5) ===")
    results = {}
    for sk in STRATS:
        for rk in REFS:
            results[(sk, rk)] = run_cell(sk, rk)

    print(f"\n{'strat':>6} {'ref':>10} {'PnL':>9} {'trades':>7} {'hit':>6} {'maxDD':>8}")
    for (sk, rk), r in results.items():
        m = r["metrics"]
        print(f"{sk:>6} {rk:>10} ${m['total_pnl_usd']:>7.2f} {m['n_trades']:>7} "
              f"{m['hit_rate']:>5.0%} ${m['max_drawdown_usd']:>6.2f}")

    print("\n=== Entry-decision divergence (recorded vs klines_1s) ===")
    entry_div = {}
    for sk in STRATS:
        ed = entry_divergence(results[(sk, "recorded")], results[(sk, "klines_1s")])
        entry_div[sk] = ed
        print(f"  {sk}: {ed['same_entry_count']}/{ed['markets']} markets identical "
              f"entry-count; diffs={ed['diffs']}")

    summary = {
        "window": [START, END], "dt": DT, "n_markets": len(sig["per_market"]),
        "sigma_divergence": sig, "entry_divergence": entry_div,
        "backtest": {f"{sk}_{rk}": results[(sk, rk)]["metrics"]
                     for sk in STRATS for rk in REFS},
    }
    (OUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nfull results → {OUT_ROOT/'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
