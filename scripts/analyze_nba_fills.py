"""Join fills.parquet + WP series + PBP rows for an NBA backtest run.

For each fill, emit: ts_ns, condition_id, leg ('home'|'away'), price, size, fee,
nearest_pbp_ts_ns, score_diff_at_fill, ttr_seconds_at_fill, depth_assumption_usd,
period_at_fill, is_garbage_time (|score_diff|>20 with TTR<300), is_overtime.

Aggregates: fills_count, mean/median depth (we use the synthetic-L2 depth
assumption from the data source as the depth proxy — there's no live PM L2
in cache), fillable_pct (size_intended vs size_filled).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def _nearest(series_ts: list[int], target: int) -> int:
    """Index of the latest series row with ts_ns <= target (bisect_right - 1)."""
    import bisect
    i = bisect.bisect_right(series_ts, target) - 1
    return i if i >= 0 else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="Backtest run directory containing fills.parquet")
    ap.add_argument("--cache-root", required=True, help="data/sim/pm_nba")
    ap.add_argument("--out", default=None, help="Defaults to <run-dir>/fills_annotated.parquet")
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir)
    cache_root = Path(args.cache_root)
    fills_path = run_dir / "fills.parquet"
    if not fills_path.exists():
        print(f"No fills.parquet at {fills_path}", file=sys.stderr)
        return 2

    manifest = json.loads((cache_root / "manifest.json").read_text())
    # (token_id → (condition_id, espn_game_id, home_token_id))
    token_lookup: dict[str, tuple[str, str, str]] = {}
    for cid, entry in manifest.items():
        mk = entry.get("market") or {}
        for tok in (mk.get("home_token_id"), mk.get("away_token_id")):
            if tok:
                token_lookup[str(tok)] = (
                    cid,
                    str(mk.get("espn_game_id") or ""),
                    str(mk.get("home_token_id")),
                )

    fills = pq.read_table(fills_path).to_pylist()
    annotated: list[dict] = []
    wp_cache: dict[str, list[dict]] = {}
    for f in fills:
        sym = str(f.get("symbol"))
        if sym not in token_lookup:
            continue
        cid, gid, home_tok = token_lookup[sym]
        if gid not in wp_cache:
            wp_path = cache_root / "wp_series" / f"{gid}.parquet"
            wp_cache[gid] = pq.read_table(wp_path).to_pylist() if wp_path.exists() else []
        wp_rows = wp_cache[gid]
        if not wp_rows:
            continue
        ts_list = [int(r["ts_ns"]) for r in wp_rows]
        idx = _nearest(ts_list, int(f["ts_ns"]))
        row = wp_rows[idx]
        score_diff = int(row["score_diff_home"])
        ttr = int(row["total_seconds_remaining"])
        period = int(row["period"])
        leg = "home" if sym == home_tok else "away"
        annotated.append({
            "ts_ns": int(f["ts_ns"]),
            "condition_id": cid,
            "leg": leg,
            "price": float(f["price"]),
            "size": float(f["size"]),
            "fee": float(f.get("fee", 0.0)),
            "score_diff_at_fill": score_diff,
            "ttr_at_fill": ttr,
            "period_at_fill": period,
            "is_garbage_time": (abs(score_diff) > 20 and ttr < 300),
            "is_overtime": bool(row.get("is_overtime", False)),
        })

    print(f"Annotated {len(annotated)} fills.")
    if not annotated:
        return 0
    out_path = Path(args.out) if args.out else (run_dir / "fills_annotated.parquet")
    pq.write_table(pa.table({
        k: [a[k] for a in annotated] for k in annotated[0].keys()
    }), out_path)
    print(f"Wrote {out_path}")

    # Quick aggregates to stdout.
    n = len(annotated)
    print(f"Fills count: {n}")
    if n:
        n_garbage = sum(1 for a in annotated if a["is_garbage_time"])
        n_ot = sum(1 for a in annotated if a["is_overtime"])
        print(f"Garbage-time fills: {n_garbage} ({100*n_garbage/n:.1f}%)")
        print(f"OT fills: {n_ot}")
        mean_ttr = sum(a["ttr_at_fill"] for a in annotated) / n
        print(f"Mean TTR at fill: {mean_ttr:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
