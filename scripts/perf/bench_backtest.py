"""Benchmark harness for the hl_hip4 vs polymarket data paths.

Runs `hl-bt run` end-to-end on a small slice, captures pyinstrument profile +
duckdb.connect() count + wall time. Writes results to a JSON file that the
report builder consumes.

Usage:
    HLBT_HL_DATA_ROOT=/path/to/data \
    HLBT_PM_CACHE_ROOT=/path/to/data/sim \
    python scripts/perf/bench_backtest.py --label baseline --out summeries/perf-bench-baseline.json
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import duckdb
from pyinstrument import Profiler


@contextlib.contextmanager
def count_duckdb_connects():
    counter = {"n": 0}
    real = duckdb.connect

    def wrapped(*a, **kw):
        counter["n"] += 1
        return real(*a, **kw)

    duckdb.connect = wrapped  # type: ignore[assignment]
    try:
        yield counter
    finally:
        duckdb.connect = real  # type: ignore[assignment]


def run_one(
    *,
    label: str,
    data_source: str,
    kind: str,
    start: str,
    end: str,
    config: str,
    max_markets: int,
    extra_args: list[str],
) -> dict:
    from hlanalysis.backtest.cli import main as cli_main

    safe_label = label.replace("/", "_")
    out_dir = tempfile.mkdtemp(prefix=f"bench-{safe_label}-")
    argv = [
        "run",
        "--strategy", "v1_late_resolution",
        "--data-source", data_source,
        "--config", config,
        "--out-dir", out_dir,
        "--start", start,
        "--end", end,
        "--kind", kind,
        "--max-markets", str(max_markets),
    ] + extra_args

    prof = Profiler(async_mode="disabled")
    t0 = time.perf_counter()
    with count_duckdb_connects() as cc:
        prof.start()
        rc = cli_main(argv)
        prof.stop()
    wall = time.perf_counter() - t0

    top_frames = prof.output_text(unicode=True, color=False, show_all=False)

    # Best-effort cleanup; keep on hard failure for forensic inspection.
    if rc == 0:
        shutil.rmtree(out_dir, ignore_errors=True)

    return {
        "label": label,
        "data_source": data_source,
        "kind": kind,
        "start": start,
        "end": end,
        "max_markets": max_markets,
        "wall_s": wall,
        "duckdb_connects": cc["n"],
        "rc": rc,
        "profile_text": top_frames,
        "out_dir": out_dir if rc != 0 else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="Label for this run (e.g. 'baseline', 'after-fix-1')")
    ap.add_argument("--out", required=True, help="JSON output path")
    ap.add_argument("--hl-config", default="/Users/shreypaharia/Documents/Projects/Trading/HLAnalysis/data/sim/configs/v1-finalize-5.json")
    ap.add_argument("--pm-config", default="/Users/shreypaharia/Documents/Projects/Trading/HLAnalysis/data/sim/configs/v1-finalize-5.json")
    ap.add_argument("--max-markets", type=int, default=1)
    ap.add_argument("--scenarios", default="hl_binary,hl_bucket,pm_binary",
                    help="Comma-separated list of scenarios to run")
    args = ap.parse_args()

    scenarios = {
        "hl_binary": dict(
            data_source="hl_hip4", kind="binary",
            start="2026-05-13", end="2026-05-15", config=args.hl_config,
            extra_args=[],
        ),
        "hl_bucket": dict(
            data_source="hl_hip4", kind="bucket",
            start="2026-05-13", end="2026-05-15", config=args.hl_config,
            extra_args=[],
        ),
        # Use a 2-day PM window matching the HL slice as closely as possible.
        # PM 'binary' = BTC Up/Down daily.
        "pm_binary": dict(
            data_source="polymarket", kind="binary",
            start="2026-05-13", end="2026-05-15", config=args.pm_config,
            extra_args=[],
        ),
        "hl_binary_3q": dict(
            data_source="hl_hip4", kind="binary",
            start="2026-05-13", end="2026-05-17", config=args.hl_config,
            extra_args=[],
        ),
        "hl_bucket_3q": dict(
            data_source="hl_hip4", kind="bucket",
            start="2026-05-13", end="2026-05-17", config=args.hl_config,
            extra_args=[],
        ),
    }

    results = []
    want = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    for name in want:
        spec = scenarios[name]
        max_m = 3 if name.endswith("_3q") else args.max_markets
        print(f"==> Running scenario {name} (max_markets={max_m})", flush=True)
        try:
            r = run_one(
                label=f"{args.label}/{name}",
                data_source=spec["data_source"],
                kind=spec["kind"],
                start=spec["start"],
                end=spec["end"],
                config=spec["config"],
                max_markets=max_m,
                extra_args=spec["extra_args"],
            )
        except Exception as e:
            print(f"   FAILED: {e}", flush=True)
            r = {"label": f"{args.label}/{name}", "error": str(e)}
        results.append({"scenario": name, **r})
        print(f"   wall_s={r.get('wall_s'):.2f}s duckdb_connects={r.get('duckdb_connects')} rc={r.get('rc')}" if "wall_s" in r else f"   ERROR: {r.get('error')}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
