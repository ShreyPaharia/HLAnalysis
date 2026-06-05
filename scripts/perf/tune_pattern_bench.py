"""Tune-access-pattern micro-benchmark for the param-independent caches.

A tuning sweep reconstructs the data source once per param cell and replays the
SAME questions through it. Two costs are param-independent yet recomputed every
cell unless cached across the sweep:

  - settlement  : ``resolved_outcome(q)``  (duckdb settlement scan + BTC ref)
  - data-prep   : ``events_arrays(q)``     (source-file glob + npz inflate)

This script simulates N cells (a fresh source each) replaying one question and
reports, with the process memo OFF vs ON (``HLBT_INPROC_BUNDLE_MEMO``):

  - settlement_impl_calls : how many cells actually computed the outcome
  - events_arrays per-cell wall (cell 1 = cold-ish, cells 2..N = replay)

Expected with the memo ON: settlement computed once (not N times) and the
replay cells' data-prep collapses toward zero.

Usage::

    HLBT_HL_DATA_ROOT=/path/to/data \
    uv run python scripts/perf/tune_pattern_bench.py --cells 5
"""
from __future__ import annotations

import argparse
import os
import time

from hlanalysis.backtest.data.hl_hip4 import HLHip4DataSource
import hlanalysis.backtest.data.hl_hip4 as hlmod
import hlanalysis.backtest.data._event_array_cache as cache


def _discover_one(root: str, start: str, end: str):
    ds = HLHip4DataSource(data_root=root)
    qs = ds.discover(start=start, end=end, underlying="BTC")
    if not qs:
        raise SystemExit(f"no HL BTC question in [{start}, {end})")
    return qs[0]


def run(memo: str, *, root: str, q, cells: int) -> None:
    os.environ["HLBT_INPROC_BUNDLE_MEMO"] = memo
    cache._inproc_clear()
    hlmod._proc_outcome_clear()

    impl_calls = {"n": 0}
    real = HLHip4DataSource._resolve_outcome_impl

    def counting(self, qq):
        impl_calls["n"] += 1
        return real(self, qq)

    HLHip4DataSource._resolve_outcome_impl = counting  # type: ignore[assignment]
    try:
        ev_ms: list[float] = []
        for _ in range(cells):
            ds = HLHip4DataSource(data_root=root)  # fresh source = next param cell
            t0 = time.perf_counter()
            ds.events_arrays(q)
            ev_ms.append((time.perf_counter() - t0) * 1000)
            ds.resolved_outcome(q)
    finally:
        HLHip4DataSource._resolve_outcome_impl = real  # type: ignore[assignment]

    label = "ON " if memo == "1" else "OFF"
    cell1 = ev_ms[0]
    rest = ev_ms[1:]
    rest_mean = sum(rest) / len(rest) if rest else 0.0
    print(
        f"memo {label}  settlement_impl_calls={impl_calls['n']:>2}/{cells}  "
        f"events_arrays: cell1={cell1:7.1f}ms  replay_mean={rest_mean:7.1f}ms  "
        f"(per-cell: {', '.join(f'{x:.0f}' for x in ev_ms)})"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-05-13")
    ap.add_argument("--end", default="2026-05-14")
    ap.add_argument("--cells", type=int, default=5)
    args = ap.parse_args()
    root = os.environ["HLBT_HL_DATA_ROOT"]
    q = _discover_one(root, args.start, args.end)
    print(f"Simulating {args.cells} tune cells replaying question {q.question_id}\n")
    run("0", root=root, q=q, cells=args.cells)
    run("1", root=root, q=q, cells=args.cells)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
