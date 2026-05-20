"""Per-phase wall-time breakdown for the HL HIP-4 backtest.

Patches the data source's ``events_arrays`` and the runner's loop with cheap
``time.perf_counter`` deltas around each major phase, then runs the full
corpus (or a subset) and prints aggregated timings:

  - Phase 1: data source read + arrow → numpy column extraction
  - Phase 2: hftbacktest asset build (np.concatenate + BacktestAsset)
  - Phase 3: scan loop (strategy.evaluate + order submission + fill capture)
  - Phase 4: settlement + persistence

No env-var-driven runtime instrumentation in the production runner —
everything here is monkey-patched at script load time.

Usage::

    HLBT_HL_DATA_ROOT=/path/to/data python scripts/perf/phase_bench.py \
        --start 2026-05-06 --end 2026-05-21
"""
from __future__ import annotations

import argparse
import os
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path


_PHASE_TOTALS: dict[str, float] = defaultdict(float)
_PHASE_COUNTS: dict[str, int] = defaultdict(int)


@contextmanager
def _phase(name: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        _PHASE_TOTALS[name] += time.perf_counter() - t0
        _PHASE_COUNTS[name] += 1


def _patch_runner():
    import hlanalysis.backtest.runner.hftbt_runner as runner_mod

    orig_run = runner_mod.run_one_question

    def wrapped_run(strategy, data_source, q, cfg, **kwargs):
        # 1) data_source.events_arrays (or events) — Phase 1.
        fast = getattr(data_source, "events_arrays", None)
        if fast is not None:
            with _phase("1_data_fetch_build"):
                bundle = fast(q)
            # Re-attach as a stub that the runner will call once.
            class _Stub:
                def events_arrays(self_, qq):
                    return bundle
                def __getattr__(self_, name):
                    return getattr(data_source, name)
            ds = _Stub()
        else:
            ds = data_source

        # 2) The rest of run_one_question — Phase 2-4 lumped under "rest".
        with _phase("2_run_one_question_rest"):
            return orig_run(strategy, ds, q, cfg, **kwargs)

    runner_mod.run_one_question = wrapped_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-05-06")
    ap.add_argument("--end", default="2026-05-21")
    ap.add_argument("--max-markets", type=int, default=None)
    ap.add_argument("--kind", default="both", choices=["binary", "bucket", "both"])
    ap.add_argument("--strategy", default="v1_late_resolution")
    ap.add_argument(
        "--config",
        default="/Users/shreypaharia/Documents/Projects/Trading/HLAnalysis/data/sim/configs/v1-finalize-5.json",
    )
    args = ap.parse_args()

    _patch_runner()

    from hlanalysis.backtest.cli import main as cli_main
    argv = [
        "run",
        "--strategy", args.strategy,
        "--data-source", "hl_hip4",
        "--config", args.config,
        "--out-dir", "/tmp/phase-bench-out",
        "--start", args.start,
        "--end", args.end,
        "--kind", args.kind,
    ]
    if args.max_markets is not None:
        argv += ["--max-markets", str(args.max_markets)]

    import shutil
    shutil.rmtree("/tmp/phase-bench-out", ignore_errors=True)

    t0 = time.perf_counter()
    rc = cli_main(argv)
    total = time.perf_counter() - t0
    print(f"\n=== Phase breakdown ===")
    print(f"Total wall: {total:.2f}s, rc={rc}")
    for name in sorted(_PHASE_TOTALS):
        cnt = _PHASE_COUNTS[name]
        tot = _PHASE_TOTALS[name]
        print(f"  {name}: {tot:.2f}s across {cnt} calls ({100*tot/total:.1f}% of wall)")


if __name__ == "__main__":
    import sys
    sys.exit(main())
