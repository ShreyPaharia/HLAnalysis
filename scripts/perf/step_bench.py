"""Step-by-step wall-time breakdown of the HL HIP-4 backtest.

Monkey-patches the boundary callables inside ``run_one_question`` and accumulates
per-step wall time, call count, and *first-call* time (to separate one-time
numba JIT compilation from steady-state per-tick cost). The native scan-loop
cost (``hbt.elapse`` + order submit + fill capture — a numba jitclass we cannot
patch) is derived as the remainder.

Steps timed:
  - data_prep            : data_source.events_arrays(q)  (cache load / build)
      · cache_key         : sha key derivation (PosixPath.stat + str)
      · cache_load        : npz inflate + undelta
  - asset_build          : _build_asset + HashMapMarketDepthBacktest ctor
  - strategy_eval        : sum of strategy.evaluate(...) over all ticks
  - settlement_outcome   : data_source.resolved_outcome(q)   (count proves redundancy)
  - settlement_payoff    : data_source.leg_payoff(q, sym)
  - loop_native+rest     : run_one_question total - sum(above inside it)

Usage::

    HLBT_HL_DATA_ROOT=/path/to/data \
    uv run python scripts/perf/step_bench.py --kind binary \
        --start 2026-05-13 --end 2026-05-18
"""
from __future__ import annotations

import argparse
import os
import shutil
import time
from collections import defaultdict


_T: dict[str, float] = defaultdict(float)
_N: dict[str, int] = defaultdict(int)
_FIRST: dict[str, float] = {}


def _acc(name: str, dt: float) -> None:
    _T[name] += dt
    _N[name] += 1
    if name not in _FIRST:
        _FIRST[name] = dt


def _wrap(obj, attr, name):
    """Wrap a bound/instance attribute or module-level callable with a timer."""
    orig = getattr(obj, attr)

    def wrapped(*a, **kw):
        t0 = time.perf_counter()
        try:
            return orig(*a, **kw)
        finally:
            _acc(name, time.perf_counter() - t0)

    setattr(obj, attr, wrapped)
    return orig


def _patch():
    import hlanalysis.backtest.runner.hftbt_runner as runner_mod
    from hlanalysis.backtest.data import hl_hip4 as hl_mod
    from hlanalysis.backtest.data import _event_array_cache as cache_mod

    # data_prep + sub-splits
    _wrap(hl_mod.HLHip4DataSource, "events_arrays", "data_prep")
    _wrap(cache_mod, "cache_key", "  · cache_key")
    _wrap(cache_mod, "_load", "  · cache_load")

    # settlement (count proves the 3x resolved_outcome redundancy)
    _wrap(hl_mod.HLHip4DataSource, "resolved_outcome", "settlement_outcome")
    _wrap(hl_mod.HLHip4DataSource, "leg_payoff", "settlement_payoff")

    # asset build: _build_asset (module fn) + the jitclass ctor
    _wrap(runner_mod, "_build_asset", "asset_build(_build_asset)")
    _wrap(runner_mod.hb, "HashMapMarketDepthBacktest", "asset_build(hbt_ctor)")

    # strategy.evaluate — patch on the class once we know which strategy.
    # Done lazily inside run_one_question wrapper (strategy instance is the arg).
    orig_rq = runner_mod.run_one_question
    _patched_eval = {"done": False}

    def wrapped_rq(strategy, data_source, q, cfg, **kwargs):
        if not _patched_eval["done"]:
            cls = type(strategy)
            if hasattr(cls, "evaluate"):
                _wrap(cls, "evaluate", "strategy_eval")
            _patched_eval["done"] = True
        t0 = time.perf_counter()
        try:
            return orig_rq(strategy, data_source, q, cfg, **kwargs)
        finally:
            _acc("run_one_question_TOTAL", time.perf_counter() - t0)

    runner_mod.run_one_question = wrapped_rq


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-05-13")
    ap.add_argument("--end", default="2026-05-18")
    ap.add_argument("--kind", default="binary", choices=["binary", "bucket", "both"])
    ap.add_argument("--strategy", default="v1_late_resolution")
    ap.add_argument("--max-markets", type=int, default=None)
    ap.add_argument(
        "--config",
        default="/Users/shreypaharia/Documents/Projects/Trading/HLAnalysis/data/sim/configs/v1-finalize-5.json",
    )
    args = ap.parse_args()

    _patch()

    from hlanalysis.backtest.cli import main as cli_main

    argv = [
        "run", "--strategy", args.strategy,
        "--data-source", "hl_hip4", "--config", args.config,
        "--out-dir", "/tmp/step-bench-out",
        "--start", args.start, "--end", args.end, "--kind", args.kind,
    ]
    if args.max_markets is not None:
        argv += ["--max-markets", str(args.max_markets)]
    shutil.rmtree("/tmp/step-bench-out", ignore_errors=True)

    t0 = time.perf_counter()
    rc = cli_main(argv)
    total = time.perf_counter() - t0

    rq_total = _T.get("run_one_question_TOTAL", 0.0)
    # Steps that live *inside* run_one_question (exclude resolved_outcome calls
    # made outside it, e.g. parallel.py QResult — those are counted globally but
    # we attribute the within-rq portion to settlement via leg_payoff+1 direct).
    inside = (
        _T["data_prep"]
        + _T["asset_build(_build_asset)"]
        + _T["asset_build(hbt_ctor)"]
        + _T["strategy_eval"]
        + _T["settlement_outcome"]
        + _T["settlement_payoff"]
    )
    loop_native_rest = max(0.0, rq_total - inside)

    order = [
        "data_prep",
        "  · cache_key",
        "  · cache_load",
        "asset_build(_build_asset)",
        "asset_build(hbt_ctor)",
        "strategy_eval",
        "settlement_outcome",
        "settlement_payoff",
    ]
    print("\n=== Step-by-step breakdown ===")
    print(f"process wall:            {total:8.2f}s  rc={rc}")
    print(f"run_one_question TOTAL:  {rq_total:8.2f}s  ({100*rq_total/total:4.1f}% of wall, {_N['run_one_question_TOTAL']} questions)")
    print(f"{'step':<28}{'total_s':>9}{'calls':>8}{'mean_ms':>10}{'first_ms':>10}{'%rq':>7}")
    for k in order:
        if _N[k] == 0:
            continue
        tot = _T[k]; n = _N[k]
        mean_ms = 1000 * tot / n
        first_ms = 1000 * _FIRST.get(k, 0.0)
        pct = 100 * tot / rq_total if rq_total else 0.0
        print(f"{k:<28}{tot:>9.3f}{n:>8}{mean_ms:>10.2f}{first_ms:>10.1f}{pct:>7.1f}")
    pct_loop = 100 * loop_native_rest / rq_total if rq_total else 0.0
    print(f"{'loop_native + rest (derived)':<28}{loop_native_rest:>9.3f}{'-':>8}{'-':>10}{'-':>10}{pct_loop:>7.1f}")
    return rc


if __name__ == "__main__":
    import sys
    sys.exit(main())
