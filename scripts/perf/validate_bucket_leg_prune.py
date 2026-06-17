#!/usr/bin/env python3
"""Validate that bucket leg-pruning is decision-identical + measure the speed-up.

For each sampled HL priceBucket question we run ``run_one_question`` twice with
an otherwise-identical config — once with leg pruning OFF and once ON (favorite
threshold from the v31 priceBucket slot) — and assert the per-tick diagnostics
and the fills are byte-for-byte identical. We also report the average legs
loaded (non-empty event arrays) and the wall time per question, off vs on.

Run from the worktree:
    HLBT_HL_DATA_ROOT=/abs/path/to/main/data uv run python \
        scripts/perf/validate_bucket_leg_prune.py [--skip 0 --n 6]
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from hlanalysis.backtest.core.source_config import SourceConfig
from hlanalysis.backtest.runner.hftbt_runner import RunConfig, run_one_question
from hlanalysis.backtest.runner.parallel import build_strategy_for_run
from hlanalysis.backtest.slot_config import backtest_params_from_slot
from hlanalysis.engine.config import load_strategies_config
from hlanalysis.marketdata.decision_input import from_backtest_params

PRUNE_THRESHOLD = 0.85  # v31 priceBucket favorite_threshold (C3)


def _read_parquet_or_none(p: Path):
    import pyarrow.parquet as pq

    return pq.read_table(p) if p.exists() else None


def _build_source(data_root: str, dt: int, prune: float | None) -> SourceConfig:
    resolved = from_backtest_params({}, track_default_source="mark")
    sc = SourceConfig(
        kind="hl_hip4",
        cache_root=data_root,
        hl_ref_source="hl_perp",
        hl_ref_event=resolved.reference_source,
        hl_ref_ticks=resolved.reference_ticks,
        reference_resample_seconds=dt,
        reference_warmup_seconds=0,
    )
    ds = sc.build()
    ds.leg_prune_favorite_threshold = prune  # build() ignores env in-proc; set directly
    return ds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()

    data_root = os.environ["HLBT_HL_DATA_ROOT"]
    slot_cfg = next(
        c for c in load_strategies_config(Path("config/strategy.yaml")).strategies if c.account_alias == "v31"
    )
    strategy_id, params = backtest_params_from_slot(slot_cfg, klass="priceBucket")
    dt = int(params["vol_sampling_dt_seconds"])
    print(f"strategy={strategy_id} dt={dt} favorite_threshold={params.get('favorite_threshold')}")

    run_cfg = RunConfig(scanner_interval_seconds=1, scan_mode="fixed")

    ds_off = _build_source(data_root, dt, None)
    ds_on = _build_source(data_root, dt, PRUNE_THRESHOLD)
    qs = ds_off.discover(start="2026-05-06", end="2026-06-19", kinds=("priceBucket",))
    qs = qs[args.skip : args.skip + args.n]

    legs_off_tot = legs_on_tot = 0
    wall_off_tot = wall_on_tot = 0.0
    all_identical = True
    for q in qs:
        outoff = Path(f"/tmp/prune_val/{q.question_id}_off")
        outon = Path(f"/tmp/prune_val/{q.question_id}_on")
        strat_off = build_strategy_for_run(strategy_id, params)
        strat_on = build_strategy_for_run(strategy_id, params)

        legs_off = sum(1 for la in ds_off.events_arrays(q).leg_arrays.values() if len(la.events) > 0)
        legs_on = sum(1 for la in ds_on.events_arrays(q).leg_arrays.values() if len(la.events) > 0)

        t = time.time()
        r_off = run_one_question(strat_off, ds_off, q, run_cfg, diagnostics_dir=outoff, fills_dir=outoff)
        w_off = time.time() - t
        t = time.time()
        r_on = run_one_question(strat_on, ds_on, q, run_cfg, diagnostics_dir=outon, fills_dir=outon)
        w_on = time.time() - t

        diag_off = _read_parquet_or_none(outoff / f"{q.question_id}.parquet")
        diag_on = _read_parquet_or_none(outon / f"{q.question_id}.parquet")
        # diagnostics + fills are written under the per-question parquet name.
        diag_ok = (diag_off is None and diag_on is None) or (
            diag_off is not None and diag_on is not None and diag_off.equals(diag_on)
        )
        pnl_ok = abs(r_off.realized_pnl_usd - r_on.realized_pnl_usd) < 1e-12
        fills_ok = r_off.n_fills == r_on.n_fills
        identical = diag_ok and pnl_ok and fills_ok
        all_identical &= identical
        legs_off_tot += legs_off
        legs_on_tot += legs_on
        wall_off_tot += w_off
        wall_on_tot += w_on
        print(
            f"{q.question_id}: legs {legs_off}->{legs_on}  pnl off={r_off.realized_pnl_usd:.4f} "
            f"on={r_on.realized_pnl_usd:.4f}  fills off={r_off.n_fills} on={r_on.n_fills}  "
            f"wall {w_off:.1f}->{w_on:.1f}s  diag_identical={diag_ok}  {'OK' if identical else '*** MISMATCH ***'}"
        )

    n = len(qs)
    print(
        f"\nSUMMARY n={n}  avg legs loaded {legs_off_tot / n:.2f} -> {legs_on_tot / n:.2f}  "
        f"avg wall {wall_off_tot / n:.2f}s -> {wall_on_tot / n:.2f}s  "
        f"({(1 - wall_on_tot / wall_off_tot) * 100:.0f}% faster)"
    )
    print("ALL BIT-IDENTICAL" if all_identical else "DECISION MISMATCH DETECTED")
    return 0 if all_identical else 1


if __name__ == "__main__":
    raise SystemExit(main())
