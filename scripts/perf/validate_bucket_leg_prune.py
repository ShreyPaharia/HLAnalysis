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


# Decision-bearing diagnostic columns. yes_bid/yes_ask/no_bid/no_ask are a
# binary-centric DISPLAY artifact (the runner stamps leg_symbols[0]/[1]'s book
# into the row); for a bucket leg[1] is a NO leg and leg[0] may be a pruned tail
# YES leg, so those columns differ when pruned — the strategy never reads them
# for a bucket decision. ref_price/sigma come from the reference feed (pruning-
# independent). p_model/edge_*/tau_yr/ln_sk + action + reason ARE the decision.
# Numeric decision quantities — must be bit-identical on every shared tick.
_NUM_COLS = ["p_model", "edge_yes", "edge_no", "sigma", "tau_yr", "ln_sk", "ref_price"]


def _diag_decisions_identical(off, on) -> bool:
    """True iff the two diagnostics tables encode identical DECISIONS.

    The runner only records a row on ticks where ``books`` is non-empty. Pruning
    removes untradeable legs from ``books``, which has two benign effects:

    1. On a tick where ONLY a pruned leg had a quote, the pruned run's
       ``if not books: continue`` skips ``strategy.evaluate`` → fewer rows. Such a
       tick can only ever be HOLD (the favorite YES leg has no book), so it
       changes no position/fill. We require the on-run ticks ⊆ off-run ticks and
       every off-only row to be a HOLD.

    2. On a shared HOLD tick the *reason string* can flip ``no_favorite`` →
       ``no_book``: unpruned sees the tail YES legs (books present but below the
       favorite threshold → "no_favorite"); pruned has them absent → "no_book".
       Same HOLD action, no fill — a pure diagnostic relabel.

    So the contract is: the **action** column is bit-identical on every shared
    tick, every numeric decision field is bit-identical, and any ``reason``
    difference occurs only on a HOLD tick. (Fills + realized PnL are checked
    separately and must be exactly equal.)
    """
    if off is None and on is None:
        return True
    if off is None or on is None:
        return False
    do = off.to_pandas().set_index("ts_ns")
    dn = on.to_pandas().set_index("ts_ns")
    if not set(dn.index).issubset(set(do.index)):
        return False
    off_only = do.loc[~do.index.isin(dn.index)]
    if len(off_only) and not (off_only["action"] == "hold").all():
        return False
    shared = sorted(set(dn.index))
    a = do.loc[shared].reset_index(drop=True)
    b = dn.loc[shared].reset_index(drop=True)
    if not (a["action"] == b["action"]).all():
        return False
    for c in _NUM_COLS:
        if not ((a[c] == b[c]) | (a[c].isna() & b[c].isna())).all():
            return False
    reason_diff = a["reason"] != b["reason"]
    if (reason_diff & (a["action"] != "hold")).any():
        return False  # a reason flip on a non-HOLD tick WOULD be a real change
    return True


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
    ap.add_argument(
        "--scan-interval",
        type=int,
        default=60,
        help="fixed scanner interval (s). Pruning equivalence is cadence-independent, so a coarse "
        "interval proves bit-identity ~60x faster; pass 1 to confirm at the sweep's live cadence.",
    )
    args = ap.parse_args()

    data_root = os.environ["HLBT_HL_DATA_ROOT"]
    slot_cfg = next(
        c for c in load_strategies_config(Path("config/strategy.yaml")).strategies if c.account_alias == "v31"
    )
    strategy_id, params = backtest_params_from_slot(slot_cfg, klass="priceBucket")
    dt = int(params["vol_sampling_dt_seconds"])
    print(
        f"strategy={strategy_id} dt={dt} favorite_threshold={params.get('favorite_threshold')} "
        f"scan_interval={args.scan_interval}s",
        flush=True,
    )

    run_cfg = RunConfig(scanner_interval_seconds=args.scan_interval, scan_mode="fixed")

    ds_off = _build_source(data_root, dt, None)
    ds_on = _build_source(data_root, dt, PRUNE_THRESHOLD)
    qs = ds_off.discover(start="2026-05-06", end="2026-06-19", kinds=("priceBucket",))
    qs = qs[args.skip : args.skip + args.n]

    legs_off_tot = legs_on_tot = 0
    wall_off_tot = wall_on_tot = 0.0
    all_identical = True
    for q in qs:
        d_off = Path(f"/tmp/prune_val/{q.question_id}_off/diag")
        f_off = Path(f"/tmp/prune_val/{q.question_id}_off/fills")
        d_on = Path(f"/tmp/prune_val/{q.question_id}_on/diag")
        f_on = Path(f"/tmp/prune_val/{q.question_id}_on/fills")
        strat_off = build_strategy_for_run(strategy_id, params)
        strat_on = build_strategy_for_run(strategy_id, params)

        legs_off = sum(1 for la in ds_off.events_arrays(q).leg_arrays.values() if len(la.events) > 0)
        legs_on = sum(1 for la in ds_on.events_arrays(q).leg_arrays.values() if len(la.events) > 0)

        t = time.time()
        r_off = run_one_question(strat_off, ds_off, q, run_cfg, diagnostics_dir=d_off, fills_dir=f_off)
        w_off = time.time() - t
        t = time.time()
        r_on = run_one_question(strat_on, ds_on, q, run_cfg, diagnostics_dir=d_on, fills_dir=f_on)
        w_on = time.time() - t

        diag_ok = _diag_decisions_identical(
            _read_parquet_or_none(d_off / f"{q.question_id}.parquet"),
            _read_parquet_or_none(d_on / f"{q.question_id}.parquet"),
        )
        pnl_ok = abs(r_off.realized_pnl_usd - r_on.realized_pnl_usd) < 1e-12

        # Fills: identical modulo the random cloid (see backtest.md determinism note).
        def _fkey(fills):
            return sorted((f.symbol, f.side, round(f.price, 9), round(f.size, 9), round(f.fee, 9)) for f in fills)

        n_off, n_on = len(r_off.fills), len(r_on.fills)
        fills_ok = _fkey(r_off.fills) == _fkey(r_on.fills)
        identical = diag_ok and pnl_ok and fills_ok
        all_identical &= identical
        legs_off_tot += legs_off
        legs_on_tot += legs_on
        wall_off_tot += w_off
        wall_on_tot += w_on
        print(
            f"{q.question_id}: legs {legs_off}->{legs_on}  pnl off={r_off.realized_pnl_usd:.4f} "
            f"on={r_on.realized_pnl_usd:.4f}  fills off={n_off} on={n_on}  "
            f"wall {w_off:.1f}->{w_on:.1f}s  diag_identical={diag_ok}  {'OK' if identical else '*** MISMATCH ***'}",
            flush=True,
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
