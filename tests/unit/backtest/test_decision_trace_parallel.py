"""Regression test: ``--decision-trace-out`` must work under parallel workers.

The per-scan decision trace originally only wrote rows on the in-process
(``--workers 1``) path; the trace writer was never threaded into the spawn
subprocess workers, so ``--workers > 1`` produced ZERO rows. The fix threads a
per-question ``decision_trace_dir`` through the picklable worker tuple (mirroring
``fills_dir`` / ``diagnostics_dir``) so each worker writes its question's trace,
then the CLI concatenates them into the final ``--decision-trace-out`` path.

These tests assert the worker path produces the SAME rows as the in-process
path, with the canonical schema keys.
"""

from __future__ import annotations

import json

from hlanalysis.backtest._cli_plumbing import _concat_jsonl, _strike_for_data_source
from hlanalysis.backtest.core.source_config import SourceConfig
from hlanalysis.backtest.data.synthetic import build_dummy_enter_strategy
from hlanalysis.backtest.runner.hftbt_runner import RunConfig
from hlanalysis.backtest.runner.parallel import run_questions_parallel

_CANONICAL_KEYS = {
    "ts_ns",
    "question_idx",
    "klass",
    "strategy_id",
    "action",
    "reason",
    "chosen_symbol",
    "chosen_side",
    "reference_price",
    "sigma",
    "p_model",
    "edge",
    "safety_d_entry",
    "safety_d_exit",
    "tte_s",
    "favorite_side",
    "intended_size",
    "intended_price",
    "bid_px",
    "bid_sz",
    "ask_px",
    "ask_sz",
    "position_qty",
    "position_avg_entry",
    "config_hash",
    "diag_fields",
}


def _run_cfg() -> RunConfig:
    return RunConfig(
        scanner_interval_seconds=60,
        slippage_bps=0.0,
        fee_taker=0.0,
        order_latency_ms=0.0,
        ioc_marketability_recheck=False,
    )


def _read_rows(path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def _trace_via(*, tmp_path, n_workers: int, label: str) -> list[dict]:
    """Run the synthetic question through ``run_questions_parallel`` and return
    the concatenated decision-trace rows.

    ``n_workers == 1`` exercises the in-process path (built data_source +
    strategy passed directly); ``n_workers >= 2`` exercises the real spawn
    subprocess workers that rebuild from ``SourceConfig``.
    """
    trace_dir = tmp_path / f"trace_{label}"
    source_config = SourceConfig(kind="synthetic")
    data_source = source_config.build()
    descriptors = list(data_source.discover(start="1970-01-01", end="2999-12-31"))
    assert descriptors, "synthetic source produced no questions"

    in_process = n_workers <= 1
    run_questions_parallel(
        descriptors=descriptors,
        strategy_id="_dummy_enter_yes",
        params={"size": 10.0},
        run_cfg=_run_cfg(),
        source_config=source_config,
        diagnostics_dir=None,
        fills_dir=None,
        strike_for=_strike_for_data_source("synthetic"),
        hedge_data_path=None,
        hedge_half_spread_bps=0.0,
        n_workers=n_workers,
        data_source=data_source if in_process else None,
        strategy=build_dummy_enter_strategy({"size": 10.0}) if in_process else None,
        decision_trace_dir=trace_dir,
        decision_trace_config_hash="deadbeef",
    )

    out = tmp_path / f"trace_{label}.jsonl"
    _concat_jsonl(trace_dir, out)
    assert out.exists(), f"{label}: concatenated trace file was not created"
    return _read_rows(out)


def test_parallel_workers_produce_trace_rows(tmp_path):
    """--workers 2 (spawn subprocess) must produce non-empty trace with schema."""
    rows = _trace_via(tmp_path=tmp_path, n_workers=2, label="w2")
    assert rows, "workers=2 produced ZERO decision-trace rows (the bug)"
    for row in rows:
        missing = _CANONICAL_KEYS - set(row.keys())
        assert not missing, f"row missing canonical keys: {missing}"
        extra = set(row.keys()) - _CANONICAL_KEYS
        assert not extra, f"row has unexpected keys: {extra}"


def test_workers_2_match_workers_1(tmp_path):
    """Parallel rows must equal in-process rows (order may differ)."""
    serial = _trace_via(tmp_path=tmp_path, n_workers=1, label="w1")
    parallel = _trace_via(tmp_path=tmp_path, n_workers=2, label="w2")
    assert serial, "workers=1 produced no rows"
    assert parallel, "workers=2 produced no rows"

    def _key(rows: list[dict]) -> list:
        return sorted(json.dumps(r, sort_keys=True) for r in rows)

    assert _key(serial) == _key(parallel), "workers=2 rows differ from workers=1"


def test_no_trace_dir_writes_nothing(tmp_path):
    """No decision_trace_dir => no trace files, no overhead."""
    source_config = SourceConfig(kind="synthetic")
    data_source = source_config.build()
    descriptors = list(data_source.discover(start="1970-01-01", end="2999-12-31"))
    run_questions_parallel(
        descriptors=descriptors,
        strategy_id="_dummy_enter_yes",
        params={"size": 10.0},
        run_cfg=_run_cfg(),
        source_config=source_config,
        diagnostics_dir=None,
        fills_dir=None,
        strike_for=_strike_for_data_source("synthetic"),
        hedge_data_path=None,
        hedge_half_spread_bps=0.0,
        n_workers=2,
        data_source=None,
        strategy=None,
        decision_trace_dir=None,
    )
    assert not list(tmp_path.glob("**/*.jsonl"))
