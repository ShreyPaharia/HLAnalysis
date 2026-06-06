"""The in-process (n_workers<=1) run path must use the caller's already-built,
correctly-configured data source + strategy — NOT reconstruct them from a
zero-arg factory.

Regression for the HL reference-resample bug: cmd_run builds an HLHip4DataSource
with reference_resample_seconds = config.vol_sampling_dt_seconds (e.g. 5), but
the refactor routed even serial runs through run_questions_parallel ->
source_config.build(), which would drop reference_resample_seconds (reverting to
60s) -> inflated sigma -> every tick gated -> 0 trades.
"""
from __future__ import annotations

from types import SimpleNamespace

from hlanalysis.backtest.core.source_config import SourceConfig
from hlanalysis.backtest.runner import parallel as P
from hlanalysis.backtest.runner.hftbt_runner import RunConfig


def test_in_process_path_uses_provided_source_and_strategy(monkeypatch):
    captured = {}

    def fake_run_one_question(strategy, data_source, q, cfg, **kw):
        captured["source"] = data_source
        captured["strategy"] = strategy
        return SimpleNamespace(realized_pnl_usd=1.0, fills=[1, 2])

    monkeypatch.setattr(P, "run_one_question", fake_run_one_question)

    sentinel_source = SimpleNamespace(resolved_outcome=lambda q: "yes")
    sentinel_strategy = object()
    q = SimpleNamespace(question_id="q1", start_ts_ns=0, end_ts_ns=1)

    results = P.run_questions_parallel(
        descriptors=[q],
        data_source=sentinel_source,
        strategy=sentinel_strategy,
        strategy_id="unused",
        params={},
        run_cfg=RunConfig(),
        # A config whose build() would raise — proves the in-process path never
        # reconstructs the source (it uses the provided one directly).
        source_config=SourceConfig(kind="should_not_build"),
        diagnostics_dir=None,
        fills_dir=None,
        strike_for=lambda _q: 0.0,
        hedge_data_path=None,
        hedge_half_spread_bps=1.0,
        n_workers=1,
    )

    assert captured["source"] is sentinel_source, (
        "in-process path must use the provided source, not reconstruct it"
    )
    assert captured["strategy"] is sentinel_strategy
    assert results[0].n_fills == 2
    assert results[0].outcome == "yes"
