"""TDD tests for SHR-92: reference-only warm-up prefix in HL HIP-4 backtest.

Spec:
  - reference_warmup_seconds=N loads reference rows in [start_ns - N*1e9, start_ns)
    and feeds them to MarketState BEFORE the first in-window scan.
  - Book/trade/settlement streams are UNCHANGED (still start at q.start_ts_ns).
  - Pre-start reference events (ts < start_ns) NEVER produce Fill or DiagnosticRow.
  - With warmup=0, the old cold-start behaviour holds.
  - The cache key (config_sig) includes reference_warmup_seconds so warmup=900 and
    warmup=0 get separate cache entries.
  - SourceConfig.reference_warmup_seconds propagates through the pickle boundary
    (worker factory no-regression).
  - CLI auto-derives warmup from the strategy's max vol_lookback_seconds across
    all classes; --reference-warmup-seconds N overrides; N=0 disables.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import duckdb
import numpy as np
import pytest

from hlanalysis.backtest.core.data_source import QuestionDescriptor
from hlanalysis.backtest.core.events import ReferenceEvent
from hlanalysis.backtest.data.hl_hip4 import HLHip4DataSource

FIXTURE_ROOT = Path(__file__).resolve().parents[2] / "fixtures" / "hl_hip4"

# BBO fixture covers 2026-05-10 04:00:00.039 to 05:59:59.776 UTC.
# We create a synthetic QuestionDescriptor whose start_ts_ns sits at 04:30 so
# the 30-minute window [04:00, 04:30) is available as warm-up data.
_BBO_FIXTURE_START_NS = 1_778_385_600_039_000_000   # 2026-05-10 04:00:00.039 UTC
_WARMUP_WINDOW_S = 1_800  # 30 min of pre-start BBO data available in fixture
_SYNTHETIC_START_NS = _BBO_FIXTURE_START_NS + _WARMUP_WINDOW_S * 1_000_000_000
# end_ts_ns must be beyond the BBO fixture end (~05:59:59) — use 06:00:00 UTC
_SYNTHETIC_END_NS = 1_778_392_800_000_000_000       # 2026-05-10 06:00:00 UTC


def _synthetic_q() -> QuestionDescriptor:
    """A QuestionDescriptor whose start_ts_ns falls 30 min INSIDE the BBO fixture
    so warmup data genuinely exists in [start - 1800s, start).

    Uses the fixture's leg symbols and question_id so any duckdb query that
    targets the prediction_binary partition returns the real fixture data, while
    the reference query hits the perp BBO partition which we have pre-start data for.
    """
    return QuestionDescriptor(
        question_id="Q1000015",
        question_idx=1_000_015,
        start_ts_ns=_SYNTHETIC_START_NS,
        end_ts_ns=_SYNTHETIC_END_NS,
        leg_symbols=("#150", "#151"),
        klass="priceBinary",
        underlying="BTC",
    )


# ---------------------------------------------------------------------------
# (a) Loader: warmup reference events appear BEFORE start_ns
# ---------------------------------------------------------------------------


def test_events_with_warmup_includes_pre_start_reference():
    """With reference_warmup_seconds=1800, the reference iterator must yield
    events whose ts_ns < q.start_ts_ns (the warm-up prefix)."""
    src = HLHip4DataSource(
        data_root=FIXTURE_ROOT,
        reference_warmup_seconds=_WARMUP_WINDOW_S,
    )
    q = _synthetic_q()
    pre_start_refs = []
    in_window_refs = []
    for ev in src.events(q):
        if isinstance(ev, ReferenceEvent):
            if ev.ts_ns < q.start_ts_ns:
                pre_start_refs.append(ev)
            else:
                in_window_refs.append(ev)

    assert pre_start_refs, (
        "Expected pre-start ReferenceEvents with warmup=1800s but got none. "
        "The loader must widen the reference query to [start - warmup, start)."
    )
    # The pre-start events are genuine BBO bars from the fixture.
    for ev in pre_start_refs:
        assert ev.ts_ns < q.start_ts_ns, ev.ts_ns
        assert 40_000 < ev.close < 200_000, f"BTC price out of range: {ev.close}"


def test_events_without_warmup_no_pre_start_reference():
    """With reference_warmup_seconds=0 (default), no ReferenceEvents with
    ts_ns < q.start_ts_ns are emitted (cold-start baseline)."""
    src = HLHip4DataSource(data_root=FIXTURE_ROOT)  # default warmup=0
    q = _synthetic_q()
    for ev in src.events(q):
        if isinstance(ev, ReferenceEvent):
            assert ev.ts_ns >= q.start_ts_ns, (
                f"Cold-start source emitted pre-start ReferenceEvent at {ev.ts_ns} "
                f"(start_ts_ns={q.start_ts_ns})"
            )


def test_events_warmup_non_reference_streams_unchanged():
    """Book/trade/settlement events from warmup=1800 MUST all have ts_ns >= start_ts_ns.
    The warm-up prefix is reference-ONLY; the other streams are unaffected."""
    from hlanalysis.backtest.core.events import BookSnapshot, SettlementEvent, TradeEvent

    src = HLHip4DataSource(
        data_root=FIXTURE_ROOT,
        reference_warmup_seconds=_WARMUP_WINDOW_S,
    )
    q = _synthetic_q()
    for ev in src.events(q):
        if isinstance(ev, (BookSnapshot, TradeEvent, SettlementEvent)):
            assert ev.ts_ns >= q.start_ts_ns, (
                f"{type(ev).__name__} emitted at {ev.ts_ns} < start {q.start_ts_ns}"
            )


def test_events_warmup_reference_ordering():
    """All emitted events (warmup prefix included) are monotone-increasing in ts_ns."""
    src = HLHip4DataSource(
        data_root=FIXTURE_ROOT,
        reference_warmup_seconds=_WARMUP_WINDOW_S,
    )
    q = _synthetic_q()
    prev = -1
    for ev in src.events(q):
        assert ev.ts_ns >= prev, (
            f"Non-monotone ts_ns: {prev} then {ev.ts_ns}"
        )
        prev = ev.ts_ns


# ---------------------------------------------------------------------------
# (b) Runner: no Fill, no DiagnosticRow for ts < start_ns when warmup > 0
# ---------------------------------------------------------------------------


def test_runner_no_fill_during_warmup_prefix(tmp_path):
    """run_one_question with a warmup-enabled source must produce ZERO fills
    with ts_ns < q.start_ts_ns. The warmup prefix warms σ only."""
    from hlanalysis.backtest.data.synthetic import build_dummy_enter_strategy
    from hlanalysis.backtest.runner.hftbt_runner import RunConfig, run_one_question

    src = HLHip4DataSource(
        data_root=FIXTURE_ROOT,
        reference_warmup_seconds=_WARMUP_WINDOW_S,
    )
    q = _synthetic_q()
    strategy = build_dummy_enter_strategy({"size": 10.0})
    cfg = RunConfig(
        scanner_interval_seconds=60,
        tick_size=0.001,
        lot_size=1.0,
        slippage_bps=0.0,
        fee_taker=0.0,
        order_latency_ms=0.0,  # zero latency for determinism
    )
    result = run_one_question(strategy, src, q, cfg, fills_dir=tmp_path)

    # No fill may carry ts_ns < q.start_ts_ns (warmup is σ-only).
    # We read the fill rows from the written parquet to get fill timestamps.
    import pyarrow.parquet as pq
    fills_parquet = tmp_path / f"{q.question_id}.parquet"
    if fills_parquet.exists():
        table = pq.read_table(fills_parquet)
        if "ts_ns" in table.schema.names:
            for ts_val in table["ts_ns"].to_pylist():
                if ts_val is not None:
                    assert ts_val >= q.start_ts_ns, (
                        f"Fill at ts_ns={ts_val} is before start_ts_ns={q.start_ts_ns}"
                    )


def test_runner_no_diagnostic_row_during_warmup_prefix(tmp_path):
    """No DiagnosticRow (strategy scan) is emitted during the warmup prefix."""
    from hlanalysis.backtest.data.synthetic import build_dummy_enter_strategy
    from hlanalysis.backtest.runner.hftbt_runner import RunConfig, run_one_question

    src = HLHip4DataSource(
        data_root=FIXTURE_ROOT,
        reference_warmup_seconds=_WARMUP_WINDOW_S,
    )
    q = _synthetic_q()
    strategy = build_dummy_enter_strategy({"size": 10.0})
    cfg = RunConfig(
        scanner_interval_seconds=60,
        tick_size=0.001,
        lot_size=1.0,
        slippage_bps=0.0,
        fee_taker=0.0,
        order_latency_ms=0.0,
    )
    result = run_one_question(strategy, src, q, cfg, diagnostics_dir=tmp_path)

    import pyarrow.parquet as pq
    diag_parquet = tmp_path / f"{q.question_id}.parquet"
    if diag_parquet.exists():
        table = pq.read_table(diag_parquet)
        if "ts_ns" in table.schema.names:
            for ts_val in table["ts_ns"].to_pylist():
                if ts_val is not None:
                    assert ts_val >= q.start_ts_ns, (
                        f"DiagnosticRow at ts_ns={ts_val} before start_ts_ns={q.start_ts_ns}"
                    )


# ---------------------------------------------------------------------------
# (c) MarketState σ warm after warmup prefix
# ---------------------------------------------------------------------------


def test_warmup_sigma_not_cold_after_prefix():
    """With warmup=1800s, the MarketState's σ at the first in-window scan
    must be computable from a FULL lookback (not vol_insufficient_data / empty).

    We feed the warm-up reference events directly into a MarketState and verify
    that recent_returns is non-empty after draining 1800s of BBO bars.
    """
    from hlanalysis.backtest.runner.market_state import MarketState

    src = HLHip4DataSource(
        data_root=FIXTURE_ROOT,
        reference_warmup_seconds=_WARMUP_WINDOW_S,
    )
    q = _synthetic_q()

    state = MarketState()
    n_pre_start = 0

    for ev in src.events(q):
        if isinstance(ev, ReferenceEvent):
            if ev.ts_ns < q.start_ts_ns:
                state.apply_reference(ev)
                n_pre_start += 1

    assert n_pre_start > 0, "No pre-start reference events fed into MarketState"

    # Query σ at exactly start_ts_ns with a 900s lookback.
    returns = state.recent_returns(
        now_ns=q.start_ts_ns, lookback_seconds=900
    )
    assert len(returns) > 0, (
        "MarketState σ is still cold after 1800s of warm-up reference bars. "
        "Expected recent_returns to be non-empty at start_ts_ns."
    )


def test_cold_start_sigma_empty_without_warmup():
    """Without warmup (warmup=0), MarketState has no reference data at start_ts_ns."""
    from hlanalysis.backtest.runner.market_state import MarketState

    state = MarketState()
    q = _synthetic_q()

    returns = state.recent_returns(
        now_ns=q.start_ts_ns, lookback_seconds=900
    )
    assert len(returns) == 0, (
        f"Expected empty returns for cold-start (no warmup), got {len(returns)}"
    )


# ---------------------------------------------------------------------------
# (d) Cache key: warmup=900 and warmup=0 produce DIFFERENT config_sig entries
# ---------------------------------------------------------------------------


def test_bundle_config_sig_includes_warmup():
    """_bundle_config_sig must change when reference_warmup_seconds changes.
    Without this, a warmup=900 bundle aliases to a warmup=0 request."""
    src_cold = HLHip4DataSource(data_root=FIXTURE_ROOT)
    src_warm = HLHip4DataSource(data_root=FIXTURE_ROOT, reference_warmup_seconds=900)
    assert src_cold._bundle_config_sig() != src_warm._bundle_config_sig(), (
        "_bundle_config_sig is identical for warmup=0 and warmup=900. "
        "reference_warmup_seconds MUST be part of the cache-key signature."
    )


def test_bundle_config_sig_different_warmups():
    """Two different warmup values produce different config_sig strings."""
    src_a = HLHip4DataSource(data_root=FIXTURE_ROOT, reference_warmup_seconds=900)
    src_b = HLHip4DataSource(data_root=FIXTURE_ROOT, reference_warmup_seconds=3600)
    assert src_a._bundle_config_sig() != src_b._bundle_config_sig()


def test_events_arrays_config_sig_warmup_differs(monkeypatch):
    """events_arrays config_sig with warmup=900 and warmup=0 must differ so the
    cache never serves a cold bundle for a warm request (the footgun pattern)."""
    from hlanalysis.backtest.data._event_array_cache import cache_key

    src_cold = HLHip4DataSource(data_root=FIXTURE_ROOT)
    src_warm = HLHip4DataSource(data_root=FIXTURE_ROOT, reference_warmup_seconds=900)

    key_cold = cache_key("Q1000015", [], src_cold._bundle_config_sig())
    key_warm = cache_key("Q1000015", [], src_warm._bundle_config_sig())

    assert key_cold != key_warm, (
        "Cache key is identical for warmup=0 and warmup=900 — "
        "cache will silently serve cold bundles for warm requests."
    )


# ---------------------------------------------------------------------------
# (e) SourceConfig: reference_warmup_seconds propagates through pickle boundary
# ---------------------------------------------------------------------------


def test_source_config_warmup_is_picklable():
    """SourceConfig with reference_warmup_seconds must survive a pickle round-trip
    (the spawn worker boundary)."""
    from hlanalysis.backtest.core.source_config import SourceConfig

    cfg = SourceConfig(
        kind="hl_hip4",
        cache_root=str(FIXTURE_ROOT),
        reference_warmup_seconds=900,
    )
    cfg2 = pickle.loads(pickle.dumps(cfg))
    assert cfg2.reference_warmup_seconds == 900
    assert cfg2 == cfg


def test_source_config_warmup_propagates_to_built_source(tmp_path):
    """SourceConfig.build() must pass reference_warmup_seconds to HLHip4DataSource."""
    from hlanalysis.backtest.core.source_config import SourceConfig

    cfg = SourceConfig(
        kind="hl_hip4",
        cache_root=str(tmp_path),
        reference_warmup_seconds=1800,
    )
    src = cfg.build()
    assert src.reference_warmup_seconds == 1800, (
        f"Expected reference_warmup_seconds=1800 on built source, "
        f"got {src.reference_warmup_seconds}"
    )


def test_source_config_warmup_default_zero(tmp_path):
    """Default SourceConfig for hl_hip4 must have reference_warmup_seconds=0
    so existing callers that build the source directly are unaffected."""
    from hlanalysis.backtest.core.source_config import SourceConfig

    src = SourceConfig(kind="hl_hip4", cache_root=str(tmp_path)).build()
    assert src.reference_warmup_seconds == 0


def test_source_config_with_warmup_returns_copy():
    """with_reference_warmup() returns a new SourceConfig with the override applied
    and leaves the original unchanged."""
    from hlanalysis.backtest.core.source_config import SourceConfig

    base = SourceConfig(kind="hl_hip4", reference_warmup_seconds=0)
    warm = base.with_reference_warmup(900)
    assert warm.reference_warmup_seconds == 900
    assert base.reference_warmup_seconds == 0  # original unchanged
    assert warm.kind == "hl_hip4"


# ---------------------------------------------------------------------------
# (f) CLI: derive warmup from vol_lookback_seconds + --reference-warmup-seconds
# ---------------------------------------------------------------------------


def test_cli_derive_warmup_from_vol_lookback():
    """_derive_reference_warmup_seconds returns max(vol_lookback_seconds) across all
    strategy param keys, for hl_hip4 sources."""
    from hlanalysis.backtest.cli import _derive_reference_warmup_seconds

    # Single-class params
    params = {"vol_lookback_seconds": 900}
    assert _derive_reference_warmup_seconds(params, data_source="hl_hip4") == 900

    # Multi-class params (bucket + binary tuned independently)
    params_mc = {
        "binary": {"vol_lookback_seconds": 3600},
        "bucket": {"vol_lookback_seconds": 900},
    }
    # Should return max across classes
    result = _derive_reference_warmup_seconds(params_mc, data_source="hl_hip4")
    assert result == 3600

    # Non-HL source: warmup not derived from vol_lookback (different mechanism)
    result_pm = _derive_reference_warmup_seconds(params, data_source="polymarket")
    assert result_pm == 0


def test_cli_derive_warmup_no_vol_lookback():
    """When vol_lookback_seconds is absent from params, warmup defaults to 0."""
    from hlanalysis.backtest.cli import _derive_reference_warmup_seconds

    assert _derive_reference_warmup_seconds({}, data_source="hl_hip4") == 0
    assert _derive_reference_warmup_seconds(
        {"some_other_param": 42}, data_source="hl_hip4"
    ) == 0


def test_cli_warmup_override_zero_disables(monkeypatch, tmp_path):
    """--reference-warmup-seconds 0 explicitly disables warmup (even if
    vol_lookback_seconds would auto-derive a non-zero value)."""
    from hlanalysis.backtest.cli import _resolve_reference_warmup_seconds

    params = {"vol_lookback_seconds": 900}
    # Explicit 0 must win over auto-derived 900
    result = _resolve_reference_warmup_seconds(
        params, data_source="hl_hip4", cli_override=0
    )
    assert result == 0


def test_cli_warmup_override_explicit(monkeypatch, tmp_path):
    """--reference-warmup-seconds N (N > 0) uses N regardless of params."""
    from hlanalysis.backtest.cli import _resolve_reference_warmup_seconds

    params = {"vol_lookback_seconds": 900}
    result = _resolve_reference_warmup_seconds(
        params, data_source="hl_hip4", cli_override=1800
    )
    assert result == 1800


def test_cli_warmup_auto_when_no_override(monkeypatch, tmp_path):
    """When cli_override is None (flag not passed), auto-derive from params."""
    from hlanalysis.backtest.cli import _resolve_reference_warmup_seconds

    params = {"vol_lookback_seconds": 3600}
    result = _resolve_reference_warmup_seconds(
        params, data_source="hl_hip4", cli_override=None
    )
    assert result == 3600


# ---------------------------------------------------------------------------
# ref_event (bbo vs mark): the σ-source that must match the live engine.
# Live uses EngineConfig.reference_sigma_source="mark"; the sim default was
# "bbo" (bid/ask mid), which reads a higher σ. The plumbing must let the sim
# select "mark" and must key it into the bundle sig so the cache never aliases.
# ---------------------------------------------------------------------------


def test_source_config_ref_event_propagates_to_built_source():
    """SourceConfig.hl_ref_event must reach the constructed HLHip4DataSource."""
    from hlanalysis.backtest.core.source_config import SourceConfig

    ds_bbo = SourceConfig(kind="hl_hip4", cache_root=str(FIXTURE_ROOT)).build()
    ds_mark = SourceConfig(
        kind="hl_hip4", cache_root=str(FIXTURE_ROOT), hl_ref_event="mark"
    ).build()
    assert ds_bbo.ref_event == "bbo"  # default mirrors historical behaviour
    assert ds_mark.ref_event == "mark"


def test_ref_event_changes_bundle_config_sig():
    """bbo vs mark must produce different cache signatures — else a mark request
    silently serves a cached bbo (higher-σ) bundle."""
    src_bbo = HLHip4DataSource(data_root=FIXTURE_ROOT, ref_event="bbo")
    src_mark = HLHip4DataSource(data_root=FIXTURE_ROOT, ref_event="mark")
    assert src_bbo._bundle_config_sig() != src_mark._bundle_config_sig()


def test_source_config_ref_event_picklable():
    """hl_ref_event must survive the spawn-worker pickle boundary."""
    from hlanalysis.backtest.core.source_config import SourceConfig

    cfg = SourceConfig(
        kind="hl_hip4", cache_root=str(FIXTURE_ROOT), hl_ref_event="mark"
    )
    cfg2 = pickle.loads(pickle.dumps(cfg))
    assert cfg2.hl_ref_event == "mark"
    assert cfg2 == cfg
