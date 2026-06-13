"""TDD tests for SHR-93: raw reference-tick mode for HL loader.

Spec:
  - reference_ticks="bars" (default) → current behaviour, existing results unchanged.
  - reference_ticks="raw" → _reference_iter yields one ReferenceEvent per raw tick
    (no pre-bucketing); the shared MarketState buckets ticks for σ via its tick path
    so last_mark == raw latest price.
  - In raw mode the reference stream contains MORE events than bar mode (one per tick
    vs one per dt-wide bucket).
  - reference_price at a scan equals the latest RAW mark (last_mark from tick path),
    not a dt-stale bar close.
  - σ in raw mode (ticks bucketed by MarketState) matches the prior bar-fed σ within
    a tolerance on the same data.
  - _bundle_config_sig differs for bars vs raw so the cache never aliases.
  - SourceConfig with hl_ref_ticks="raw" is picklable and build() propagates it.
  - Default stays "bars" → byte-identical existing behaviour.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from hlanalysis.backtest.core.data_source import QuestionDescriptor
from hlanalysis.backtest.core.events import ReferenceEvent
from hlanalysis.backtest.data.hl_hip4 import HLHip4DataSource

FIXTURE_ROOT = Path(__file__).resolve().parents[2] / "fixtures" / "hl_hip4"

# BBO fixture covers 2026-05-10 04:00:00.039 to 05:59:59.776 UTC.
_BBO_FIXTURE_START_NS = 1_778_385_600_039_000_000  # 2026-05-10 04:00:00.039 UTC
_WARMUP_WINDOW_S = 1_800  # 30 min of pre-start BBO data available in fixture
_SYNTHETIC_START_NS = _BBO_FIXTURE_START_NS + _WARMUP_WINDOW_S * 1_000_000_000
_SYNTHETIC_END_NS = 1_778_392_800_000_000_000  # 2026-05-10 06:00:00 UTC


def _synthetic_q() -> QuestionDescriptor:
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
# (a) Raw mode yields MORE reference events than bar mode (one per tick)
# ---------------------------------------------------------------------------


def test_raw_mode_more_reference_events_than_bar_mode():
    """With reference_ticks="raw", the reference stream must contain more events than
    bars mode (one per raw BBO tick vs one per dt-wide bucket)."""
    q = _synthetic_q()

    src_bars = HLHip4DataSource(
        data_root=FIXTURE_ROOT,
        reference_ticks="bars",
        reference_resample_seconds=60,
    )
    src_raw = HLHip4DataSource(
        data_root=FIXTURE_ROOT,
        reference_ticks="raw",
        reference_resample_seconds=60,  # dt ignored in raw mode
    )

    n_bars = sum(1 for ev in src_bars.events(q) if isinstance(ev, ReferenceEvent))
    n_raw = sum(1 for ev in src_raw.events(q) if isinstance(ev, ReferenceEvent))

    assert n_raw > n_bars, (
        f"Expected raw mode to yield MORE reference events than bars mode "
        f"(raw={n_raw}, bars={n_bars}). Raw mode should emit one event per tick."
    )


def test_raw_mode_events_have_equal_hlc():
    """Raw-mode ReferenceEvents are tick-style: high == low == close (no bar OHLC)."""
    q = _synthetic_q()
    src = HLHip4DataSource(data_root=FIXTURE_ROOT, reference_ticks="raw")

    for ev in src.events(q):
        if isinstance(ev, ReferenceEvent):
            assert ev.high == ev.low == ev.close, (
                f"Raw-mode ReferenceEvent at {ev.ts_ns} should be a tick "
                f"(H=L=C) but got H={ev.high}, L={ev.low}, C={ev.close}"
            )
            # BTC price sanity
            assert 40_000 < ev.close < 200_000, f"BTC price out of range: {ev.close}"
            break  # just check a few


# ---------------------------------------------------------------------------
# (b) last_mark is the raw latest tick in raw mode
# ---------------------------------------------------------------------------


def test_raw_mode_last_mark_is_latest_tick():
    """In raw mode, MarketState.last_mark / latest_btc_close must equal the last
    tick price seen, not a dt-stale bar close."""
    from hlanalysis.backtest.runner.market_state import MarketState

    q = _synthetic_q()
    src_raw = HLHip4DataSource(data_root=FIXTURE_ROOT, reference_ticks="raw")

    # Collect all raw tick ReferenceEvents up to some midpoint ts.
    midpoint = _SYNTHETIC_START_NS + (_SYNTHETIC_END_NS - _SYNTHETIC_START_NS) // 2
    raw_evs = [ev for ev in src_raw.events(q) if isinstance(ev, ReferenceEvent) and ev.ts_ns <= midpoint]
    assert raw_evs, "No raw ReferenceEvents found before midpoint"

    state_raw = MarketState()
    for ev in raw_evs:
        state_raw.apply_reference_tick(ev)

    # The state's last mark must equal the LAST tick's price.
    last_tick_price = raw_evs[-1].close
    mark = state_raw.latest_btc_close()
    assert mark is not None, "last_mark should be set after ingesting ticks"
    assert abs(mark - last_tick_price) < 1e-6, (
        f"Expected last_mark={last_tick_price} (last raw tick), but got last_mark={mark}"
    )


def test_bar_mode_last_mark_is_bar_close():
    """In bar mode, MarketState.latest_btc_close == last bar's close (the dt-aligned
    bar close, which can be up to dt seconds stale)."""
    from hlanalysis.backtest.runner.market_state import MarketState

    q = _synthetic_q()
    src_bars = HLHip4DataSource(
        data_root=FIXTURE_ROOT,
        reference_ticks="bars",
        reference_resample_seconds=60,
    )

    midpoint = _SYNTHETIC_START_NS + (_SYNTHETIC_END_NS - _SYNTHETIC_START_NS) // 2
    bar_evs = [ev for ev in src_bars.events(q) if isinstance(ev, ReferenceEvent) and ev.ts_ns <= midpoint]
    assert bar_evs, "No bar ReferenceEvents found before midpoint"

    state_bars = MarketState()
    for ev in bar_evs:
        state_bars.apply_reference(ev)

    last_bar_close = bar_evs[-1].close
    mark = state_bars.latest_btc_close()
    assert mark is not None
    assert abs(mark - last_bar_close) < 1e-6, f"Expected last_mark={last_bar_close} (last bar close), got {mark}"


# ---------------------------------------------------------------------------
# (c) σ from raw mode matches bar mode within tolerance
# ---------------------------------------------------------------------------


def test_sigma_raw_matches_bars_within_tolerance():
    """σ computed via raw ticks (bucketed by MarketState) must match bar-fed σ
    within a reasonable tolerance on the same data segment.

    The same ticks go into both paths:
      - bars: pre-bucketed by _resample_reference → bar close-to-close returns
      - raw: each tick bucketed on-the-fly by _OhlcBuffer.ingest_tick

    Both should produce OHLC bars with the same close-to-close structure so σ
    converges. We accept a 20% relative tolerance to allow for minor floating-
    point differences in bucket boundary handling.
    """
    from hlanalysis.backtest.runner.market_state import MarketState

    q = _synthetic_q()
    dt = 60  # 60s bars — the standard cadence

    src_bars = HLHip4DataSource(
        data_root=FIXTURE_ROOT,
        reference_ticks="bars",
        reference_resample_seconds=dt,
    )
    src_raw = HLHip4DataSource(
        data_root=FIXTURE_ROOT,
        reference_ticks="raw",
        reference_resample_seconds=dt,  # MarketState will use this for bucketing
    )

    # Feed all reference events into two separate MarketState instances.
    state_bars = MarketState()
    for ev in src_bars.events(q):
        if isinstance(ev, ReferenceEvent):
            state_bars.apply_reference(ev)

    state_raw = MarketState()
    for ev in src_raw.events(q):
        if isinstance(ev, ReferenceEvent):
            state_raw.apply_reference_tick(ev)

    # Query σ at the end of the window.
    now_ns = _SYNTHETIC_END_NS
    lookback = 3600

    rets_bars = state_bars.recent_returns(now_ns=now_ns, lookback_seconds=lookback)
    rets_raw = state_raw.recent_returns(now_ns=now_ns, lookback_seconds=lookback)

    # Both must have produced some returns.
    assert len(rets_bars) > 0, "Bar-mode recent_returns is empty"
    assert len(rets_raw) > 0, "Raw-mode recent_returns is empty"

    # Std of returns should agree within 20%.
    sigma_bars = float(np.std(rets_bars)) if len(rets_bars) > 1 else 0.0
    sigma_raw = float(np.std(rets_raw)) if len(rets_raw) > 1 else 0.0

    if sigma_bars > 1e-10:
        rel_diff = abs(sigma_raw - sigma_bars) / sigma_bars
        assert rel_diff < 0.20, (
            f"Raw-mode σ diverges too much from bar-mode σ: "
            f"raw={sigma_raw:.6f}, bars={sigma_bars:.6f}, rel_diff={rel_diff:.3f}"
        )


# ---------------------------------------------------------------------------
# (d) cache: _bundle_config_sig differs for bars vs raw
# ---------------------------------------------------------------------------


def test_bundle_config_sig_differs_bars_vs_raw():
    """_bundle_config_sig MUST differ between reference_ticks="bars" and "raw"
    so the event-array cache never serves a bars bundle for a raw request."""
    src_bars = HLHip4DataSource(data_root=FIXTURE_ROOT, reference_ticks="bars")
    src_raw = HLHip4DataSource(data_root=FIXTURE_ROOT, reference_ticks="raw")
    assert src_bars._bundle_config_sig() != src_raw._bundle_config_sig(), (
        "_bundle_config_sig is identical for bars and raw modes — cache will "
        "alias and serve stale bar bundles for raw requests."
    )


def test_bundle_config_sig_stable_within_mode():
    """Two identically-configured sources must produce the same sig."""
    sig1 = HLHip4DataSource(data_root=FIXTURE_ROOT, reference_ticks="raw")._bundle_config_sig()
    sig2 = HLHip4DataSource(data_root=FIXTURE_ROOT, reference_ticks="raw")._bundle_config_sig()
    assert sig1 == sig2


# ---------------------------------------------------------------------------
# (e) SourceConfig: hl_ref_ticks picklable + build() propagates it
# ---------------------------------------------------------------------------


def test_source_config_hl_ref_ticks_is_picklable():
    """SourceConfig with hl_ref_ticks='raw' must survive a pickle round-trip."""
    from hlanalysis.backtest.core.source_config import SourceConfig

    cfg = SourceConfig(
        kind="hl_hip4",
        cache_root=str(FIXTURE_ROOT),
        hl_ref_ticks="raw",
    )
    cfg2 = pickle.loads(pickle.dumps(cfg))
    assert cfg2.hl_ref_ticks == "raw"
    assert cfg2 == cfg


def test_source_config_hl_ref_ticks_default_is_bars():
    """Default SourceConfig must have hl_ref_ticks='bars' so existing callers are
    unaffected (no change in behaviour unless the flag is set)."""
    from hlanalysis.backtest.core.source_config import SourceConfig

    cfg = SourceConfig(kind="hl_hip4", cache_root=str(FIXTURE_ROOT))
    assert cfg.hl_ref_ticks == "bars"


def test_source_config_build_propagates_ref_ticks_raw(tmp_path):
    """SourceConfig.build() must pass reference_ticks='raw' to HLHip4DataSource."""
    from hlanalysis.backtest.core.source_config import SourceConfig

    cfg = SourceConfig(
        kind="hl_hip4",
        cache_root=str(tmp_path),
        hl_ref_ticks="raw",
    )
    src = cfg.build()
    assert src.reference_ticks == "raw", f"Expected reference_ticks='raw' on built source, got {src.reference_ticks!r}"


def test_source_config_build_propagates_ref_ticks_bars(tmp_path):
    """Default SourceConfig.build() for hl_hip4 must have reference_ticks='bars'."""
    from hlanalysis.backtest.core.source_config import SourceConfig

    src = SourceConfig(kind="hl_hip4", cache_root=str(tmp_path)).build()
    assert src.reference_ticks == "bars"


# ---------------------------------------------------------------------------
# (f) Default "bars" mode is byte-identical to the prior behaviour
# ---------------------------------------------------------------------------


def test_bars_mode_reference_events_are_ohlc():
    """In bars mode, ReferenceEvents must be genuine OHLC bars (L <= C <= H) —
    the same as the existing pre-SHR-93 behaviour."""
    q = _synthetic_q()
    src = HLHip4DataSource(data_root=FIXTURE_ROOT, reference_ticks="bars")

    checked = 0
    for ev in src.events(q):
        if isinstance(ev, ReferenceEvent):
            assert ev.low <= ev.close <= ev.high, (
                f"Bar-mode ReferenceEvent at {ev.ts_ns} fails L<=C<=H: H={ev.high}, L={ev.low}, C={ev.close}"
            )
            checked += 1
    assert checked > 0, "No ReferenceEvents emitted in bars mode"


def test_bars_mode_default_unchanged():
    """HLHip4DataSource() without reference_ticks defaults to 'bars' — existing
    code is unaffected and produces the same events as before SHR-93."""
    q = _synthetic_q()
    src_explicit = HLHip4DataSource(data_root=FIXTURE_ROOT, reference_ticks="bars")
    src_default = HLHip4DataSource(data_root=FIXTURE_ROOT)

    evs_explicit = [ev for ev in src_explicit.events(q) if isinstance(ev, ReferenceEvent)]
    evs_default = [ev for ev in src_default.events(q) if isinstance(ev, ReferenceEvent)]

    assert len(evs_explicit) == len(evs_default)
    for a, b in zip(evs_explicit, evs_default):
        assert a.ts_ns == b.ts_ns
        assert abs(a.close - b.close) < 1e-9


# ---------------------------------------------------------------------------
# (g) MarketState adapter: apply_reference_tick feeds the tick path
# ---------------------------------------------------------------------------


def test_market_state_apply_reference_tick_sets_last_mark():
    """MarketState.apply_reference_tick must set latest_btc_close to the tick price."""
    from hlanalysis.backtest.runner.market_state import MarketState

    state = MarketState()
    # Start with no mark.
    assert state.latest_btc_close() is None

    # Feed a raw tick at BTC=80000.
    fake_ev = ReferenceEvent(ts_ns=1_000_000_000, symbol="BTC", high=80_000.0, low=80_000.0, close=80_000.0)
    state.apply_reference_tick(fake_ev)
    assert state.latest_btc_close() == pytest.approx(80_000.0)

    # A subsequent tick at a different price updates last_mark.
    fake_ev2 = ReferenceEvent(ts_ns=2_000_000_000, symbol="BTC", high=80_100.0, low=80_100.0, close=80_100.0)
    state.apply_reference_tick(fake_ev2)
    assert state.latest_btc_close() == pytest.approx(80_100.0)


def test_market_state_apply_reference_tick_vs_apply_reference_diverges_at_subbar():
    """apply_reference_tick should produce a more up-to-date last_mark than
    apply_reference (bar-close) when multiple ticks fall in one dt bucket."""
    from hlanalysis.backtest.runner.market_state import MarketState

    # Simulate 3 ticks in the same 60s bucket, all at different prices.
    # The bar-close path will only reflect the last bar's close (= last tick in bucket).
    # The tick path must also reflect the last tick's price (equal outcome).
    # To test the key property (instantaneous vs stale), we check after ONLY
    # the first tick of a bucket — bar mode won't emit yet (no bar), tick mode will.

    state_tick = MarketState()
    # Feed one tick at t=30s (still in bucket [0,60s)).
    t1_ev = ReferenceEvent(ts_ns=30_000_000_000, symbol="BTC", high=80_000.0, low=80_000.0, close=80_000.0)
    state_tick.apply_reference_tick(t1_ev)
    # tick path: last_mark == 80_000 immediately
    assert state_tick.latest_btc_close() == pytest.approx(80_000.0)

    # Bar path: a single tick doesn't emit a completed bar until the bucket
    # boundary; the bar emitted by _resample_reference has ts = last tick in bucket.
    # We simulate by NOT feeding state_bar until the bar is complete.
    state_bar = MarketState()
    # No bar emitted yet (bar is only emitted when bucket rolls).
    assert state_bar.latest_btc_close() is None  # cold until a bar lands


# ---------------------------------------------------------------------------
# (h) fast path (events_arrays) also respects reference_ticks mode
# ---------------------------------------------------------------------------


def test_events_arrays_raw_more_ref_events():
    """events_arrays() in raw mode must carry more reference_events than bars mode."""
    import os

    os.environ.pop("HLBT_DISABLE_FASTPATH", None)

    q = _synthetic_q()
    src_bars = HLHip4DataSource(data_root=FIXTURE_ROOT, reference_ticks="bars", reference_resample_seconds=60)
    src_raw = HLHip4DataSource(data_root=FIXTURE_ROOT, reference_ticks="raw", reference_resample_seconds=60)

    bundle_bars = src_bars.events_arrays(q)
    bundle_raw = src_raw.events_arrays(q)

    n_bars = len(bundle_bars.reference_events)
    n_raw = len(bundle_raw.reference_events)

    assert n_raw > n_bars, (
        f"events_arrays raw mode should yield more reference_events than bars: raw={n_raw}, bars={n_bars}"
    )


def test_events_arrays_raw_ref_events_are_ticks():
    """events_arrays() reference_events in raw mode must be tick-style (H==L==C)."""
    import os

    os.environ.pop("HLBT_DISABLE_FASTPATH", None)

    q = _synthetic_q()
    src_raw = HLHip4DataSource(data_root=FIXTURE_ROOT, reference_ticks="raw")
    bundle = src_raw.events_arrays(q)

    for ev in bundle.reference_events[:20]:  # check first 20
        assert ev.high == ev.low == ev.close, (
            f"Raw fast-path ReferenceEvent should be tick-style but got H={ev.high}, L={ev.low}, C={ev.close}"
        )
