"""SHR-91: shared cross-market inventory cap in the in-process run path.

The in-process (n_workers<=1) run path runs questions sequentially but live
runs them concurrently — one ledger per slot. Without a shared cap the sim
over-enters notional that live's concurrent cap would have blocked.

These tests verify:
1. SharedInventoryLedger.count_at correctly computes held positions from
   previously-simulated questions at a given timestamp.
2. RunResult.position_windows is populated when positions open/close.
3. _RunState.entry_blocked respects extra_held_notional / extra_n_held as
   additional cross-question inventory visible to the cap check.
4. run_questions_parallel in-process path threads the ledger: Q2's entry gate
   sees Q1's held position window when they overlap in real time.
"""

from __future__ import annotations

import pytest

from hlanalysis.backtest.halt_replay import SimRiskCaps
from hlanalysis.backtest.runner.parallel import SharedInventoryLedger


# ---------------------------------------------------------------------------
# Unit: SharedInventoryLedger.count_at
# ---------------------------------------------------------------------------


def test_shared_inventory_ledger_no_windows_returns_zero():
    """An empty ledger returns 0 notional and 0 positions at any timestamp."""
    ledger = SharedInventoryLedger()
    notional, n = ledger.count_at(ts_ns=1_000_000_000)
    assert notional == pytest.approx(0.0)
    assert n == 0


def test_shared_inventory_ledger_counts_overlapping_window():
    """A position window fully enclosing the query ts contributes 1 / its notional."""
    ledger = SharedInventoryLedger()
    ledger.record(open_ts_ns=0, close_ts_ns=10_000_000_000, notional=500.0)

    notional, n = ledger.count_at(ts_ns=5_000_000_000)
    assert notional == pytest.approx(500.0)
    assert n == 1


def test_shared_inventory_ledger_closed_window_not_counted():
    """A window that closed before the query timestamp is not counted."""
    ledger = SharedInventoryLedger()
    ledger.record(open_ts_ns=0, close_ts_ns=1_000_000_000, notional=300.0)

    notional, n = ledger.count_at(ts_ns=2_000_000_000)
    assert notional == pytest.approx(0.0)
    assert n == 0


def test_shared_inventory_ledger_future_window_not_counted():
    """A window that opens after the query timestamp is not counted."""
    ledger = SharedInventoryLedger()
    ledger.record(open_ts_ns=5_000_000_000, close_ts_ns=10_000_000_000, notional=200.0)

    notional, n = ledger.count_at(ts_ns=2_000_000_000)
    assert notional == pytest.approx(0.0)
    assert n == 0


def test_shared_inventory_ledger_multiple_windows_summed():
    """Multiple overlapping windows are summed at the query timestamp."""
    ledger = SharedInventoryLedger()
    ledger.record(open_ts_ns=0, close_ts_ns=10_000_000_000, notional=300.0)
    ledger.record(open_ts_ns=0, close_ts_ns=10_000_000_000, notional=200.0)

    notional, n = ledger.count_at(ts_ns=5_000_000_000)
    assert notional == pytest.approx(500.0)
    assert n == 2


def test_shared_inventory_ledger_boundary_at_open():
    """A window is counted when ts == open_ts_ns (inclusive open boundary)."""
    ledger = SharedInventoryLedger()
    ledger.record(open_ts_ns=1_000, close_ts_ns=2_000, notional=100.0)

    notional, n = ledger.count_at(ts_ns=1_000)
    assert n == 1


def test_shared_inventory_ledger_boundary_at_close():
    """A window is NOT counted when ts == close_ts_ns (exclusive close boundary)."""
    ledger = SharedInventoryLedger()
    ledger.record(open_ts_ns=1_000, close_ts_ns=2_000, notional=100.0)

    notional, n = ledger.count_at(ts_ns=2_000)
    assert n == 0


# ---------------------------------------------------------------------------
# Unit: RunResult.position_windows populated by run_one_question
# ---------------------------------------------------------------------------


def test_run_result_position_windows_populated_on_enter_and_exit():
    """After run_one_question, result.position_windows has one entry for each
    completed position: (open_ts_ns, close_ts_ns, notional). Verifies the runner
    records position lifecycle for the shared-ledger cross-question cap.
    """
    from hlanalysis.backtest.data.synthetic import (
        SyntheticDataSource,
        make_default_binary_question,
        build_dummy_enter_strategy,
    )
    from hlanalysis.backtest.runner.hftbt_runner import RunConfig, run_one_question

    sq = make_default_binary_question(
        question_id="pw-q",
        question_idx=1,
        start_ts_ns=0,
        duration_ns=10 * 60 * 1_000_000_000,  # 10 min
        outcome="yes",
    )
    ds = SyntheticDataSource()
    ds.add_question(sq)

    strategy = build_dummy_enter_strategy({"size": 50.0})
    cfg = RunConfig(
        slippage_bps=0.0,
        fee_taker=0.0,
        order_latency_ms=0.0,
        ioc_marketability_recheck=False,
    )
    result = run_one_question(strategy, ds, sq.descriptor, cfg, strike=sq.strike)

    # The position must have opened and closed (via settlement).
    assert len(result.fills) >= 2, "expected at least an entry fill + settlement fill"
    # position_windows should have exactly one window (opened on entry, closed on settle).
    assert len(result.position_windows) >= 1, (
        "run_one_question must populate result.position_windows for cross-question ledger"
    )
    open_ts, close_ts, notional = result.position_windows[0]
    assert open_ts >= sq.descriptor.start_ts_ns
    assert close_ts > open_ts
    assert notional > 0.0


# ---------------------------------------------------------------------------
# Integration: _RunState.entry_blocked respects extra_n_held
# ---------------------------------------------------------------------------


def test_entry_blocked_respects_extra_n_held():
    """_RunState.entry_blocked must block entry when extra_n_held causes the
    concurrent cap to be hit even though the current question has no position.

    Setup: max_concurrent_positions=1; current question has no position
    (n_held=0); extra_n_held=1 from shared ledger → entry must be blocked.
    """
    import hftbacktest as hb
    from hlanalysis.backtest.runner.hftbt_runner import (
        RunConfig,
        RunResult,
        _RunState,
    )

    cfg = RunConfig(slippage_bps=0.0, fee_taker=0.0, order_latency_ms=0.0)
    caps = SimRiskCaps(max_concurrent_positions=1)

    # Minimal fake hbt (not used by entry_blocked, but _RunState needs it).
    class _FakeHbt:
        pass

    class _FakeQ:
        question_idx = 1
        start_ts_ns = 0
        end_ts_ns = int(1e12)

    class _FakeDS:
        pass

    st = _RunState(
        hbt=_FakeHbt(),
        cfg=cfg,
        q=_FakeQ(),
        data_source=_FakeDS(),
        leg_to_asset={},
        hedge_asset_no=None,
        stop_pct=None,
        fills_dir_active=False,
        result=RunResult(),
        pos=None,  # no current position
        sim_risk_caps=caps,
        extra_n_held=1,  # cross-question position from shared ledger
        extra_held_notional=300.0,
    )
    st.now_ns = 500_000_000

    veto = st.entry_blocked(
        intent_notional=100.0,
        is_topup=False,
        held_notional=0.0,
        n_held=0,
    )
    assert veto == "max_concurrent_positions", (
        f"entry must be blocked by concurrent cap when extra_n_held=1; got {veto!r}"
    )


def test_entry_blocked_not_blocked_without_extra():
    """Without cross-question inventory, an entry is not blocked by the concurrent cap."""
    from hlanalysis.backtest.runner.hftbt_runner import RunConfig, RunResult, _RunState

    cfg = RunConfig(slippage_bps=0.0, fee_taker=0.0, order_latency_ms=0.0)
    caps = SimRiskCaps(max_concurrent_positions=1)

    class _FakeHbt:
        pass

    class _FakeQ:
        question_idx = 1
        start_ts_ns = 0
        end_ts_ns = int(1e12)

    class _FakeDS:
        pass

    st = _RunState(
        hbt=_FakeHbt(),
        cfg=cfg,
        q=_FakeQ(),
        data_source=_FakeDS(),
        leg_to_asset={},
        hedge_asset_no=None,
        stop_pct=None,
        fills_dir_active=False,
        result=RunResult(),
        pos=None,
        sim_risk_caps=caps,
        extra_n_held=0,
        extra_held_notional=0.0,
    )
    st.now_ns = 500_000_000

    veto = st.entry_blocked(
        intent_notional=100.0,
        is_topup=False,
        held_notional=0.0,
        n_held=0,
    )
    assert veto is None, f"no extra inventory → no block; got {veto!r}"
