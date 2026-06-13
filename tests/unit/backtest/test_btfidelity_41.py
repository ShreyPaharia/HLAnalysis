"""Test for finding #41: end-of-data boundary comparison off-by-one.

Bug: hftbt_runner.py:~599 uses `if now_ns >= q.end_ts_ns: break` while the
backtest core (question.py:~37) uses strict `now_ns > q.end_ts_ns` for the
``settled`` flag.  The runner breaks one tick too early — a scan tick at
EXACTLY end_ts_ns is dropped, potentially missing the final scan before
settlement.

Fix: change `>=` to `>` so the runner's break condition matches the core's
convention: `now_ns == end_ts_ns` is STILL a valid scan tick; only
`now_ns > end_ts_ns` is post-settlement.
"""
from __future__ import annotations

import pytest


def test_question_view_settled_at_end_ts():
    """Core: build_question_view uses now_ns > end_ts_ns for settled.

    A tick at EXACTLY end_ts_ns is NOT settled (settled = False).
    """
    from hlanalysis.backtest.core.question import build_question_view
    from hlanalysis.backtest.core.data_source import QuestionDescriptor

    end_ts = 600_000_000_000  # 600s
    q = QuestionDescriptor(
        question_id="q1",
        question_idx=1,
        start_ts_ns=0,
        end_ts_ns=end_ts,
        leg_symbols=("YES", "NO"),
        klass="priceBinary",
        underlying="BTC",
    )

    # Exactly at end_ts_ns → NOT yet settled (strict >)
    qv_at = build_question_view(q, now_ns=end_ts, strike=100.0)
    assert qv_at.settled is False, (
        f"Core: now_ns == end_ts_ns should NOT be settled (strict >); got settled={qv_at.settled}"
    )

    # One nanosecond AFTER → settled
    qv_after = build_question_view(q, now_ns=end_ts + 1, strike=100.0)
    assert qv_after.settled is True, (
        f"Core: now_ns > end_ts_ns should be settled; got settled={qv_after.settled}"
    )


def test_runner_break_uses_strict_greater_than():
    """Runner: the scan loop break condition must be now_ns > end_ts_ns (not >=).

    We run a question and check that a scan at EXACTLY end_ts_ns is NOT skipped.
    The question runs from 0 to 120s; with scanner_interval=30s the scan ticks
    are at 30s, 60s, 90s, 120s. With `>= end_ts_ns` the 120s scan is dropped
    (break fires before the scan logic runs); with `> end_ts_ns` it fires.

    NOTE: hftbacktest returns rc=1 when the event array is exhausted BEFORE
    the elapse target. We use scanner_interval=30s instead of 60s because
    with 60s intervals hbt processes all remaining events in one step and
    returns rc=1 before we can check now_ns. With 30s intervals, hbt advances
    through intermediate events and can reach now_ns=120s with rc=0.

    Observable effect: n_decisions at 120s fires with the fix (>), not with
    the bug (>=).
    """
    from hlanalysis.backtest.data.synthetic import (
        SyntheticDataSource,
        build_dummy_enter_strategy,
        make_default_binary_question,
    )
    from hlanalysis.backtest.runner.hftbt_runner import RunConfig, run_one_question

    # Question: 0 → 120s; scan at 30s intervals → ticks at 30,60,90,120s.
    # The 120s tick is exactly at end_ts_ns.
    duration_ns = 120 * 1_000_000_000
    sq = make_default_binary_question(start_ts_ns=0, duration_ns=duration_ns)
    ds = SyntheticDataSource()
    ds.add_question(sq)

    # Strategy that always tries to enter (just to increment n_decisions)
    strat = build_dummy_enter_strategy({"size": 5.0})

    cfg = RunConfig(
        scanner_interval_seconds=30,  # 30s interval: scans at 30,60,90,120s
        slippage_bps=0.0,
        fee_taker=0.0,
        ioc_marketability_recheck=False,
    )

    res = run_one_question(strat, ds, sq.descriptor, cfg, strike=sq.strike)

    # With `>= end_ts_ns`: scan at 120s is skipped → n_decisions ≤ 3 (30,60,90).
    # With `> end_ts_ns`: scan at 120s fires → n_decisions = 4.
    #
    # The fix ensures now_ns == end_ts_ns is processed, not dropped.
    assert res.n_decisions >= 4, (
        f"Runner must process the scan at exactly end_ts_ns (now_ns == end_ts_ns "
        f"should NOT break early); expected n_decisions >= 4 (scans at 30,60,90,120s), "
        f"got {res.n_decisions}. "
        f"If n_decisions == 3, the runner uses `>=` (off-by-one bug) instead of `>`."
    )


def test_runner_break_boundary_consistent_with_core():
    """A tick at end_ts_ns must be processed by the runner (consistent with core).

    Core says: settled = now_ns > end_ts_ns (exclusive end). Runner must agree:
    the scan at now_ns == end_ts_ns fires (not skipped), and the question view
    built at that point reflects settled=False.
    """
    from hlanalysis.backtest.core.question import build_question_view
    from hlanalysis.backtest.core.data_source import QuestionDescriptor

    end_ts_ns = 600_000_000_000
    q = QuestionDescriptor(
        question_id="boundary-q",
        question_idx=1,
        start_ts_ns=0,
        end_ts_ns=end_ts_ns,
        leg_symbols=("YES", "NO"),
        klass="priceBinary",
        underlying="BTC",
    )

    # Boundary tick: now_ns == end_ts_ns → core says NOT settled.
    # The runner (post-fix) must NOT break before processing this tick.
    # We verify the core's convention directly:
    qv = build_question_view(q, now_ns=end_ts_ns, strike=1000.0)
    assert not qv.settled, "tick at end_ts_ns is the last valid scan, not yet settled"

    # Post-settlement: now_ns > end_ts_ns → settled.
    qv_after = build_question_view(q, now_ns=end_ts_ns + 1, strike=1000.0)
    assert qv_after.settled, "tick past end_ts_ns must be settled"
