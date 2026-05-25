"""PM-only watchdog tests for `_pm_check_unconfirmed_orders` and
`_pm_check_redemption_timeouts`. Both detectors are pure against slot state,
so we drive them with a SimpleNamespace stub and a real StateDAL instead of
spinning up the full EngineRuntime.
"""
from __future__ import annotations

from types import SimpleNamespace

from hlanalysis.engine.runtime import (
    PM_REDEMPTION_TIMEOUT_S,
    PM_UNCONFIRMED_THRESHOLD_S,
    _pm_check_redemption_timeouts,
    _pm_check_unconfirmed_orders,
)
from hlanalysis.engine.state import OpenOrder, StateDAL


def _make_slot(tmp_path, *, alias: str = "v31_pm"):
    dal = StateDAL(tmp_path / "state.db")
    dal.run_migrations()
    return SimpleNamespace(
        alias=alias,
        dal=dal,
        pm_alerted_unconfirmed_cloids=set(),
        pm_settlements={},
        pm_alerted_redemption_qidxs=set(),
    )


def _put_order(dal: StateDAL, *, cloid: str, status: str, last_update_ts_ns: int,
               placed_ts_ns: int | None = None, side: str = "buy",
               size: float = 10.0, price: float = 0.55) -> None:
    dal.upsert_order(OpenOrder(
        cloid=cloid, venue_oid=f"v-{cloid}", question_idx=42,
        symbol="0xdeadbeef", side=side, price=price, size=size, status=status,
        placed_ts_ns=placed_ts_ns or last_update_ts_ns,
        last_update_ts_ns=last_update_ts_ns, strategy_id="v31",
    ))


def test_unconfirmed_fires_for_open_order_past_threshold(tmp_path):
    slot = _make_slot(tmp_path)
    now = 1_000_000_000_000_000_000
    # last_update is THRESHOLD + 1s in the past
    stale_ts = now - int((PM_UNCONFIRMED_THRESHOLD_S + 1.0) * 1e9)
    _put_order(slot.dal, cloid="hla-v31-a", status="open", last_update_ts_ns=stale_ts)

    out = _pm_check_unconfirmed_orders(slot, now)
    assert len(out) == 1
    ev = out[0]
    assert ev.cloid == "hla-v31-a"
    assert ev.account_alias == "v31_pm"
    assert ev.age_seconds >= PM_UNCONFIRMED_THRESHOLD_S
    assert "hla-v31-a" in slot.pm_alerted_unconfirmed_cloids


def test_unconfirmed_silent_below_threshold(tmp_path):
    slot = _make_slot(tmp_path)
    now = 1_000_000_000_000_000_000
    # Just under the threshold
    fresh_ts = now - int((PM_UNCONFIRMED_THRESHOLD_S - 1.0) * 1e9)
    _put_order(slot.dal, cloid="hla-v31-b", status="open", last_update_ts_ns=fresh_ts)

    out = _pm_check_unconfirmed_orders(slot, now)
    assert out == []
    assert slot.pm_alerted_unconfirmed_cloids == set()


def test_unconfirmed_skips_non_open_status(tmp_path):
    slot = _make_slot(tmp_path)
    now = 1_000_000_000_000_000_000
    stale_ts = now - int((PM_UNCONFIRMED_THRESHOLD_S + 5.0) * 1e9)
    # pending/partially_filled are returned by live_orders() but only `open`
    # should trip the unconfirmed watchdog — partial fills are progressing,
    # pending is pre-ack.
    _put_order(slot.dal, cloid="hla-pending", status="pending", last_update_ts_ns=stale_ts)
    _put_order(slot.dal, cloid="hla-partial", status="partially_filled", last_update_ts_ns=stale_ts)
    out = _pm_check_unconfirmed_orders(slot, now)
    assert out == []


def test_unconfirmed_deduplicates_until_cloid_clears(tmp_path):
    slot = _make_slot(tmp_path)
    now = 1_000_000_000_000_000_000
    stale_ts = now - int((PM_UNCONFIRMED_THRESHOLD_S + 5.0) * 1e9)
    _put_order(slot.dal, cloid="hla-v31-c", status="open", last_update_ts_ns=stale_ts)

    first = _pm_check_unconfirmed_orders(slot, now)
    assert len(first) == 1
    # Re-run: same cloid still stale, but should NOT re-emit
    second = _pm_check_unconfirmed_orders(slot, now + 5_000_000_000)
    assert second == []

    # Now flip status to filled — alerted set should be evicted on next scan
    slot.dal.update_order_status(
        cloid="hla-v31-c", status="filled", now_ns=now + 10_000_000_000,
    )
    _pm_check_unconfirmed_orders(slot, now + 10_000_000_000)
    assert "hla-v31-c" not in slot.pm_alerted_unconfirmed_cloids


def test_redemption_fires_past_threshold(tmp_path):
    slot = _make_slot(tmp_path)
    settled_ts = 1_000_000_000_000_000_000
    # 7h old — past the 6h threshold
    now = settled_ts + int((PM_REDEMPTION_TIMEOUT_S + 3600.0) * 1e9)
    # Winner: realized_pnl > 0 → expected payout = qty
    slot.pm_settlements[42] = (settled_ts, "0xdeadbeefcafebabe", 100.0, 30.0)
    # Loser: realized_pnl < 0 → expected payout = 0
    slot.pm_settlements[43] = (settled_ts, "0xabcd", 50.0, -25.0)

    out = _pm_check_redemption_timeouts(slot, now)
    assert len(out) == 2
    by_q = {ev.question_idx: ev for ev in out}
    assert by_q[42].expected_payout_usd == 100.0
    assert by_q[43].expected_payout_usd == 0.0
    assert by_q[42].age_seconds >= PM_REDEMPTION_TIMEOUT_S
    assert slot.pm_alerted_redemption_qidxs == {42, 43}


def test_redemption_silent_below_threshold(tmp_path):
    slot = _make_slot(tmp_path)
    settled_ts = 1_000_000_000_000_000_000
    # 5h after settlement — under the 6h threshold
    now = settled_ts + int((PM_REDEMPTION_TIMEOUT_S - 3600.0) * 1e9)
    slot.pm_settlements[42] = (settled_ts, "0xdead", 100.0, 30.0)
    out = _pm_check_redemption_timeouts(slot, now)
    assert out == []
    assert slot.pm_alerted_redemption_qidxs == set()


def test_redemption_dedupes_after_first_emit(tmp_path):
    slot = _make_slot(tmp_path)
    settled_ts = 1_000_000_000_000_000_000
    now = settled_ts + int((PM_REDEMPTION_TIMEOUT_S + 3600.0) * 1e9)
    slot.pm_settlements[42] = (settled_ts, "0xdead", 100.0, 30.0)

    first = _pm_check_redemption_timeouts(slot, now)
    assert len(first) == 1
    # Re-run an hour later — already alerted, must not re-emit
    second = _pm_check_redemption_timeouts(slot, now + 3600 * 10**9)
    assert second == []
