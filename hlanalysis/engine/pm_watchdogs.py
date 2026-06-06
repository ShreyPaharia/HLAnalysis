"""Pure PM watchdog detectors: scan an AccountSlot's live order / settlement
state and return the alert events that are now due.

Extracted from runtime.py. Each detector is pure-ish — it reads slot state and
returns a list of risk events — with the single documented side effect of
recording which alerts have already fired on the slot (so the continuous-checks
loop doesn't re-spam every tick). EngineRuntime calls these from
``_continuous_checks_loop`` and publishes whatever they return.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .risk_events import OrderUnconfirmed, RedemptionTimeout

if TYPE_CHECKING:  # avoid a runtime import cycle (AccountSlot lives in runtime)
    from .runtime import AccountSlot


# PM unconfirmed-order watchdog: OrderUnconfirmed fires once a live PM order
# has sat in flight (status=open/pending/partially_filled) past this threshold
# without a status change. 30s gives PM CLOB plenty of room under heavy chain
# load while still surfacing stuck orders before the next scan tick would
# top up on top of stale state.
PM_UNCONFIRMED_THRESHOLD_S: float = 30.0

# PM redemption watchdog: RedemptionTimeout fires this long after the
# settlement Exit if the operator hasn't redeemed yet (we don't watch USDC
# on-chain). 6h is a generous window that catches genuinely-forgotten
# settlements without nagging through the normal redemption flow.
PM_REDEMPTION_TIMEOUT_S: float = 6 * 3600.0


def _pm_check_unconfirmed_orders(
    slot: "AccountSlot", now_ns: int, *,
    threshold_s: float = PM_UNCONFIRMED_THRESHOLD_S,
) -> list[OrderUnconfirmed]:
    """Pure detector: scan slot.dal.live_orders() and return one
    OrderUnconfirmed for each open order older than threshold_s that hasn't
    already alerted. Mutates `slot.pm_alerted_unconfirmed_cloids` to record
    new alerts and to evict cloids no longer live.
    """
    live = slot.dal.live_orders()
    live_cloids: set[str] = {o.cloid for o in live}
    # Garbage-collect alerted set so a re-placed order with the same cloid
    # would re-fire after its next stall. Without this, the set grows
    # unbounded over the process lifetime.
    slot.pm_alerted_unconfirmed_cloids &= live_cloids
    out: list[OrderUnconfirmed] = []
    for o in live:
        if o.status != "open":
            continue
        age_s = (now_ns - o.last_update_ts_ns) / 1e9
        if age_s < threshold_s:
            continue
        if o.cloid in slot.pm_alerted_unconfirmed_cloids:
            continue
        out.append(OrderUnconfirmed(
            ts_ns=now_ns, account_alias=slot.alias,
            cloid=o.cloid, symbol=o.symbol, side=o.side,  # type: ignore[arg-type]
            size=o.size, limit_price=o.price, age_seconds=age_s,
            venue_oid=o.venue_oid or "",
        ))
        slot.pm_alerted_unconfirmed_cloids.add(o.cloid)
    return out


def _pm_check_redemption_timeouts(
    slot: "AccountSlot", now_ns: int, *,
    threshold_s: float = PM_REDEMPTION_TIMEOUT_S,
) -> list[RedemptionTimeout]:
    """Pure detector: walk slot.pm_settlements and return one
    RedemptionTimeout per PM settlement older than threshold_s that hasn't
    alerted. Mutates `slot.pm_alerted_redemption_qidxs`.
    """
    out: list[RedemptionTimeout] = []
    for qidx, (settled_ts_ns, symbol, qty, realized_pnl) in slot.pm_settlements.items():
        if qidx in slot.pm_alerted_redemption_qidxs:
            continue
        age_s = (now_ns - settled_ts_ns) / 1e9
        if age_s < threshold_s:
            continue
        # No on-chain check (see RedemptionTimeout docstring): for a winning
        # leg the operator should see qty USDC arrive; for a loser, zero.
        # Winner heuristic: realized_pnl > 0 (PM binary payouts make this
        # equivalent under positive entry prices, which is always the case).
        expected_payout = qty if realized_pnl > 0 else 0.0
        out.append(RedemptionTimeout(
            ts_ns=now_ns, account_alias=slot.alias,
            question_idx=qidx, symbol=symbol, qty=qty,
            settled_ts_ns=settled_ts_ns, age_seconds=age_s,
            expected_payout_usd=expected_payout,
        ))
        slot.pm_alerted_redemption_qidxs.add(qidx)
    return out
