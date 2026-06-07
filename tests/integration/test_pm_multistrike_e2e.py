"""End-to-end: a Gamma multi-strike event → grouped priceBucket question →
strategy leg geometry → per-leg settlement PnL.

Exercises the live-enablement seam without a websocket: the shared adapter
parser groups the event, MarketState rebuilds the flat leg_symbols, the
above_ladder region map yields the right winning intervals per leg, and the
settlement-PnL path books the held leg correctly even when other legs' settle
events arrive first.
"""
from __future__ import annotations

import json

from hlanalysis.adapters.polymarket_normalize import (
    parse_gamma_event_to_bucket_question_meta,
)
from hlanalysis.engine.market_state import MarketState
from hlanalysis.strategy.regions import winning_region
from hlanalysis.strategy.render import settlement_pnl_usd


def _event():
    def mk(s, y, n, c):
        return {
            "conditionId": c,
            "groupItemTitle": f"${s:,.0f}",
            "clobTokenIds": json.dumps([y, n]),
            "outcomePrices": json.dumps(["0", "0"]),
            "endDate": "2026-06-12T16:00:00Z",
        }
    return {
        "slug": "btc-ms-x",
        "startDate": "2026-06-08T16:00:00Z",
        "endDate": "2026-06-12T16:00:00Z",
        # out of order on purpose to prove strike-sort
        "markets": [mk(90000, "y90", "n90", "c90"), mk(80000, "y80", "n80", "c80")],
    }


def test_bucket_question_legs_and_regions_consistent():
    qm = parse_gamma_event_to_bucket_question_meta(
        _event(), series_slug="btc-multi-strikes-weekly",
        local_recv_ts=1, underlying="BTC",
    )
    ms = MarketState()
    ms.apply(qm)
    q = ms.question(qm.question_idx)
    assert q.klass == "priceBucket"
    assert q.leg_symbols == ("y80", "n80", "y90", "n90")
    # above_ladder: y80 wins above 80000; n90 wins at-or-below 90000
    assert winning_region(q, "y80") == (80000.0, None)
    assert winning_region(q, "n90") == (None, 90000.0)


def test_held_leg_settlement_pnl_robust_to_arrival_order():
    qm = parse_gamma_event_to_bucket_question_meta(
        _event(), series_slug="btc-multi-strikes-weekly",
        local_recv_ts=1, underlying="BTC",
    )
    ms = MarketState()
    ms.apply(qm)
    qidx = qm.question_idx
    # Final spot ~85000 → above-80000 YES (y80) wins, above-90000 NO (n90) wins.
    # Settlement events arrive n90 FIRST (a leg we don't hold), then y80.
    from hlanalysis.events import SettlementEvent, ProductType, Mechanism

    def _settle(sym):
        return SettlementEvent(
            venue="polymarket", product_type=ProductType.PREDICTION_BINARY,
            mechanism=Mechanism.CLOB, symbol=sym, exchange_ts=2, local_recv_ts=2,
            settled_side_idx=0, settle_price=1.0, settle_ts=2,
            keys=[], values=[],
        )
    ms.apply(_settle("n90"))
    ms.apply(_settle("y80"))
    q = ms.question(qidx)
    # We hold y80 (a winner) entered at 0.90 for 10 shares → +$1.00.
    pnl = settlement_pnl_usd(q, "y80", qty=10.0, avg_entry=0.90)
    assert abs(pnl - 10.0 * (1.0 - 0.90)) < 1e-9
    # A held losing leg (y90, did not win) booked as loss.
    pnl_loss = settlement_pnl_usd(q, "y90", qty=10.0, avg_entry=0.40)
    assert abs(pnl_loss - 10.0 * (0.0 - 0.40)) < 1e-9
