# tests/unit/test_sim_question_builder.py
from __future__ import annotations

from hlanalysis.sim.data.schemas import PMMarket
from hlanalysis.sim.question_builder import build_question_view


def _market() -> PMMarket:
    return PMMarket(
        condition_id="0xabc",
        yes_token_id="11",
        no_token_id="22",
        start_ts_ns=0,
        end_ts_ns=86_400_000_000_000,
        resolved_outcome="yes",
        total_volume_usd=10_000.0,
        n_trades=100,
    )


def test_question_view_uses_day_open_as_strike():
    qv = build_question_view(_market(), day_open_btc=100_000.0, now_ns=3_600_000_000_000)
    assert qv.strike == 100_000.0
    assert qv.expiry_ns == 86_400_000_000_000
    assert qv.yes_symbol == "11"
    assert qv.no_symbol == "22"
    assert qv.period == "24h"
    assert qv.underlying == "BTC"
    assert not qv.settled


def test_question_view_settles_after_expiry():
    qv = build_question_view(_market(), day_open_btc=100_000.0, now_ns=86_400_000_000_001)
    assert qv.settled
    assert qv.settled_side == "yes"
