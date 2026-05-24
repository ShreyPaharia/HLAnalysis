from __future__ import annotations

import pytest

from hlanalysis.engine.exec_types import PlaceRequest
from hlanalysis.engine.pm_client import PMClient


@pytest.fixture
def paper() -> PMClient:
    return PMClient(paper_mode=True)


def test_paper_place_marketable_fills(paper):
    req = PlaceRequest(
        cloid="hla-v31_pm-1", symbol="71321...992563",
        side="buy", size=100.0, price=0.92,
        reduce_only=False, time_in_force="ioc",
    )
    ack = paper.place(req)
    assert ack.status == "filled"
    assert ack.fill_price == 0.92 and ack.fill_size == 100.0


def test_paper_place_rejects_nonpositive_price(paper):
    req = PlaceRequest(
        cloid="hla-v31_pm-2", symbol="t", side="buy", size=10, price=0.0,
        reduce_only=False, time_in_force="ioc",
    )
    ack = paper.place(req)
    assert ack.status == "rejected"
    assert ack.error == "non_marketable_price"


def test_paper_clearinghouse_state_reflects_fills(paper):
    paper.place(PlaceRequest(
        cloid="hla-v31_pm-3", symbol="tok", side="buy", size=100, price=0.9,
        reduce_only=False, time_in_force="ioc",
    ))
    state = paper.clearinghouse_state()
    pos = [p for p in state.positions if p.symbol == "tok"]
    assert pos and pos[0].qty == 100
    assert pos[0].avg_entry == 0.9


def test_paper_realized_pnl_zero_on_open(paper):
    paper.place(PlaceRequest(
        cloid="hla-v31_pm-4", symbol="tok", side="buy", size=50, price=0.9,
        reduce_only=False, time_in_force="ioc",
    ))
    assert paper.realized_pnl_since(0) == 0.0


def test_paper_place_idempotent_per_cloid(paper):
    req = PlaceRequest(
        cloid="hla-v31_pm-5", symbol="tok", side="buy", size=10, price=0.5,
        reduce_only=False, time_in_force="ioc",
    )
    a = paper.place(req)
    b = paper.place(req)
    assert a.venue_oid == b.venue_oid
    # Only one fill recorded, not two.
    assert len(paper.user_fills(since_ts_ns=0)) == 1


def test_paper_avg_up_on_add(paper):
    paper.place(PlaceRequest(
        cloid="c1", symbol="tok", side="buy", size=100, price=0.9,
        reduce_only=False, time_in_force="ioc",
    ))
    paper.place(PlaceRequest(
        cloid="c2", symbol="tok", side="buy", size=100, price=0.8,
        reduce_only=False, time_in_force="ioc",
    ))
    state = paper.clearinghouse_state()
    pos = next(p for p in state.positions if p.symbol == "tok")
    assert pos.qty == 200
    assert pos.avg_entry == pytest.approx((100 * 0.9 + 100 * 0.8) / 200)


def test_paper_position_removed_when_netted_to_zero(paper):
    paper.place(PlaceRequest(
        cloid="c1", symbol="tok", side="buy", size=100, price=0.9,
        reduce_only=False, time_in_force="ioc",
    ))
    paper.place(PlaceRequest(
        cloid="c2", symbol="tok", side="sell", size=100, price=0.95,
        reduce_only=False, time_in_force="ioc",
    ))
    state = paper.clearinghouse_state()
    assert all(p.symbol != "tok" for p in state.positions)


def test_paper_cancel_returns_false_when_no_open(paper):
    assert paper.cancel(cloid="never-placed", symbol="tok") is False


def test_live_mode_construct_does_not_crash():
    """paper_mode=False must construct without hitting the network — the
    actual live wiring lands in Phase 8 and raises NotImplementedError on
    first order I/O."""
    c = PMClient(
        paper_mode=False,
        clob_host="https://clob.polymarket.com",
        private_key="0xdead",
        clob_api_key="k", clob_api_secret="s", clob_api_passphrase="p",
    )
    with pytest.raises(NotImplementedError):
        c.place(PlaceRequest(
            cloid="x", symbol="t", side="buy", size=1, price=0.5,
            reduce_only=False, time_in_force="ioc",
        ))
