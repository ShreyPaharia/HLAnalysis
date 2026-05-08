from __future__ import annotations

import pytest

from hlanalysis.engine.hl_client import (
    HLClient, OrderAck, PlaceRequest, RestError,
)


@pytest.fixture
def paper_client() -> HLClient:
    return HLClient(
        account_address="0xtest",
        api_secret_key="0xfake",
        base_url="https://api.hyperliquid.xyz",
        paper_mode=True,
    )


def test_paper_place_returns_synthesized_ack(paper_client):
    req = PlaceRequest(
        cloid="hla-1", symbol="@30", side="buy", size=10.0, price=0.95,
        reduce_only=False, time_in_force="ioc",
    )
    ack = paper_client.place(req)
    assert isinstance(ack, OrderAck)
    assert ack.cloid == "hla-1"
    assert ack.status == "filled"  # paper IOC at-or-better assumes immediate fill
    assert ack.venue_oid.startswith("paper-")


def test_paper_place_is_idempotent_per_cloid(paper_client):
    req = PlaceRequest(
        cloid="hla-2", symbol="@30", side="buy", size=10.0, price=0.95,
        reduce_only=False, time_in_force="ioc",
    )
    a = paper_client.place(req)
    b = paper_client.place(req)
    assert a.venue_oid == b.venue_oid


def test_paper_open_orders_returns_what_we_placed_and_did_not_fill(paper_client):
    # Force a non-marketable IOC: paper mode treats price <= 0.0 as "no fill"
    req = PlaceRequest(
        cloid="hla-3", symbol="@30", side="buy", size=10.0, price=0.0,
        reduce_only=False, time_in_force="ioc",
    )
    ack = paper_client.place(req)
    assert ack.status == "rejected"
    assert paper_client.open_orders() == []


def test_paper_clearinghouse_state_tracks_filled_orders(paper_client):
    paper_client.place(PlaceRequest(
        cloid="hla-4", symbol="@30", side="buy", size=10.0, price=0.95,
        reduce_only=False, time_in_force="ioc",
    ))
    state = paper_client.clearinghouse_state()
    assert any(p.symbol == "@30" and p.qty == 10.0 for p in state.positions)


def test_live_mode_requires_sdk(monkeypatch):
    # Live mode is not exercised in this test file; we only assert that
    # constructing a non-paper client doesn't crash and that place() raises if
    # the SDK is unreachable.
    c = HLClient(
        account_address="0xtest", api_secret_key="0xfake",
        base_url="https://api.hyperliquid.xyz", paper_mode=False,
    )

    # Patch the underlying sdk Exchange with one that always raises
    class _Boom:
        def order(self, *a, **kw):
            raise ConnectionError("network")
    c._exchange = _Boom()  # type: ignore[attr-defined]
    with pytest.raises(RestError):
        c.place(PlaceRequest(
            cloid="hla-5", symbol="@30", side="buy", size=1.0, price=0.5,
            reduce_only=False, time_in_force="ioc",
        ))
