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


class _FakeInfo:
    """Stub for hyperliquid.info.Info used in clearinghouse_state tests."""

    def __init__(self, *, perp_state: dict, spot_state: dict) -> None:
        self._perp = perp_state
        self._spot = spot_state

    def user_state(self, _addr):  # noqa: D401 - mimics SDK shape
        return self._perp

    def spot_user_state(self, _addr):
        return self._spot


def _live_client_with(perp_state: dict, spot_state: dict) -> HLClient:
    c = HLClient(
        account_address="0xtest", api_secret_key="0xfake",
        base_url="https://api.hyperliquid.xyz", paper_mode=False,
    )
    c._info = _FakeInfo(perp_state=perp_state, spot_state=spot_state)  # type: ignore[assignment]
    return c


def test_clearinghouse_state_merges_hip4_spot_balances():
    # Reconcile depends on this: HIP-4 outcome shares live in
    # spotClearinghouseState (coin "+N"), NOT in the perp assetPositions list.
    # If we forget to merge them, the reconciler thinks every HIP-4 position
    # vanished from venue, deletes the local row, and the strategy re-enters.
    perp = {
        "assetPositions": [],
        "marginSummary": {"accountValue": "0.0"},
    }
    spot = {
        "balances": [
            {"coin": "USDC", "total": "270.37", "entryNtl": "0"},
            {"coin": "USDH", "total": "0.86", "entryNtl": "0.86"},
            {"coin": "HYPE", "total": "0.02", "entryNtl": "1.07"},
            {"coin": "+551", "total": "500.0", "entryNtl": "494.236736"},
            {"coin": "+580", "total": "570.0", "entryNtl": "539.45930939"},
            {"coin": "+581", "total": "0.0", "entryNtl": "0.0"},   # zero qty
            {"coin": "o468", "total": "0.0", "entryNtl": "0.0"},    # non-HIP4
        ],
    }
    c = _live_client_with(perp, spot)
    state = c.clearinghouse_state()

    by_sym = {p.symbol: p for p in state.positions}
    assert "#551" in by_sym, "missing #551 HIP-4 position from spot merge"
    assert "#580" in by_sym, "missing #580 HIP-4 position from spot merge"
    assert by_sym["#551"].qty == 500.0
    assert by_sym["#580"].qty == 570.0
    # avg_entry = entryNtl / qty
    assert by_sym["#551"].avg_entry == pytest.approx(494.236736 / 500.0)
    assert by_sym["#580"].avg_entry == pytest.approx(539.45930939 / 570.0)
    # Non-HIP4 spot coins must not be reported as positions.
    assert "USDC" not in by_sym
    assert "USDH" not in by_sym
    assert "o468" not in by_sym
    # Zero qty HIP-4 entries skipped.
    assert "#581" not in by_sym


def test_clearinghouse_state_empty_spot_balances_ok():
    perp = {
        "assetPositions": [
            {"position": {"coin": "BTC", "szi": "0.1", "entryPx": "100000",
                          "unrealizedPnl": "0"}},
        ],
        "marginSummary": {"accountValue": "10000.0"},
    }
    spot = {"balances": []}
    c = _live_client_with(perp, spot)
    state = c.clearinghouse_state()
    syms = [p.symbol for p in state.positions]
    assert syms == ["BTC"]
    assert state.account_value_usd == 10000.0
