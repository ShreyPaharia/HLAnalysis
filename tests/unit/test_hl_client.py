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

    def __init__(self, *, perp_state: dict, spot_state: dict, fills: list[dict] | None = None) -> None:
        self._perp = perp_state
        self._spot = spot_state
        self._fills = fills or []

    def user_state(self, _addr):  # noqa: D401 - mimics SDK shape
        return self._perp

    def spot_user_state(self, _addr):
        return self._spot

    def user_fills(self, _addr):
        return self._fills


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


def test_realized_pnl_since_sums_closedpnl_minus_fee():
    """The daily-loss gate now sources PnL from HL user_fills (closedPnl - fee)
    instead of the local DB. This is the contract: any fill at-or-after the
    cutoff contributes, anything older is excluded."""
    # closedPnl shape mirrors HL's spec: 0 on opens, signed on reduces.
    perp = {"assetPositions": [], "marginSummary": {"accountValue": "0.0"}}
    spot = {"balances": []}
    fills = [
        # Older than cutoff — excluded.
        {"time": 999, "hash": "f0", "cloid": "0x0", "coin": "#601",
         "side": "B", "px": "0.99", "sz": "200", "fee": "0.10",
         "closedPnl": "0"},
        # Open at midnight+1ms — closed_pnl=0, fee=0.07 → contributes -0.07.
        {"time": 1001, "hash": "f1", "cloid": "0x1", "coin": "#601",
         "side": "B", "px": "0.99", "sz": "200", "fee": "0.07",
         "closedPnl": "0"},
        # Reduce later → closedPnl=-10.0, fee=0.02 → contributes -10.02.
        {"time": 2000, "hash": "f2", "cloid": "0x2", "coin": "#601",
         "side": "A", "px": "0.94", "sz": "200", "fee": "0.02",
         "closedPnl": "-10.0"},
    ]
    c = HLClient(account_address="0xtest", api_secret_key="0xfake",
                 base_url="https://api.hyperliquid.xyz", paper_mode=False,
                 pnl_cache_ttl_s=0.0)
    c._info = _FakeInfo(perp_state=perp, spot_state=spot, fills=fills)  # type: ignore[assignment]
    cutoff_ns = 1000 * 1_000_000  # 1000 ms in ns
    pnl = c.realized_pnl_since(cutoff_ns)
    assert pnl == pytest.approx(-10.09)


def test_realized_pnl_since_is_cached_to_bound_rest_calls():
    """The scanner runs at 1Hz; we cannot hit HL every tick. realized_pnl_since
    must reuse a recent result while the TTL is live."""
    perp = {"assetPositions": [], "marginSummary": {"accountValue": "0.0"}}
    spot = {"balances": []}
    fills = [
        {"time": 1000, "hash": "f1", "cloid": "0x1", "coin": "#601",
         "side": "B", "px": "0.99", "sz": "200", "fee": "0.10",
         "closedPnl": "0"},
    ]
    c = HLClient(account_address="0xtest", api_secret_key="0xfake",
                 base_url="https://api.hyperliquid.xyz", paper_mode=False,
                 pnl_cache_ttl_s=60.0)
    fake = _FakeInfo(perp_state=perp, spot_state=spot, fills=fills)
    c._info = fake  # type: ignore[assignment]
    # Count REST calls by wrapping user_fills.
    call_count = {"n": 0}
    orig = fake.user_fills
    def counted(addr):  # noqa: ANN001
        call_count["n"] += 1
        return orig(addr)
    fake.user_fills = counted  # type: ignore[assignment]
    for _ in range(5):
        c.realized_pnl_since(0)
    assert call_count["n"] == 1  # cache hit on the next four


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
