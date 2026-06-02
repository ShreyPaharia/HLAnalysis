"""Live-mode PMClient tests. All paths inject a `_FakeClob` via
`client._sdk` — no network and no real py-clob-client-v2 SDK calls.
"""
from __future__ import annotations

from hlanalysis.engine.exec_types import PlaceRequest
from hlanalysis.engine.pm_client import PMClient


class _FakeClob:
    """Stand-in for py_clob_client_v2.ClobClient — captures inputs and
    returns canned responses shaped like the real SDK."""

    def __init__(
        self,
        *,
        place_resp: dict | None = None,
        cancel_resp: dict | None = None,
        open_orders: list[dict] | None = None,
        balance_allowance: dict | None = None,
        trades: list[dict] | None = None,
    ) -> None:
        self.placed: list[dict] = []
        self.market_placed: list[dict] = []
        self.canceled: list[str] = []
        self._place_resp = place_resp
        self._cancel_resp = cancel_resp or {"canceled": []}
        self._open_orders = open_orders or []
        self._balance_allowance = balance_allowance or {"balance": "0"}
        self._trades = trades or []

    def create_and_post_order(self, *, order_args, options, order_type):
        self.placed.append({
            "token_id": order_args.token_id,
            "price": order_args.price,
            "size": order_args.size,
            "side": str(order_args.side),
            "order_type": str(order_type),
        })
        if self._place_resp is not None:
            return self._place_resp
        # PM CLOB taker-perspective convention: BUY → making=USDC paid,
        # taking=shares received. Confirmed empirically against a live
        # $19.50 fill 2026-05-26.
        is_buy = str(order_args.side).endswith("BUY") or str(order_args.side) == "0"
        usdc = order_args.size * order_args.price
        shares = order_args.size
        return {
            "success": True,
            "orderID": "0xfakeid",
            "status": "matched",
            "makingAmount": str(usdc if is_buy else shares),
            "takingAmount": str(shares if is_buy else usdc),
        }

    def create_and_post_market_order(
        self, *, order_args, options, order_type, defer_exec=False,
    ):
        self.market_placed.append({
            "token_id": order_args.token_id,
            "amount": order_args.amount,
            "price": order_args.price,
            "side": str(order_args.side),
            "order_type": str(order_type),
        })
        if self._place_resp is not None:
            return self._place_resp
        # Market BUY: `amount` is USDC paid → making=USDC, taking=shares.
        # Market SELL: `amount` is shares sold → making=shares, taking=USDC.
        is_buy = str(order_args.side).endswith("BUY") or str(order_args.side) == "0"
        if is_buy:
            making = order_args.amount
            taking = order_args.amount / order_args.price if order_args.price else 0.0
        else:
            making = order_args.amount
            taking = order_args.amount * order_args.price
        return {
            "success": True,
            "orderID": "0xfakeid",
            "status": "matched",
            "makingAmount": str(making),
            "takingAmount": str(taking),
        }

    def cancel_order(self, payload):
        oid = getattr(payload, "orderID", None) or payload.get("orderID")  # type: ignore[attr-defined]
        self.canceled.append(oid)
        return self._cancel_resp

    def get_open_orders(self):
        return self._open_orders

    def get_balance_allowance(self, params=None):
        return self._balance_allowance

    def get_trades(self, params=None):
        return self._trades


def _client() -> PMClient:
    return PMClient(
        paper_mode=False, clob_host="x", chain_id=137,
        private_key="0x0", clob_api_key="k", clob_api_secret="s",
        clob_api_passphrase="p",
    )


def test_live_place_translates_request_to_FAK_market_order():
    # IOC orders are marketable → routed through PM's market-order endpoint
    # (create_and_post_market_order), not the limit path. For a BUY the market
    # `amount` is the USDC to spend (price·size).
    fake = _FakeClob()
    c = _client()
    c._sdk = fake  # inject fake
    ack = c.place(PlaceRequest(
        cloid="hla-v31_pm-1", symbol="71321...992563",
        side="buy", size=100, price=0.92,
        reduce_only=False, time_in_force="ioc",
    ))
    assert not fake.placed, "ioc must not use the limit (create_and_post_order) path"
    assert fake.market_placed, "create_and_post_market_order was not invoked"
    placed = fake.market_placed[0]
    assert placed["token_id"] == "71321...992563"
    assert placed["price"] == 0.92
    assert placed["amount"] == 92.0  # USDC = 0.92 * 100
    assert placed["order_type"].endswith("FAK")
    assert "BUY" in placed["side"] or placed["side"].endswith("0")
    assert ack.status == "filled"
    assert ack.fill_size == 100
    assert ack.fill_price == 0.92
    # cloid → orderID mapping is populated for subsequent cancel.
    assert c._cloid_to_oid["hla-v31_pm-1"] == "0xfakeid"


def test_live_place_ioc_buy_rounds_usdc_amount_to_2_decimals():
    # Regression: PM rejects market-buy orders whose USDC (maker) amount has
    # >2 decimals ("invalid amounts ... max accuracy of 2 decimals"). With the
    # old limit path, 0.94 * 53.19 = 49.9986 was sent at 4 decimals and PM
    # rejected every order. Must round the USDC amount down to cents.
    fake = _FakeClob()
    c = _client()
    c._sdk = fake
    ack = c.place(PlaceRequest(
        cloid="m1", symbol="tok", side="buy", size=53.19, price=0.94,
        reduce_only=False, time_in_force="ioc",
    ))
    assert fake.market_placed, "create_and_post_market_order was not invoked"
    amount = fake.market_placed[0]["amount"]
    assert amount == 49.99  # round_down(0.94 * 53.19, 2)
    # USDC maker amount must carry at most 2 decimal places.
    assert round(amount, 2) == amount
    assert ack.status == "filled"


def test_live_place_ioc_sell_amount_is_share_count():
    # Market SELL: `amount` is the share count (maker = shares), not USDC.
    fake = _FakeClob()
    c = _client()
    c._sdk = fake
    c.place(PlaceRequest(
        cloid="s1", symbol="tok", side="sell", size=40.0, price=0.94,
        reduce_only=False, time_in_force="ioc",
    ))
    assert fake.market_placed[0]["amount"] == 40.0
    assert fake.market_placed[0]["order_type"].endswith("FAK")


def test_live_place_gtc_maps_to_GTC_order_type():
    fake = _FakeClob()
    c = _client()
    c._sdk = fake
    c.place(PlaceRequest(
        cloid="g1", symbol="tok", side="sell", size=10, price=0.4,
        reduce_only=False, time_in_force="gtc",
    ))
    assert fake.placed[0]["order_type"].endswith("GTC")


def test_live_place_failure_response_returns_rejected_ack():
    fake = _FakeClob(place_resp={
        "success": False, "errorMsg": "below minimum size", "orderID": "",
    })
    c = _client()
    c._sdk = fake
    ack = c.place(PlaceRequest(
        cloid="r1", symbol="tok", side="buy", size=1, price=0.5,
        reduce_only=False, time_in_force="ioc",
    ))
    assert ack.status == "rejected"
    assert "below minimum size" in (ack.error or "")
    # no mapping recorded on failure
    assert "r1" not in c._cloid_to_oid


def test_live_cancel_resolves_cloid_via_local_map():
    fake = _FakeClob(cancel_resp={"canceled": ["0xfakeid"]})
    c = _client()
    c._sdk = fake
    # Place first to populate the cloid→oid map.
    c.place(PlaceRequest(
        cloid="c1", symbol="tok", side="buy", size=10, price=0.5,
        reduce_only=False, time_in_force="ioc",
    ))
    ok = c.cancel(cloid="c1", symbol="tok")
    assert ok is True
    assert fake.canceled == ["0xfakeid"]
    # cloid is dropped from the map post-cancel.
    assert "c1" not in c._cloid_to_oid


def test_live_cancel_unknown_cloid_returns_false():
    fake = _FakeClob()
    c = _client()
    c._sdk = fake
    assert c.cancel(cloid="never-placed", symbol="tok") is False
    assert fake.canceled == []


def test_live_open_orders_maps_response_to_OpenOrderRow():
    fake = _FakeClob(open_orders=[
        {
            "id": "0xabc", "asset_id": "tok-1", "side": "BUY",
            "price": "0.55", "original_size": "100", "size_matched": "30",
            "created_at": 1_700_000_000,
        },
        {
            "id": "0xdef", "asset_id": "tok-2", "side": "SELL",
            "price": "0.30", "original_size": "50", "size_matched": "0",
            "created_at": 1_700_000_100,
        },
    ])
    c = _client()
    c._sdk = fake
    # Seed a known mapping for one of the orders to validate reverse-lookup.
    c._cloid_to_oid["hla-1"] = "0xabc"
    rows = c.open_orders()
    assert len(rows) == 2
    by_oid = {r.venue_oid: r for r in rows}
    a = by_oid["0xabc"]
    assert a.cloid == "hla-1"
    assert a.symbol == "tok-1"
    assert a.side == "buy"
    assert a.price == 0.55
    assert a.size == 70.0  # remaining = 100 - 30
    assert a.placed_ts_ns == 1_700_000_000 * 1_000_000_000
    b = by_oid["0xdef"]
    assert b.side == "sell"
    assert b.size == 50.0
    # No cloid mapping → falls back to the venue oid.
    assert b.cloid == "0xdef"


def test_live_clearinghouse_state_reflects_usdc_balance():
    fake = _FakeClob(balance_allowance={"balance": "1500000000"})  # 1500 USDC
    c = _client()
    c._sdk = fake
    state = c.clearinghouse_state()
    assert state.positions == ()
    assert state.account_value_usd == 1500.0


def _client_with_funder() -> PMClient:
    return PMClient(
        paper_mode=False, clob_host="x", chain_id=137, private_key="0x0",
        clob_api_key="k", clob_api_secret="s", clob_api_passphrase="p",
        funder_address="0xFUNDER",
    )


def test_live_clearinghouse_state_reports_data_api_positions():
    # The reconciler relies on this to adopt/keep PM positions as venue truth.
    fake = _FakeClob(balance_allowance={"balance": "1000000"})  # 1 USDC
    c = _client_with_funder()
    c._sdk = fake
    captured = {}

    def fake_get(url):
        captured["url"] = url
        return [
            {"asset": "tok-down", "size": 51.536, "avgPrice": 0.9699, "cashPnl": -0.77},
            {"asset": "tok-zero", "size": 0, "avgPrice": 0.5},  # filtered: flat
        ]

    c._data_api_get = fake_get
    state = c.clearinghouse_state()
    assert state.positions_known is True
    assert "0xFUNDER" in captured["url"]
    syms = {p.symbol: p for p in state.positions}
    assert "tok-zero" not in syms  # zero-size dropped
    assert syms["tok-down"].qty == 51.536
    assert syms["tok-down"].avg_entry == 0.9699


def test_live_clearinghouse_state_positions_unknown_on_data_api_failure():
    # A fetch failure must report positions_known=False (NOT an empty set) so
    # the reconciler skips position reconciliation instead of vanish-deleting
    # every live position.
    fake = _FakeClob(balance_allowance={"balance": "1000000"})
    c = _client_with_funder()
    c._sdk = fake

    def boom(url):
        raise RuntimeError("data-api 503")

    c._data_api_get = boom
    state = c.clearinghouse_state()
    assert state.positions == ()
    assert state.positions_known is False
    assert state.account_value_usd == 1.0  # balance still read


def test_live_clearinghouse_state_positions_unknown_without_funder():
    fake = _FakeClob(balance_allowance={"balance": "0"})
    c = _client()  # no funder configured
    c._sdk = fake
    state = c.clearinghouse_state()
    assert state.positions == ()
    assert state.positions_known is False


def test_live_user_fills_maps_trades_to_UserFillRow():
    fake = _FakeClob(trades=[
        {
            "id": "t-1", "taker_order_id": "0xabc", "asset_id": "tok-1",
            "side": "BUY", "price": "0.55", "size": "100",
            "fee_rate_bps": "10", "match_time": 1_700_000_050,
        },
        {
            "id": "t-2", "taker_order_id": "0xdef", "asset_id": "tok-2",
            "side": "SELL", "price": "0.30", "size": "50",
            "fee_rate_bps": "0", "match_time": 1_700_000_200,
        },
    ])
    c = _client()
    c._sdk = fake
    c._cloid_to_oid["hla-1"] = "0xabc"
    fills = c.user_fills(since_ts_ns=0)
    assert len(fills) == 2
    by_id = {f.fill_id: f for f in fills}
    a = by_id["t-1"]
    assert a.cloid == "hla-1"
    assert a.symbol == "tok-1"
    assert a.side == "buy"
    assert a.price == 0.55
    assert a.size == 100.0
    # 10 bps on $55 notional = $0.055
    assert abs(a.fee - (10 / 10_000) * 0.55 * 100) < 1e-9
    assert a.ts_ns == 1_700_000_050 * 1_000_000_000
    b = by_id["t-2"]
    assert b.side == "sell"
    assert b.cloid == "0xdef"  # no mapping → falls back to taker_order_id


def test_live_user_fills_filters_by_since_ts_ns():
    fake = _FakeClob(trades=[
        {
            "id": "old", "taker_order_id": "x", "asset_id": "tok",
            "side": "BUY", "price": "0.5", "size": "10",
            "fee_rate_bps": "0", "match_time": 1_000,
        },
        {
            "id": "new", "taker_order_id": "y", "asset_id": "tok",
            "side": "BUY", "price": "0.5", "size": "10",
            "fee_rate_bps": "0", "match_time": 2_000,
        },
    ])
    c = _client()
    c._sdk = fake
    # since_ts_ns=1500s in ns → only "new" survives.
    fills = c.user_fills(since_ts_ns=1500 * 1_000_000_000)
    assert [f.fill_id for f in fills] == ["new"]
