"""PMClient resilience: transient network/5xx blips on the live read AND write
paths must RETRY (so a momentary data-api hiccup is not silently treated as an
empty book / rejected order), while genuine business rejections fail fast with
no retry. Mirrors hl_client's read/write retry discipline.
"""
from __future__ import annotations

import requests

from hlanalysis.engine.exec_types import PlaceRequest
from hlanalysis.engine.pm_client import PMClient


def _client() -> PMClient:
    return PMClient(
        paper_mode=False, clob_host="x", chain_id=137,
        private_key="0x0", clob_api_key="k", clob_api_secret="s",
        clob_api_passphrase="p",
    )


class _FlakyReadClob:
    """get_open_orders raises a transient error `fail_times` times, then
    returns one canned row. Counts calls so the test can assert the retry."""

    def __init__(self, *, exc: Exception, fail_times: int) -> None:
        self._exc = exc
        self._fail_times = fail_times
        self.calls = 0

    def get_open_orders(self):
        self.calls += 1
        if self.calls <= self._fail_times:
            raise self._exc
        return [{
            "id": "oid-1", "asset_id": "tok-1", "side": "BUY",
            "price": "0.40", "original_size": "10", "size_matched": "0",
            "created_at": 1,
        }]


def test_live_read_retries_transient_then_succeeds():
    fake = _FlakyReadClob(
        exc=requests.exceptions.ConnectionError("data-api blip"), fail_times=2,
    )
    c = _client()
    c._sdk = fake
    rows = c.open_orders()
    assert fake.calls == 3  # 2 failures + 1 success
    assert len(rows) == 1
    assert rows[0].venue_oid == "oid-1"


def test_live_read_does_not_retry_business_error():
    # A non-transient exception (e.g. a malformed request the venue rejects)
    # must NOT retry — it fails fast and soft-fails to [].
    fake = _FlakyReadClob(exc=ValueError("invalid token id"), fail_times=99)
    c = _client()
    c._sdk = fake
    rows = c.open_orders()
    assert fake.calls == 1  # no retry on a business error
    assert rows == []


class _FlakyWriteClob:
    """create_and_post_market_order raises a transient error `fail_times`
    times, then returns a successful fill ack."""

    def __init__(self, *, exc: Exception, fail_times: int) -> None:
        self._exc = exc
        self._fail_times = fail_times
        self.calls = 0

    def create_and_post_market_order(self, *, order_args, options, order_type,
                                     defer_exec=False):
        self.calls += 1
        if self.calls <= self._fail_times:
            raise self._exc
        return {
            "success": True, "orderID": "0xfakeid", "status": "matched",
            "makingAmount": "4.0", "takingAmount": "10.0",
        }


def _buy_req() -> PlaceRequest:
    return PlaceRequest(
        cloid="hla-v31_pm-abc", symbol="tok-1", side="buy",
        size=10.0, price=0.40, reduce_only=False, time_in_force="ioc",
    )


def test_live_write_retries_transient_then_succeeds():
    fake = _FlakyWriteClob(
        exc=requests.exceptions.Timeout("clob 504"), fail_times=2,
    )
    c = _client()
    c._sdk = fake
    ack = c.place(_buy_req())
    assert fake.calls == 3
    assert ack.status == "filled"
    assert ack.fill_size == 10.0


def test_live_write_does_not_retry_business_error():
    # A non-transient SDK exception (genuine rejection) must fail fast: one
    # call, rejected ack.
    fake = _FlakyWriteClob(exc=ValueError("not enough balance"), fail_times=99)
    c = _client()
    c._sdk = fake
    ack = c.place(_buy_req())
    assert fake.calls == 1
    assert ack.status == "rejected"
