from __future__ import annotations

from hlanalysis.engine.exec_client import ExecutionClient
from hlanalysis.engine.hl_client import HLClient
from hlanalysis.engine.pm_client import PMClient


def test_hl_client_satisfies_execution_client_protocol():
    """HLClient must structurally conform to ExecutionClient.

    Protocols use runtime-checkable duck typing; this test pins the surface
    so any future HLClient/PMClient change that drops a method fails fast.
    """
    paper = HLClient(
        account_address="0xtest", api_secret_key="0xfake",
        base_url="https://api.hyperliquid.xyz", paper_mode=True,
    )
    assert isinstance(paper, ExecutionClient)


def test_pm_client_paper_satisfies_execution_client_protocol():
    """PMClient (paper mode) must also conform — engine wiring depends on
    both venue clients being interchangeable behind the Protocol."""
    paper = PMClient(paper_mode=True)
    assert isinstance(paper, ExecutionClient)


def test_pm_client_live_constructs_without_sdk_import():
    """Paper_mode=False MUST construct successfully — the live SDK
    (py-clob-client-v2) lands in Phase 8 and must not be imported here,
    so an unconfigured env still boots the engine. Order I/O raises
    NotImplementedError only when actually invoked."""
    import pytest

    live = PMClient(
        paper_mode=False,
        clob_host="https://clob.polymarket.com",
        chain_id=137,
        private_key="0xstub",
        clob_api_key="stub",
        clob_api_secret="stub",
        clob_api_passphrase="stub",
    )
    assert isinstance(live, ExecutionClient)
    # Read-only calls return empty/zero so the engine's restart-drift and
    # PnL paths don't crash before Phase 8.
    assert live.open_orders() == []
    assert live.user_fills() == []
    assert live.realized_pnl_since(0) == 0.0
    # place / cancel raise — surfaces a misconfigured slot at first order.
    from hlanalysis.engine.exec_types import PlaceRequest
    req = PlaceRequest(
        cloid="x", symbol="tok", side="buy", size=1.0, price=0.9,
        reduce_only=False, time_in_force="ioc",
    )
    with pytest.raises(NotImplementedError):
        live.place(req)
    with pytest.raises(NotImplementedError):
        live.cancel(cloid="x", symbol="tok")
