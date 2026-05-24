from __future__ import annotations

from hlanalysis.engine.exec_client import ExecutionClient
from hlanalysis.engine.hl_client import HLClient


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
