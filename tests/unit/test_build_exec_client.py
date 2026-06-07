import pytest

from hlanalysis.engine.config_builders import build_exec_client
from hlanalysis.engine.config import HyperliquidAccount, PolymarketAccount
from hlanalysis.engine.hl_client import HLClient
from hlanalysis.engine.pm_client import PMClient


def test_build_exec_client_hl():
    acct = HyperliquidAccount(
        account_address="0xabc",
        api_secret_key="0xkey",
        base_url="https://api.hyperliquid.xyz",
    )
    client = build_exec_client("v1", acct, paper_mode=True)
    assert isinstance(client, HLClient)


def test_build_exec_client_pm():
    acct = PolymarketAccount(
        private_key="0xpk",
        clob_api_key="k",
        clob_api_secret="s",
        clob_api_passphrase="p",
    )
    client = build_exec_client("v1_pm", acct, paper_mode=True)
    assert isinstance(client, PMClient)


def test_build_exec_client_unknown_type_raises():
    with pytest.raises(TypeError):
        build_exec_client("x", object(), paper_mode=True)
