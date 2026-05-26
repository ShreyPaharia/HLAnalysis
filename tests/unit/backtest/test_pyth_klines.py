from hlanalysis.backtest.data._pyth_klines import active_cl_contract


def test_active_cl_contract_april_active_in_march():
    # CL April-26 contract (J6) is active in early March until its last
    # trade day. Anything in the J6 window should return J6.
    assert active_cl_contract("2026-03-10") == "Commodities.WTIJ6/USD"


def test_active_cl_contract_june_active_in_may():
    assert active_cl_contract("2026-05-15") == "Commodities.WTIM6/USD"


def test_active_cl_contract_july_active_late_may():
    assert active_cl_contract("2026-05-26") == "Commodities.WTIN6/USD"


def test_active_cl_contract_pre_corpus_returns_first_entry():
    assert active_cl_contract("2026-02-15").startswith("Commodities.WTI")


from unittest.mock import patch, Mock
from hlanalysis.backtest.data import _pyth_klines as pk


def test_fetch_klines_1m_parses_shim_response():
    fake = Mock()
    fake.json.return_value = {
        "s": "ok",
        "t": [1700000000, 1700000060],
        "o": [60.5, 60.6], "h": [60.7, 60.8],
        "l": [60.4, 60.5], "c": [60.6, 60.7],
    }
    fake.raise_for_status = Mock()
    with patch("hlanalysis.backtest.data._pyth_klines.requests.get", return_value=fake):
        rows = pk.fetch_klines_1m("Commodities.WTIN6/USD", 1700000000_000_000_000, 1700000120_000_000_000)
    assert len(rows) == 2
    assert rows[0] == {"ts_ns": 1700000000_000_000_000, "open": 60.5, "high": 60.7, "low": 60.4, "close": 60.6}
    assert rows[1]["close"] == 60.7


def test_fetch_klines_1m_returns_empty_on_error_status():
    fake = Mock()
    fake.json.return_value = {"s": "error", "errmsg": "Too many datapoints"}
    fake.raise_for_status = Mock()
    with patch("hlanalysis.backtest.data._pyth_klines.requests.get", return_value=fake):
        rows = pk.fetch_klines_1m("Commodities.WTIN6/USD", 0, 0)
    assert rows == []
