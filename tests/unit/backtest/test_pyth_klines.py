from hlanalysis.backtest.data._pyth_klines import active_cl_contract


def test_active_cl_contract_returns_pyth_wti_symbol():
    # Pyth deprecates expired CL contracts and removes them from the
    # tradingview shim, so the table collapses to the latest active
    # contract that retains historical depth (currently WTIN6). Any date
    # in the WTI Up/Down corpus window should map to a valid Pyth WTI
    # symbol — exact contract is whatever the table backfills with.
    for d in ("2026-03-10", "2026-04-15", "2026-05-15", "2026-05-26"):
        sym = active_cl_contract(d)
        assert sym.startswith("Commodities.WTI") and sym.endswith("/USD"), sym


def test_active_cl_contract_pre_corpus_returns_first_entry():
    # Dates before the earliest table entry should clamp to the first
    # entry rather than raise.
    assert active_cl_contract("2025-01-01").startswith("Commodities.WTI")


from unittest.mock import patch, Mock
from hlanalysis.backtest.data import _pyth_klines as pk


def test_fetch_klines_1m_parses_shim_response():
    fake = Mock()
    fake.json.return_value = {
        "s": "ok",
        "t": [1700000000, 1700000060],
        "o": [60.5, 60.6],
        "h": [60.7, 60.8],
        "l": [60.4, 60.5],
        "c": [60.6, 60.7],
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
