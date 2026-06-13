from __future__ import annotations

from unittest.mock import patch

from hlanalysis.backtest.data.binance_klines import fetch_perp_klines


def test_fetch_perp_klines_hits_fapi_endpoint() -> None:
    captured: list[str] = []

    def fake_get(url, params, timeout):
        captured.append(url)

        class R:
            def raise_for_status(self): ...
            def json(self):
                # Return one row, then empty to end the loop
                if len(captured) == 1:
                    return [
                        [
                            1746057600000,
                            "94172.0",
                            "94177.96",
                            "94130.43",
                            "94147.3",
                            "6.47436",
                            1746057659999,
                            "",
                            0,
                            "",
                            "",
                            "",
                        ]
                    ]
                return []

        return R()

    with patch("hlanalysis.backtest.data.binance_klines.requests.get", side_effect=fake_get):
        rows = fetch_perp_klines(1746057600000, 1746057660000)

    assert any("fapi.binance.com" in u for u in captured)
    assert len(rows) == 1
    assert rows[0].open == 94172.0
    assert rows[0].close == 94147.3
