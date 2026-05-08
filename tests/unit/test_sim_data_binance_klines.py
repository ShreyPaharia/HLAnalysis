from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from hlanalysis.sim.data.binance_klines import fetch_klines


def test_fetch_klines_parses_open_close():
    fx = json.loads(Path("tests/fixtures/binance_klines_btcusdt_1m.json").read_text())
    with patch("hlanalysis.sim.data.binance_klines._http_get",
               side_effect=[fx, []]):
        rows = fetch_klines(start_ts_ms=0, end_ts_ms=10**13)
    assert len(rows) == len(fx)
    r = rows[0]
    assert r.ts_ns > 0
    assert r.open > 0 and r.close > 0
