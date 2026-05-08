from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from hlanalysis.sim.data.polymarket import fetch_prices_history


def test_prices_history_returns_sorted_records():
    fx = json.loads(Path("tests/fixtures/pm_clob_prices_history.json").read_text())
    with patch("hlanalysis.sim.data.polymarket._http_get", return_value=fx):
        rows = fetch_prices_history("token_xyz", interval="1m")
    assert len(rows) > 0
    assert all("ts_ns" in r and "price" in r for r in rows)
    ts = [r["ts_ns"] for r in rows]
    assert ts == sorted(ts)
