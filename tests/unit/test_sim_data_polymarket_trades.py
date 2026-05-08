from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from hlanalysis.sim.data.polymarket import fetch_trades


def test_fetch_trades_paginates_and_parses(tmp_path):
    fx = json.loads(Path("tests/fixtures/pm_clob_trades.json").read_text())
    # First call returns the fixture; second returns empty (end of pagination)
    with patch(
        "hlanalysis.sim.data.polymarket._http_get",
        side_effect=[fx, []],
    ):
        trades = fetch_trades("0xfakecond")
    assert len(trades) == len(fx)
    t = trades[0]
    assert t.ts_ns > 0
    assert t.side in ("buy", "sell")
    assert 0 <= t.price <= 1
    assert t.size > 0
