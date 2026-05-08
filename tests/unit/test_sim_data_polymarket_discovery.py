from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from hlanalysis.sim.data.polymarket import discover_btc_updown_markets


def test_discover_parses_gamma_event_into_pmmarket(tmp_path):
    fixture = json.loads(
        Path("tests/fixtures/pm_gamma_response.json").read_text()
    )
    with patch("hlanalysis.sim.data.polymarket._http_get", return_value=fixture):
        markets = discover_btc_updown_markets(date_iso_list=["2026-05-09"])
    assert len(markets) >= 1
    m = markets[0]
    assert m.condition_id.startswith("0x") or m.condition_id  # non-empty
    assert m.yes_token_id and m.no_token_id and m.yes_token_id != m.no_token_id
    assert m.end_ts_ns > m.start_ts_ns
