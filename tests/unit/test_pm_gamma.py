from __future__ import annotations

import json
from pathlib import Path

from hlanalysis.adapters.polymarket_gamma import GammaClient


FIXTURE = Path("tests/fixtures/pm/gamma_active_btc_updown.json")


class _FakeHttp:
    def __init__(self, pages: list[list[dict]]):
        self._pages = list(pages)
        self.calls: list[tuple[str, dict]] = []

    def get(self, url: str, params: dict):
        self.calls.append((url, params))
        return self._pages.pop(0) if self._pages else []


def test_fetch_active_markets_paginates_until_short_page():
    page1 = [{"id": f"e{i}"} for i in range(100)]
    page2 = [{"id": "e100"}, {"id": "e101"}]
    http = _FakeHttp(pages=[page1, page2])
    gc = GammaClient(http_get=http.get)
    out = gc.fetch_events(series_slug="btc-up-or-down-daily", closed=False)
    assert len(out) == 102
    assert http.calls[0][1]["offset"] == 0
    assert http.calls[1][1]["offset"] == 100


def test_extract_active_binary_markets_from_fixture():
    raw = json.loads(FIXTURE.read_text())
    http = _FakeHttp(pages=[raw, []])
    gc = GammaClient(http_get=http.get)
    events = gc.fetch_events(series_slug="btc-up-or-down-daily", closed=False)
    markets = list(gc.iter_binary_markets(events))
    assert markets  # fixture must contain at least one
    for m in markets:
        assert m.get("conditionId")
        assert m.get("clobTokenIds")


def test_iter_yields_every_market_in_multi_market_events():
    """Crypto weekly / equity hit-price / NBA games each pack multiple
    independent priceBinary markets under one event. The helper now yields
    each sub-market provided it has valid clobTokenIds + conditionId."""
    events = [{
        "id": "ev-multi",
        "markets": [
            {"conditionId": "c1", "clobTokenIds": "[\"t1y\", \"t1n\"]"},
            {"conditionId": "c2", "clobTokenIds": "[\"t2y\", \"t2n\"]"},
            {"conditionId": "c3", "clobTokenIds": "[\"t3y\", \"t3n\"]"},
            # missing clobTokenIds → filtered out
            {"conditionId": "c4"},
            # missing conditionId → filtered out
            {"clobTokenIds": "[\"t5y\", \"t5n\"]"},
        ],
    }]
    gc = GammaClient(http_get=lambda url, params: events)
    markets = list(gc.iter_binary_markets(events))
    assert [m["conditionId"] for m in markets] == ["c1", "c2", "c3"]
