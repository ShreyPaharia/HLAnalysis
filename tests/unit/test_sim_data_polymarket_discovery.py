from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from hlanalysis.sim.data.polymarket import (
    _event_in_window,
    _gamma_event_to_pmmarket,
    discover_btc_updown_markets,
)


def _load_series_fixture() -> list[dict]:
    return json.loads(Path("tests/fixtures/pm_gamma_series_page.json").read_text())


def _load_single_event_fixture() -> list[dict]:
    return json.loads(Path("tests/fixtures/pm_gamma_response.json").read_text())


def test_event_in_window_inclusive_lower_exclusive_upper():
    ev = {"endDate": "2026-03-13T16:00:00Z"}
    assert _event_in_window(ev, "2026-03-13", "2026-03-14")
    assert _event_in_window(ev, "2026-01-01", "2026-03-14")
    assert not _event_in_window(ev, "2026-03-14", "2026-04-01")
    assert not _event_in_window(ev, "2026-01-01", "2026-03-13")  # upper exclusive
    assert not _event_in_window({"endDate": ""}, "2026-01-01", "2027-01-01")
    assert not _event_in_window({}, "2026-01-01", "2027-01-01")


def test_gamma_event_to_pmmarket_parses_pre_and_post_slug_shift():
    """Parser is slug-agnostic: works on both `...-may-10` and `...-may-8-2026`."""
    events = _load_series_fixture()
    parsed = [_gamma_event_to_pmmarket(ev) for ev in events]
    assert all(m is not None for m in parsed), "every fixture event should parse"
    # All three are resolved (closed series events)
    for m in parsed:
        assert m is not None
        assert m.condition_id.startswith("0x")
        assert m.yes_token_id and m.no_token_id and m.yes_token_id != m.no_token_id
        assert m.end_ts_ns > m.start_ts_ns
        assert m.resolved_outcome in {"yes", "no"}


def test_discover_filters_by_window_and_paginates():
    """Single-page response (len < page limit) terminates pagination cleanly,
    and only events with endDate in [start, end) survive the filter."""
    events = _load_series_fixture()
    # Window covers only the middle event (Oct 7 2025).
    with patch("hlanalysis.sim.data.polymarket._http_get", return_value=events):
        markets = discover_btc_updown_markets("2025-10-01", "2025-11-01")
    assert len(markets) == 1
    # Window covers all three.
    with patch("hlanalysis.sim.data.polymarket._http_get", return_value=events):
        markets = discover_btc_updown_markets("2025-01-01", "2027-01-01")
    assert len(markets) == 3


def test_discover_paginates_until_short_page():
    """When Gamma returns a full page, discovery loops; an empty next page stops it."""
    full_page = _load_series_fixture() * 200  # 600 records, > 500 page limit
    pages = [full_page[:500], full_page[500:600]]  # 500 then 100 (< 500 → stop)
    call_log: list[dict] = []

    def fake_get(url, params=None):
        call_log.append(params or {})
        return pages[len(call_log) - 1]

    with patch("hlanalysis.sim.data.polymarket._http_get", side_effect=fake_get):
        # Wide window so nothing gets filtered out
        markets = discover_btc_updown_markets("2020-01-01", "2030-01-01")
    assert len(call_log) == 2
    assert call_log[0]["offset"] == 0
    assert call_log[1]["offset"] == 500
    assert len(markets) == 600


def test_discover_back_compat_with_single_event_payload():
    """Older fixture (single event from per-slug call) still parses correctly when
    Gamma returns it via the series query."""
    payload = _load_single_event_fixture()  # length-1 list
    with patch("hlanalysis.sim.data.polymarket._http_get", return_value=payload):
        markets = discover_btc_updown_markets("2026-05-01", "2026-05-31")
    assert len(markets) == 1
    m = markets[0]
    assert m.yes_token_id and m.no_token_id
    assert m.end_ts_ns > m.start_ts_ns
