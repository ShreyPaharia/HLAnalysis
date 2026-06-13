"""Interval-aware paging for ``fetch_klines`` (1s support).

``fetch_klines`` historically advanced its paging cursor by a hardcoded 60_000ms
(one minute). That is correct only for ``interval="1m"``; for ``interval="1s"``
it would skip 59 of every 60 seconds. These tests pin the interval-aware cursor
advance and assert the 1m path stays bit-identical (+60_000).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from hlanalysis.backtest.data.binance_klines import _LIMIT, fetch_klines


def _row(open_ms: int, close: str = "81000.0") -> list:
    # [openTime, open, high, low, close, volume, closeTime, ...]
    return [open_ms, "81000.0", "81005.0", "80995.0", close, "1.0", open_ms + 999, "", 1, "", "", ""]


def _paging_fetch(interval: str):
    """Run fetch_klines against a fake Binance that returns one FULL page
    (``_LIMIT`` rows) spaced by the real interval, then an empty page. A full
    page forces the loop to continue (it only breaks on a short page), so the
    second call's ``startTime`` exercises the cursor-advance logic."""
    cursors: list[int] = []
    spacing = {"1s": 1_000, "1m": 60_000}[interval]
    base = 1_778_000_000_000
    first_page_open_ms = [base + i * spacing for i in range(_LIMIT)]

    def fake_get(url, params, timeout):
        cursors.append(int(params["startTime"]))
        page = [_row(o) for o in first_page_open_ms] if len(cursors) == 1 else []

        class R:
            def raise_for_status(self): ...
            def json(self):
                return page

        return R()

    with patch("hlanalysis.backtest.data.binance_klines.requests.get", side_effect=fake_get):
        rows = fetch_klines(base, base + _LIMIT * spacing * 2, interval=interval)
    return rows, cursors, spacing, first_page_open_ms


def test_1s_interval_pages_by_one_second() -> None:
    # The cursor must advance 1000ms past the last bar's open, not 60_000, or
    # the next page would skip 59 of every 60 seconds.
    rows, cursors, spacing, page_open_ms = _paging_fetch("1s")
    assert len(rows) == _LIMIT
    assert cursors[1] == page_open_ms[-1] + spacing  # = +1000


def test_1m_interval_bit_identical_advance() -> None:
    # The 1m path must keep advancing by exactly 60_000ms (no behavior change).
    rows, cursors, spacing, page_open_ms = _paging_fetch("1m")
    assert len(rows) == _LIMIT
    assert cursors[1] == page_open_ms[-1] + 60_000


def test_unknown_interval_raises() -> None:
    with pytest.raises(ValueError, match="interval"):
        fetch_klines(0, 1000, interval="7s")


def test_1s_interval_passed_to_api() -> None:
    captured: dict = {}

    def fake_get(url, params, timeout):
        captured.update(params)

        class R:
            def raise_for_status(self): ...
            def json(self):
                return []

        return R()

    with patch("hlanalysis.backtest.data.binance_klines.requests.get", side_effect=fake_get):
        fetch_klines(1_778_000_000_000, 1_778_000_010_000, interval="1s")
    assert captured["interval"] == "1s"
