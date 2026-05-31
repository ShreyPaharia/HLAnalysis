from __future__ import annotations

from hlanalysis.adapters.binance_klines import binance_1m_close_at

# May 30 2026 16:00:00 UTC, in ns and the matching minute-start ms.
_REF_NS = 1_780_156_800_000_000_000
_MIN_MS = 1_780_156_800_000


def _kline(open_ms: int, close: str) -> list:
    # Binance kline: [openTime, open, high, low, close, volume, closeTime, ...]
    return [open_ms, "73888.00", "73895.30", "73861.20", close, "66.2",
            open_ms + 59_999, "0", 974, "0", "0", "0"]


def test_returns_perp_close_when_perp_available():
    calls = []

    def fake_get(url, params):
        calls.append(url)
        return [_kline(_MIN_MS, "73861.30")]

    assert binance_1m_close_at(_REF_NS, http_get=fake_get) == 73861.30
    # Perp (fapi) is tried first.
    assert "fapi.binance.com" in calls[0]


def test_falls_back_to_spot_when_perp_fails():
    def fake_get(url, params):
        if "fapi.binance.com" in url:
            raise RuntimeError("perp blocked")
        return [_kline(_MIN_MS, "73905.42")]

    assert binance_1m_close_at(_REF_NS, http_get=fake_get) == 73905.42


def test_rejects_candle_for_wrong_minute():
    # A gap: Binance returns the next candle, not the requested minute → reject.
    def fake_get(url, params):
        return [_kline(_MIN_MS + 60_000, "73861.30")]

    assert binance_1m_close_at(_REF_NS, http_get=fake_get) is None


def test_returns_none_on_empty_and_total_failure():
    assert binance_1m_close_at(_REF_NS, http_get=lambda u, p: []) is None

    def boom(url, params):
        raise RuntimeError("down")

    assert binance_1m_close_at(_REF_NS, http_get=boom) is None


def test_floors_ts_to_minute_start():
    captured = {}

    def fake_get(url, params):
        captured["startTime"] = params["startTime"]
        return [_kline(_MIN_MS, "73861.30")]

    # 37s past the minute → still floors to the minute start.
    binance_1m_close_at(_REF_NS + 37_000_000_000, http_get=fake_get)
    assert captured["startTime"] == _MIN_MS
