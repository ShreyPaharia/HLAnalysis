from __future__ import annotations

from hlanalysis.adapters.binance_klines import binance_1m_close_at

# May 30 2026 16:00:00 UTC, in ns and the matching minute-start ms.
_REF_NS = 1_780_156_800_000_000_000
_MIN_MS = 1_780_156_800_000


def _kline(open_ms: int, close: str) -> list:
    # Binance kline: [openTime, open, high, low, close, volume, closeTime, ...]
    return [open_ms, "73888.00", "73895.30", "73861.20", close, "66.2", open_ms + 59_999, "0", 974, "0", "0", "0"]


def test_uses_spot_close_only():
    # PM "BTC Up or Down" markets resolve against the Binance *spot* BTC/USDT
    # 1m candle close (api.binance.com), NOT the perp (fapi). The strike must
    # come from spot so it matches PM's literal settlement reference; the
    # perp/spot basis (tens of dollars) would otherwise bias every entry's edge.
    calls = []

    def fake_get(url, params):
        calls.append(url)
        return [_kline(_MIN_MS, "73644.92")]

    assert binance_1m_close_at(_REF_NS, http_get=fake_get) == 73644.92
    # Spot only — the perp endpoint must never be consulted.
    assert all("fapi.binance.com" not in u for u in calls)
    assert any("api.binance.com" in u for u in calls)


def test_does_not_fall_back_to_perp_when_spot_fails():
    # Spot is the only acceptable source. If spot can't be fetched we return
    # None (slot skips the market) rather than silently trading on a
    # basis-biased perp strike.
    def fake_get(url, params):
        if "api.binance.com" in url and "fapi" not in url:
            raise RuntimeError("spot blocked")
        return [_kline(_MIN_MS, "73609.30")]  # perp value — must NOT be used

    assert binance_1m_close_at(_REF_NS, http_get=fake_get) is None


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
        return [_kline(_MIN_MS, "73644.92")]

    # 37s past the minute → still floors to the minute start.
    binance_1m_close_at(_REF_NS + 37_000_000_000, http_get=fake_get)
    assert captured["startTime"] == _MIN_MS
