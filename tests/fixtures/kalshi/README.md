# Kalshi fixture

Tiny synthetic Kalshi BTC bucket event for backtest smoke. One event,
three markets, nine trades total.

**Refresh:** `uv run python tests/fixtures/kalshi/_build_fixture.py`

Why synthetic instead of captured? The smoke test exercises adapter mechanics
(descriptor build → events stream → settlement). A captured live response
would pin us to Kalshi schema details that may rev. We catch schema drift
separately in `tests/integration/test_kalshi_live_smoke.py` (gated, runs
locally).
