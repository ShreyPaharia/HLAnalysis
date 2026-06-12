"""Free-function helpers extracted from ``runtime.py``.

These are module-level utilities with no dependency on ``EngineRuntime`` state.
Imported back into ``runtime.py`` for backward-compatibility — all existing
callers (including tests that import directly from ``hlanalysis.engine.runtime``)
continue to work without modification.

Do NOT import ``runtime.py`` from this module — that would create a cycle.
"""
from __future__ import annotations

import msgspec

from ..events import NormalizedEvent, ProductType

# Map of Binance SPOT symbols → internal remapped symbols used in MarketState.
# This constant lives in runtime.py too (back-compat). Both point at the same
# logical mapping; keep them in sync if the set of SPOT symbols ever changes.
_SPOT_REF_SYMBOLS = {"BTCUSDT": "BTCUSDT_SPOT", "ETHUSDT": "ETHUSDT_SPOT"}


def _is_transient_venue_error(exc: BaseException) -> bool:
    """True for expected, self-recovering venue read failures the client's
    read-retry already exhausted — HL 429 rate limits, connection blips, 5xx.

    The reconcile loop uses this to log such failures concisely and retry next
    cycle, rather than emitting a full ``reconcile crashed`` traceback for
    routine venue flakiness (e.g. HL 429 bursts when two slots poll /info from
    one IP). Genuine, non-transient errors still surface with a full traceback.

    Covers both venues: HL's RateLimitError plus the PM client's own
    transient classifier (requests/builtin conn+timeout, HTTP 5xx and 429)."""
    from .hl_client import RateLimitError
    from .pm_client import _pm_is_transient
    if isinstance(exc, (RateLimitError, ConnectionError, TimeoutError)):
        return True
    return _pm_is_transient(exc)


def _remap_reference_symbol(ev: NormalizedEvent) -> NormalizedEvent:
    """Rename Binance SPOT bbo events to ``<symbol>_SPOT`` so PM slots'
    ``reference_symbol`` resolves to the spot feed (not any perp entry).
    No-op for every other event."""
    if ev.venue == "binance" and ev.product_type == ProductType.SPOT:
        mapped = _SPOT_REF_SYMBOLS.get(ev.symbol)
        if mapped is not None:
            return msgspec.structs.replace(ev, symbol=mapped)
    return ev


def _stub_question(p):
    from ..strategy.types import QuestionView
    return QuestionView(
        question_idx=p.question_idx, yes_symbol=p.symbol, no_symbol="",
        strike=0.0, expiry_ns=0, underlying="", klass="", period="",
    )
