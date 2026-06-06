"""Shared IOC order-intent construction for the policy strategies.

Every strategy builds the same two ``OrderIntent`` shapes:

* an ENTER intent — ``side="buy"``, positive size, IOC limit at the ask, fresh
  ``hla-<uuid>`` cloid;
* an EXIT intent — side flipped off the held qty's sign, ``size=abs(qty)``,
  IOC limit at the bid, ``reduce_only=True``, and an ``exit_reason`` tag.

These were copy-pasted 8+ times across the strategy modules. Consolidating them
keeps the cloid scheme and reduce-only/exit-reason wiring in one place. The 2-dp
size rounding (``floor(usd/px * 100) / 100``) is also shared here as
``round_size``.
"""
from __future__ import annotations

import math
import uuid

from .types import OrderIntent, Position, QuestionView


def round_size(usd: float, px: float) -> float:
    """Contracts affordable for ``usd`` at price ``px``, floored to 2 dp.

    Matches the legacy inline ``math.floor((usd / px) * 100) / 100``. Callers
    that need a non-negative floor still wrap this in ``max(0.0, ...)`` exactly
    as before — this helper does not clamp so the topup path (which already
    guarantees a positive shortfall) is bit-identical.
    """
    return math.floor((usd / px) * 100) / 100


def make_entry_intent(
    question: QuestionView, *, symbol: str, size: float, limit_price: float
) -> OrderIntent:
    """Build a buy IOC entry intent with a fresh cloid."""
    return OrderIntent(
        question_idx=question.question_idx,
        symbol=symbol,
        side="buy",
        size=size,
        limit_price=limit_price,
        cloid=f"hla-{uuid.uuid4()}",
        time_in_force="ioc",
    )


def make_exit_intent(
    question: QuestionView,
    position: Position,
    *,
    limit_price: float,
    exit_reason: str = "",
) -> OrderIntent:
    """Build a reduce-only IOC exit intent that flattens ``position``.

    Side is flipped off the held qty's sign and size is ``abs(qty)``.
    ``exit_reason`` defaults to "" so callers that only tag the diagnostic (not
    the intent) stay bit-identical.
    """
    return OrderIntent(
        question_idx=question.question_idx,
        symbol=position.symbol,
        side="sell" if position.qty > 0 else "buy",
        size=abs(position.qty),
        limit_price=limit_price,
        cloid=f"hla-{uuid.uuid4()}",
        time_in_force="ioc",
        reduce_only=True,
        exit_reason=exit_reason,
    )
