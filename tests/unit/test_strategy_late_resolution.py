from __future__ import annotations

import dataclasses

import pytest

from hlanalysis.strategy.types import Action, Decision, Diagnostic, OrderIntent


def test_decision_is_frozen():
    d = Decision(action=Action.HOLD, intents=(), diagnostics=(Diagnostic("info", "noop"),))
    with pytest.raises(dataclasses.FrozenInstanceError):
        d.action = Action.ENTER  # type: ignore[misc]


def test_order_intent_signed_size_is_required():
    intent = OrderIntent(
        question_idx=42,
        symbol="@30",
        side="buy",
        size=10.0,
        limit_price=0.95,
        cloid="hla-test",
        time_in_force="ioc",
    )
    assert intent.size > 0
    assert intent.side in ("buy", "sell")


from hlanalysis.strategy.base import Strategy


def test_strategy_abc_cannot_be_instantiated_directly():
    import pytest as _p
    with _p.raises(TypeError):
        Strategy()  # type: ignore[abstract]
