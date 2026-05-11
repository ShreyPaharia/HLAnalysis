from __future__ import annotations

import pytest

from hlanalysis.backtest.core import registry
from hlanalysis.strategy.base import Strategy
from hlanalysis.strategy.types import Action, Decision, Diagnostic


class _StubStrategy(Strategy):
    name = "stub"

    def __init__(self, params: dict):
        self.params = params

    def evaluate(self, **_kwargs) -> Decision:  # type: ignore[override]
        return Decision(
            action=Action.HOLD,
            diagnostics=(Diagnostic(level="info", message="stub"),),
        )


@pytest.fixture(autouse=True)
def _isolate_registry():
    """Each test starts with an empty registry to avoid cross-test bleed."""
    snapshot = dict(registry._REGISTRY)
    registry._reset_for_tests()
    try:
        yield
    finally:
        registry._reset_for_tests()
        registry._REGISTRY.update(snapshot)


def test_register_then_build():
    @registry.register("stub")
    def _factory(params: dict) -> Strategy:
        return _StubStrategy(params)

    assert registry.ids() == ["stub"]
    strat = registry.build("stub", {"a": 1})
    assert isinstance(strat, _StubStrategy)
    assert strat.params == {"a": 1}


def test_duplicate_registration_raises():
    @registry.register("dup")
    def _f1(p):
        return _StubStrategy(p)

    with pytest.raises(ValueError, match="already registered"):
        @registry.register("dup")
        def _f2(p):  # pragma: no cover - error path
            return _StubStrategy(p)


def test_build_unknown_raises():
    with pytest.raises(KeyError, match="Unknown strategy id"):
        registry.build("does-not-exist", {})


def test_ids_sorted_lexicographically():
    @registry.register("zeta")
    def _z(p):
        return _StubStrategy(p)

    @registry.register("alpha")
    def _a(p):
        return _StubStrategy(p)

    assert registry.ids() == ["alpha", "zeta"]


def test_registry_isolation_across_tests():
    # Fixture must clear between tests; this is checked by the previous fixture
    # plus this assertion (no prior registrations leak in).
    assert registry.ids() == []
