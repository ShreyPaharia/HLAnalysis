from __future__ import annotations

from typing import Any, Callable

from hlanalysis.strategy.base import Strategy


_REGISTRY: dict[str, Callable[[dict[str, Any]], Strategy]] = {}


def register(strategy_id: str) -> Callable[
    [Callable[[dict[str, Any]], Strategy]], Callable[[dict[str, Any]], Strategy]
]:
    """Decorator: ``@register("v1_late_resolution")`` above the factory function."""

    def _wrap(
        fn: Callable[[dict[str, Any]], Strategy],
    ) -> Callable[[dict[str, Any]], Strategy]:
        if strategy_id in _REGISTRY:
            raise ValueError(f"Strategy id already registered: {strategy_id}")
        _REGISTRY[strategy_id] = fn
        return fn

    return _wrap


def build(strategy_id: str, params: dict[str, Any]) -> Strategy:
    if strategy_id not in _REGISTRY:
        raise KeyError(
            f"Unknown strategy id: {strategy_id}. Registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[strategy_id](params)


def ids() -> list[str]:
    return sorted(_REGISTRY)


def _reset_for_tests() -> None:
    """Test helper: drop all registrations. Not part of the public contract."""
    _REGISTRY.clear()


__all__ = ["register", "build", "ids"]
