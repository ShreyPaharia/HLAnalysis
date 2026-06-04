from __future__ import annotations

import inspect

from hlanalysis.engine.runtime import EngineRuntime


def test_enforce_stop_losses_is_extracted_method():
    # The stop-loss enforcement must be callable in isolation (so the
    # event-driven loop and any test can drive one pass).
    assert hasattr(EngineRuntime, "_enforce_stop_losses")
    sig = inspect.signature(EngineRuntime._enforce_stop_losses)
    assert {"slot", "now_ns"} <= set(sig.parameters)


def test_stop_loss_loop_exists():
    assert hasattr(EngineRuntime, "_stop_loss_loop")
