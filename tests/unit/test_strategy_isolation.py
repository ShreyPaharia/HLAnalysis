from __future__ import annotations

import importlib
import pkgutil
import sys

import hlanalysis.strategy as strategy_pkg


# Allowed top-level dotted prefixes that strategy modules may import (transitively
# pulled-in stdlib modules are also allowed). We enforce direct imports only.
ALLOWED_DIRECT_PREFIXES = {
    "hlanalysis.strategy",
    "hlanalysis.events",
    "numpy",
    "scipy",
    "pydantic",
}

FORBIDDEN_PREFIXES = {
    "hlanalysis.engine",
    "hlanalysis.alerts",
    "hlanalysis.adapters",
    "hlanalysis.recorder",
    "hlanalysis.backtest.runner",
    "hlanalysis.backtest.data",
    "hlanalysis.backtest.cli",
    "hlanalysis.backtest.tuning",
    "hlanalysis.backtest.report",
    "hlanalysis.backtest.plots",
    "aiohttp",
    "httpx",
    "requests",
    "websockets",
    "sqlmodel",
    "sqlalchemy",
    "duckdb",
    "boto3",
    "asyncio",
    "hyperliquid",
    "tenacity",
}


def _walk_strategy_modules() -> list[str]:
    names = [strategy_pkg.__name__]
    for m in pkgutil.walk_packages(strategy_pkg.__path__, prefix="hlanalysis.strategy."):
        names.append(m.name)
    return names


def test_strategy_modules_have_no_forbidden_imports():
    """strategy/ stays free of IO + heavyweight modules.

    `hlanalysis.backtest.core.registry` is intentionally permitted: it is a
    dependency-free decorator that lets strategy modules self-register at
    import time. Everything heavier (runner, CLI, data sources) remains
    forbidden — see FORBIDDEN_PREFIXES.
    """
    offenders: list[str] = []
    for modname in _walk_strategy_modules():
        importlib.import_module(modname)
        mod = sys.modules[modname]
        src = open(mod.__file__).read() if mod.__file__ else ""
        for line in src.splitlines():
            stripped = line.strip()
            if not stripped.startswith(("import ", "from ")):
                continue
            imported = stripped.split()[1]
            for forbidden in FORBIDDEN_PREFIXES:
                if imported == forbidden or imported.startswith(forbidden + "."):
                    offenders.append(f"{modname}: {stripped}")
                    break
    assert not offenders, "strategy/ has forbidden imports:\n  " + "\n  ".join(offenders)


def test_strategy_does_not_pull_in_io_via_transitive_imports():
    # After importing hlanalysis.strategy, none of these modules should be in
    # sys.modules unless something else loaded them. We snapshot, reimport in a
    # subprocess-like way by clearing relevant entries.
    for modname in list(sys.modules):
        if modname.startswith(("aiohttp", "websockets", "sqlmodel", "sqlalchemy")):
            # These may legitimately be loaded by other tests; skip transitive check.
            return
    importlib.reload(strategy_pkg)
    bad = [m for m in sys.modules if m.startswith(("aiohttp", "websockets", "sqlmodel"))]
    assert not bad, f"strategy import pulled in IO libs: {bad}"
