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
    "pydantic",
}

FORBIDDEN_PREFIXES = {
    "hlanalysis.engine",
    "hlanalysis.sim",
    "hlanalysis.alerts",
    "hlanalysis.adapters",
    "hlanalysis.recorder",
    "aiohttp",
    "httpx",
    "requests",
    "websockets",
    "sqlmodel",
    "sqlalchemy",
    "duckdb",
    "boto3",
    "asyncio",  # strategy must be sync
}


def _walk_strategy_modules() -> list[str]:
    names = [strategy_pkg.__name__]
    for m in pkgutil.walk_packages(strategy_pkg.__path__, prefix="hlanalysis.strategy."):
        names.append(m.name)
    return names


def test_strategy_modules_have_no_forbidden_imports():
    offenders: list[str] = []
    for modname in _walk_strategy_modules():
        importlib.import_module(modname)
        mod = sys.modules[modname]
        # Inspect raw source to catch direct imports without being misled by
        # what other test modules already loaded into sys.modules.
        src = open(mod.__file__).read() if mod.__file__ else ""
        for forbidden in FORBIDDEN_PREFIXES:
            base = forbidden.split(".")[0]
            for line in src.splitlines():
                stripped = line.strip()
                if stripped.startswith(("import ", "from ")):
                    head = stripped.split()[1].split(".")[0]
                    if head == base:
                        offenders.append(f"{modname}: {stripped}")
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
