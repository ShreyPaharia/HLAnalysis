"""Guard: spawned backtest workers must import the PARENT's hlanalysis checkout.

SHR-100: the venv editable install is a `.pth` pointing at the main checkout. A
spawned multiprocessing worker can resolve `import hlanalysis` to MAIN instead of
the parent's checkout (e.g. a git worktree), silently running stale sim code. The
fix pins the parent's package root at `sys.path[0]` in every worker via a
ProcessPoolExecutor `initializer`.
"""
from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import hlanalysis
from hlanalysis.backtest.runner.parallel import (
    parent_package_root,
    worker_path_init,
)


def _probe(_):
    """Worker body: where does THIS process resolve hlanalysis, and what is
    its sys.path? Must be module-level so spawn can pickle it."""
    import sys

    import hlanalysis as _hl

    return (_hl.__file__, list(sys.path))


def test_parent_package_root_is_repo_root():
    assert parent_package_root() == str(Path(hlanalysis.__file__).resolve().parents[1])


def test_spawn_worker_imports_under_parent_package_root():
    root = parent_package_root()
    with ProcessPoolExecutor(
        max_workers=1,
        mp_context=mp.get_context("spawn"),
        initializer=worker_path_init,
        initargs=(root,),
    ) as ex:
        worker_file, worker_path = list(ex.map(_probe, [0]))[0]

    # The initializer pins the parent's package root as the first import path...
    assert worker_path[0] == root
    # ...so the worker's hlanalysis resolves under that exact checkout, not MAIN.
    assert Path(worker_file).resolve().is_relative_to(Path(root))
