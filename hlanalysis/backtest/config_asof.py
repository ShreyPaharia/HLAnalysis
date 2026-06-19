"""Resolve config/strategy.yaml at a historical git commit date.

Used by the ``--slot-config-asof`` CLI flag to source the live config
as it existed on a specific date — e.g. for backtesting a past period
with the config that was actually deployed at that time.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


def resolve_config_asof(date: str, repo_root: Path) -> tuple[str, Path]:
    """Find the last git commit touching config/strategy.yaml with commit
    date <= ``date`` (YYYY-MM-DD).

    Returns ``(commit_hash, temp_file_path)`` where ``temp_file_path`` is a
    :class:`~tempfile.NamedTemporaryFile` path containing the file contents at
    that commit. The caller is responsible for deleting the temp file.

    Raises :class:`ValueError` if no commit is found before or on ``date``.
    """
    # git log returns commits newest-first; each line is "<hash> <ISO-datetime>".
    result = subprocess.run(
        ["git", "log", "--format=%H %ci", "--", "config/strategy.yaml"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    lines = [ln.strip() for ln in result.stdout.splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"No git history found for config/strategy.yaml in {repo_root}. Is this a git repository?")

    # Parse date prefix "YYYY-MM-DD" from the commit timestamp
    # "<hash> YYYY-MM-DD HH:MM:SS +NNNN".
    target_date = date.strip()
    chosen_hash: str | None = None
    for line in lines:
        parts = line.split(" ", 2)
        if len(parts) < 2:
            continue
        commit_hash = parts[0]
        commit_date = parts[1]  # "YYYY-MM-DD"
        if commit_date <= target_date:
            chosen_hash = commit_hash
            break

    if chosen_hash is None:
        earliest_date = lines[-1].split(" ", 2)[1] if lines else "unknown"
        raise ValueError(
            f"No config/strategy.yaml commit found on or before {target_date!r}. "
            f"Earliest commit date is {earliest_date!r}."
        )

    # Extract the file contents at that commit.
    show_result = subprocess.run(
        ["git", "show", f"{chosen_hash}:config/strategy.yaml"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )

    # Write to a temp file the caller can pass as --slot-config.
    # Use NamedTemporaryFile to get a unique path, then write and close it so
    # the caller reads the file after this function returns (delete=False keeps
    # the file alive until the caller explicitly unlinks it).
    with tempfile.NamedTemporaryFile(
        suffix=".yaml",
        prefix="strategy_asof_",
        mode="w",
        encoding="utf-8",
        delete=False,
    ) as tmp:
        tmp.write(show_result.stdout)
        tmp_name = tmp.name

    return chosen_hash, Path(tmp_name)


def _repo_root_from_file(file: str | Path) -> Path:
    """Walk up from ``file`` to find the git repo root.

    Uses ``git rev-parse --show-toplevel`` so worktrees (which use a ``.git``
    *file*, not a directory) are handled correctly.
    """
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=Path(file).parent,
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(result.stdout.strip())
