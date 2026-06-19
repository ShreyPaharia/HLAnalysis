"""Tests for hlanalysis.backtest.config_asof.

Uses the repo's real git history of config/strategy.yaml.
"""

from __future__ import annotations

import pytest
import yaml

from hlanalysis.backtest.config_asof import _repo_root_from_file, resolve_config_asof

# Derive the repo root from this file's location (works from worktrees too).
_REPO_ROOT = _repo_root_from_file(__file__)


def _skip_if_no_git_history() -> None:
    """Skip if config/strategy.yaml has no git history in this repo."""
    import subprocess

    result = subprocess.run(
        ["git", "log", "--format=%H", "--", "config/strategy.yaml"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if not result.stdout.strip():
        pytest.skip("config/strategy.yaml has no git history in this repo")


class TestResolveConfigAsof:
    def test_resolves_known_date(self, tmp_path):
        """Resolve config as of 2026-06-13; should return c718f54a or an earlier commit."""
        _skip_if_no_git_history()
        commit_hash, out_path = resolve_config_asof("2026-06-13", _REPO_ROOT)
        try:
            assert len(commit_hash) == 40, f"Expected 40-hex commit hash, got: {commit_hash!r}"
            # c718f54a is the commit dated 2026-06-13 in the canonical repo.
            # Allow any valid hash that is a real git commit.
            assert all(c in "0123456789abcdef" for c in commit_hash.lower()), f"Not a hex hash: {commit_hash!r}"
            assert out_path.exists(), f"Temp file missing: {out_path}"
            assert out_path.suffix == ".yaml"
            # Must be valid YAML.
            content = out_path.read_text(encoding="utf-8")
            parsed = yaml.safe_load(content)
            assert isinstance(parsed, dict), "Expected YAML dict at top level"
        finally:
            out_path.unlink(missing_ok=True)

    def test_resolves_to_specific_commit(self, tmp_path):
        """Resolve as of 2026-06-13 should produce the commit dated exactly that day."""
        _skip_if_no_git_history()
        import subprocess

        result = subprocess.run(
            ["git", "log", "--format=%H %ci", "--", "config/strategy.yaml"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        lines = [ln.strip() for ln in result.stdout.splitlines() if ln.strip()]
        # Find the expected commit: first one with date <= 2026-06-13.
        expected_hash = None
        for line in lines:
            parts = line.split(" ", 2)
            if len(parts) >= 2 and parts[1] <= "2026-06-13":
                expected_hash = parts[0]
                break
        if expected_hash is None:
            pytest.skip("No commit found on or before 2026-06-13 in this repo")

        commit_hash, out_path = resolve_config_asof("2026-06-13", _REPO_ROOT)
        try:
            assert commit_hash == expected_hash, f"Expected {expected_hash}, got {commit_hash}"
        finally:
            out_path.unlink(missing_ok=True)

    def test_raises_for_date_before_any_commits(self):
        """A date before any commits should raise ValueError."""
        _skip_if_no_git_history()
        with pytest.raises(ValueError, match="No config/strategy.yaml commit found"):
            resolve_config_asof("1990-01-01", _REPO_ROOT)

    def test_returned_path_is_valid_yaml(self):
        """The returned path must contain parseable YAML."""
        _skip_if_no_git_history()
        commit_hash, out_path = resolve_config_asof("2026-06-13", _REPO_ROOT)
        try:
            content = out_path.read_text(encoding="utf-8")
            parsed = yaml.safe_load(content)
            # The live config is a dict with at least a "strategies" key.
            assert isinstance(parsed, dict)
            assert "strategies" in parsed, f"Expected 'strategies' key, got keys: {list(parsed)}"
        finally:
            out_path.unlink(missing_ok=True)

    def test_temp_file_has_yaml_suffix(self):
        """The temp file path must end in .yaml."""
        _skip_if_no_git_history()
        _, out_path = resolve_config_asof("2026-06-13", _REPO_ROOT)
        try:
            assert out_path.suffix == ".yaml"
        finally:
            out_path.unlink(missing_ok=True)

    def test_repo_root_from_file(self):
        """_repo_root_from_file should return a path containing config/strategy.yaml."""
        root = _repo_root_from_file(__file__)
        assert (root / "config" / "strategy.yaml").exists(), f"config/strategy.yaml not found under {root}"
