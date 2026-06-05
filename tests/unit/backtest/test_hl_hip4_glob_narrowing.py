"""Narrowing the perp-BTC reference glob to in-range ``date=`` sub-partitions.

The whole-partition ``**/*.parquet`` recursive walk over a 12k-file partition is
fragile under heavy spawn-worker concurrency: duckdb's parallel glob expansion
intermittently raises ``IOException: No files found`` even though the files exist
(SHR-71). Restricting the read to the handful of in-range ``date=`` dirs avoids
the deep recursive walk; a flat (non-date-partitioned) layout falls back to the
original glob unchanged.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hlanalysis.backtest.data.hl_hip4 import HLHip4DataSource


def _touch(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x")


def test_narrowed_globs_restricts_to_existing_date_partitions(tmp_path):
    """When ``date=`` sub-partitions exist, the read is restricted to the
    in-range ones — the whole-partition ``**`` glob is NOT used."""
    part = tmp_path / "event=bbo" / "symbol=BTC"
    _touch(part / "date=2026-05-10" / "hour=00" / "a.parquet")
    _touch(part / "date=2026-05-11" / "hour=00" / "a.parquet")
    base = str(part / "**" / "*.parquet")
    src = HLHip4DataSource(data_root=tmp_path)

    globs = src._narrowed_globs(base, ["2026-05-10", "2026-05-11"])

    assert base not in globs  # avoids the 12k-file recursive walk
    assert any("date=2026-05-10" in g for g in globs)
    assert any("date=2026-05-11" in g for g in globs)


def test_narrowed_globs_skips_absent_dates(tmp_path):
    """A date in the range whose partition dir does not exist is dropped (the
    +1-day padding in ``_date_partitions_in_range`` produces such dates)."""
    part = tmp_path / "event=bbo" / "symbol=BTC"
    _touch(part / "date=2026-05-10" / "a.parquet")
    base = str(part / "**" / "*.parquet")
    src = HLHip4DataSource(data_root=tmp_path)

    globs = src._narrowed_globs(base, ["2026-05-10", "2026-05-99-absent"])

    assert globs == [str(part / "date=2026-05-10" / "**" / "*.parquet")]


def test_narrowed_globs_falls_back_for_flat_layout(tmp_path):
    """A flat layout (files directly under ``symbol=BTC``, ``date`` as a column,
    not a directory — the fixture shape) has no ``date=`` dirs, so narrowing
    falls back to the base glob and behaviour is unchanged."""
    part = tmp_path / "event=bbo" / "symbol=BTC"
    _touch(part / "fixture.parquet")
    base = str(part / "**" / "*.parquet")
    src = HLHip4DataSource(data_root=tmp_path)

    globs = src._narrowed_globs(base, ["2026-05-10", "2026-05-11"])

    assert globs == [base]
