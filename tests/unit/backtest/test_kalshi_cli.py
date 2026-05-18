from __future__ import annotations

from pathlib import Path

from hlanalysis.backtest.cli import (
    _resolve_data_source,
    make_kalshi_source,
)


def test_resolve_data_source_kalshi(tmp_path):
    ds = _resolve_data_source("kalshi", cache_root=str(tmp_path))
    assert ds.name == "kalshi"


def test_make_kalshi_source_uses_env_default(monkeypatch, tmp_path):
    monkeypatch.setenv("HLBT_KALSHI_CACHE_ROOT", str(tmp_path))
    ds = make_kalshi_source()
    assert ds.name == "kalshi"
