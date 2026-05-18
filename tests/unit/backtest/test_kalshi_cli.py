from __future__ import annotations

import json

from pathlib import Path
from unittest.mock import patch

from hlanalysis.backtest.cli import (
    _resolve_data_source,
    main as cli_main,
    make_kalshi_source,
)


def test_resolve_data_source_kalshi(tmp_path):
    ds = _resolve_data_source("kalshi", cache_root=str(tmp_path))
    assert ds.name == "kalshi"


def test_make_kalshi_source_uses_env_default(monkeypatch, tmp_path):
    monkeypatch.setenv("HLBT_KALSHI_CACHE_ROOT", str(tmp_path))
    ds = make_kalshi_source()
    assert ds.name == "kalshi"


def test_cli_fetch_kalshi_invokes_fetch_and_cache(tmp_path):
    cache_root = tmp_path / "kalshi"
    with patch(
        "hlanalysis.backtest.data.kalshi.KalshiDataSource.fetch_and_cache",
        return_value=[],
    ) as mock_fetch:
        rc = cli_main([
            "fetch", "--data-source", "kalshi",
            "--start", "2025-05-18", "--end", "2026-05-18",
            "--cache-root", str(cache_root),
        ])
    assert rc == 0
    mock_fetch.assert_called_once()
    call_kwargs = mock_fetch.call_args.kwargs
    assert call_kwargs["start"] == "2025-05-18"
    assert call_kwargs["end"] == "2026-05-18"
