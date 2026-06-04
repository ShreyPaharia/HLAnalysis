"""`hl-bt run` must produce identical results whether serial or parallel."""
from __future__ import annotations
from pathlib import Path
import pytest
from hlanalysis.backtest.cli import main as cli_main

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "hl_hip4"
CONFIG = Path(__file__).resolve().parents[2] / "fixtures" / "configs" / "v1-smoke.json"


def _run(tmp: Path, workers: int) -> str:
    out = tmp / f"w{workers}"
    rc = cli_main([
        "run", "--strategy", "v1_late_resolution",
        "--data-source", "hl_hip4", "--config", str(CONFIG),
        "--cache-root", str(FIXTURE),
        "--out-dir", str(out), "--start", "2026-05-09", "--end", "2026-05-11",
        "--kind", "binary", "--max-markets", "2", "--workers", str(workers),
    ])
    assert rc == 0
    return (out / "report.md").read_text()


@pytest.mark.skipif(not FIXTURE.exists(), reason="hl_hip4 fixture missing")
def test_serial_and_parallel_reports_match(tmp_path):
    assert _run(tmp_path, 1) == _run(tmp_path, 2)
