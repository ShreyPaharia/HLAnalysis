"""End-to-end CLI smoke test against the synthetic data source.

Acceptance criteria (Task A §4):
- ``hl-bt run --strategy <id> --data-source synthetic --config <json>
  --out-dir <dir>`` produces ``report.md``, ``fills.parquet``,
  ``diagnostics.parquet`` in ``--out-dir``.
- ``hl-bt strategies`` lists registered strategies.

The real ``v1_late_resolution`` won't be in the registry until Task E wires
``@register(...)`` on the strategy modules. We use the dummy enter strategy
shipped with the synthetic data source as the substitute.
"""
from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from hlanalysis.backtest.cli import main as cli_main


def test_synthetic_run_smoke(tmp_path: Path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"size": 10.0}))
    out_dir = tmp_path / "out"
    rc = cli_main(
        [
            "run",
            "--strategy",
            "_dummy_enter_yes",
            "--data-source",
            "synthetic",
            "--config",
            str(config_path),
            "--out-dir",
            str(out_dir),
        ]
    )
    assert rc == 0

    # All three top-level artifacts present.
    assert (out_dir / "report.md").exists()
    assert (out_dir / "fills.parquet").exists()
    assert (out_dir / "diagnostics.parquet").exists()

    # Report mentions the strategy id.
    report = (out_dir / "report.md").read_text()
    assert "_dummy_enter_yes" in report

    # Fills carry at least one ENTER and one settlement row.
    fills = pq.read_table(out_dir / "fills.parquet").to_pydict()
    assert "settle" in fills["cloid"]
    assert any(c != "settle" for c in fills["cloid"])

    # Diagnostics: ts ascending.
    diag = pq.read_table(out_dir / "diagnostics.parquet").to_pydict()
    assert diag["ts_ns"] == sorted(diag["ts_ns"])


def test_strategies_subcommand_lists_registry(capsys):
    rc = cli_main(["strategies"])
    assert rc == 0
    captured = capsys.readouterr()
    # Registry may be empty until Task E wires @register decorators. Either
    # the empty marker or one-or-more ids is acceptable.
    out = captured.out.strip()
    assert out == "(no strategies registered)" or all(
        line and " " not in line for line in out.splitlines()
    )
