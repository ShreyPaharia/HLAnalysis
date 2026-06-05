"""Regression guard for the daily-compaction layout (scripts/compact-data.sh).

Daily compaction writes one file per sealed day under a sentinel `hour=all`
partition:

    .../date=YYYY-MM-DD/hour=all/compacted.parquet

instead of the recorder's per-hour `hour=HH/*.parquet`. The sentinel exists so
that legacy hourly files and new daily files coexist under the unchanged
`.../date=*/hour=*/*.parquet` glob that every analysis/backtest reader uses:
dropping the `hour=` level entirely makes DuckDB raise "Hive partition
mismatch" when daily files sit next to legacy hourly ones.

These tests pin that contract: the standard reader path must read a mixed
hourly + daily tree, and the daily file's `hour` partition value is the
harmless string "all".
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from hlanalysis.analysis.helpers import glob_for
from hlanalysis.recorder.read import read_recorded


def _write(path: Path, n: int, px0: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ts": range(n), "px": [px0 + i for i in range(n)]}).to_parquet(path)


def _tree(root: Path) -> Path:
    """A single stream with one legacy hourly partition and one daily one."""
    stream = (
        root
        / "venue=test"
        / "product_type=perp"
        / "mechanism=clob"
        / "event=trade"
        / "symbol=BTC"
        / "date=2026-06-04"
    )
    _write(stream / "hour=05" / "compacted.parquet", 3, 100.0)  # legacy hourly
    _write(stream / "hour=all" / "compacted.parquet", 4, 200.0)  # daily-compacted
    return stream


def test_duckdb_hive_glob_reads_mixed_hourly_and_daily(tmp_path: Path, monkeypatch) -> None:
    import hlanalysis.analysis.helpers as helpers

    monkeypatch.setattr(helpers, "DATA_ROOT", tmp_path)
    _tree(tmp_path)

    glob = glob_for(venue="test", product_type="perp", event="trade", symbol="BTC")
    df = duckdb.connect().execute(
        f"SELECT hour, count(*) AS n FROM read_parquet('{glob}', hive_partitioning=true) "
        "GROUP BY hour ORDER BY hour"
    ).df()

    # Both partitions are read through the single unchanged glob.
    assert dict(zip(df["hour"], df["n"])) == {"05": 3, "all": 4}


def test_read_recorded_reads_daily_file(tmp_path: Path) -> None:
    stream = _tree(tmp_path)
    table = read_recorded(str(stream / "**" / "*.parquet"))
    # 3 hourly + 4 daily rows, depth-agnostic recursive read.
    assert table.num_rows == 7
