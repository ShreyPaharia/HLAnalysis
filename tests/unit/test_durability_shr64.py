"""SHR-64: ParquetWriter._flush_key must fsync the written file after os.replace
so OOM-kills and power-loss can't drop un-fsync'd buffered rows.

Tests:
  - flush calls os.fsync at least once on the written file descriptor
  - normal happy-path (data readable after flush) is unchanged
  - fsync also fires on the row-count auto-flush path
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pyarrow.parquet as pq

from hlanalysis.recorder.writer import ParquetWriter


def _row(sym: str = "BTC") -> dict:
    return {
        "venue": "hl",
        "product_type": "binary",
        "mechanism": "hip4",
        "event_type": "trade",
        "symbol": sym,
        "exchange_ts": 1_000_000_000,
    }


def test_flush_fsyncs_written_file(tmp_path: Path):
    """After _flush_key, os.fsync must have been called at least once so that
    the flushed parquet file is durable on disk."""
    synced_fds: list[int] = []

    orig_fsync = os.fsync

    def spy_fsync(fd: int) -> None:
        synced_fds.append(fd)
        orig_fsync(fd)

    import hlanalysis.recorder.writer as wr

    with patch.object(wr.os, "fsync", side_effect=spy_fsync):
        w = ParquetWriter(tmp_path, max_buffer_rows=1000)
        for _ in range(5):
            w.write(_row())
        w.flush_all()

    assert synced_fds, (
        "os.fsync was never called after flush; un-fsync'd parquet data would "
        "be lost on OOM-kill or power-loss (SHR-64)"
    )


def test_flush_data_readable_after_flush(tmp_path: Path):
    """Happy-path: after flush_all, the written parquet files are complete and
    readable. This ensures the fsync addition doesn't corrupt the write path.
    Note: the writer partitions by symbol, so 7 distinct symbols produce up to
    7 separate partition files; we check total rows across all of them."""
    w = ParquetWriter(tmp_path, max_buffer_rows=1000)
    # Use the same symbol so all rows land in one file
    for i in range(7):
        w.write(_row(sym="BTC"))
    w.flush_all()

    parquet_files = list(tmp_path.rglob("*.parquet"))
    assert parquet_files, "no .parquet file written after flush_all"
    total_rows = sum(pq.ParquetFile(f).read().num_rows for f in parquet_files)
    assert total_rows == 7


def test_fsync_called_on_row_count_trigger(tmp_path: Path):
    """os.fsync must also fire on the row-count-triggered auto-flush path
    (not just explicit flush_all), since that path calls _flush_key directly."""
    synced_fds: list[int] = []

    orig_fsync = os.fsync

    def spy_fsync(fd: int) -> None:
        synced_fds.append(fd)
        orig_fsync(fd)

    import hlanalysis.recorder.writer as wr

    with patch.object(wr.os, "fsync", side_effect=spy_fsync):
        # max_buffer_rows=3 → flush triggers automatically after 3rd write
        w = ParquetWriter(tmp_path, max_buffer_rows=3)
        for _ in range(3):
            w.write(_row())

    assert synced_fds, "os.fsync not called on row-count-triggered flush (SHR-64)"
