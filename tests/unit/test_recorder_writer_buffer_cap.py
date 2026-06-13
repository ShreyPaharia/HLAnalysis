from pathlib import Path
import os
import pyarrow as pa  # noqa: F401  (writer imports it)
import pyarrow.parquet as pq
import pytest
from hlanalysis.recorder.writer import ParquetWriter


def _row(sym="BTC", et="trade"):
    return {"venue": "v", "product_type": "p", "mechanism": "m", "event_type": et, "symbol": sym, "exchange_ts": 1}


def test_global_cap_drops_oldest_on_persistent_write_failure(tmp_path: Path, monkeypatch):
    w = ParquetWriter(tmp_path, max_buffer_rows=10, max_total_buffer_rows=25)

    # Force every flush to fail so rows re-buffer (the OOM path).
    import hlanalysis.recorder.writer as wr

    monkeypatch.setattr(wr.pq, "write_table", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))

    # Write far more than the global cap across several keys.
    for i in range(200):
        w.write(_row(sym=f"S{i % 5}"))

    total = sum(len(rows) for rows in w._buffers.values())
    assert total <= 25  # global cap enforced
    assert w.dropped_rows > 0  # and it tracked the drop
    # 200 rows written, cap=25; exactly 175 rows must have been dropped
    assert w.dropped_rows == 200 - total  # dropped == written - surviving


# ---------------------------------------------------------------------------
# Fix 4a: drop escalation — the FIRST drop must emit an ERROR-level log and
# increment dropped_rows_total so operations can detect data loss.
# ---------------------------------------------------------------------------


def test_drop_emits_error_log_and_increments_counter(tmp_path: Path, monkeypatch, caplog):
    """When the global cap is exceeded, the writer must:
    (a) emit a log.error (not just a warning),
    (b) increment the dropped_rows counter.

    The test forces write failure so rows re-buffer, then writes enough rows
    to trip the cap.
    """
    import hlanalysis.recorder.writer as wr
    import logging

    monkeypatch.setattr(wr.pq, "write_table", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))

    w = ParquetWriter(tmp_path, max_buffer_rows=10, max_total_buffer_rows=30)

    with caplog.at_level(logging.ERROR, logger="hlanalysis.recorder.writer"):
        for i in range(200):
            w.write(_row(sym=f"S{i % 5}"))

    # (a) An ERROR-level log was emitted.
    error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert error_records, "expected at least one ERROR log on data drop; only got: " + str(
        [(r.levelno, r.message) for r in caplog.records]
    )
    # The message should mention data loss / dropped rows.
    combined = " ".join(r.message for r in error_records).lower()
    assert "drop" in combined or "loss" in combined, f"ERROR log doesn't mention drop/loss: {combined!r}"

    # (b) The counter was incremented.
    assert w.dropped_rows > 0, "dropped_rows counter was not incremented"


# ---------------------------------------------------------------------------
# Fix 4b: atomic write — _flush_key must write to a .parquet.tmp file first
# and then os.replace() to the final path (atomic on POSIX), so a mid-write
# crash can't leave a truncated file that breaks the reader.
# ---------------------------------------------------------------------------


def test_flush_key_uses_atomic_write(tmp_path: Path, monkeypatch):
    """_flush_key must write to a temp path then rename atomically.

    We monkeypatch os.replace (via the writer module's os reference) to
    capture the arguments, then verify:
    (a) os.replace was called,
    (b) the source is a .parquet.tmp path,
    (c) the destination is a .parquet path (no .tmp),
    (d) no .tmp files remain on disk after flush.
    """
    import hlanalysis.recorder.writer as wr

    replaces: list[tuple] = []
    _orig_replace = os.replace

    def capturing_replace(src, dst):
        replaces.append((str(src), str(dst)))
        _orig_replace(src, dst)

    # Patch the os module reference inside writer so os.replace is intercepted.
    monkeypatch.setattr(wr.os, "replace", capturing_replace)

    w = ParquetWriter(tmp_path, max_buffer_rows=1000)
    # Write 5 rows then explicit flush.
    for _ in range(5):
        w.write(_row())
    w.flush_all()

    assert replaces, "os.replace was never called — write is not atomic"
    src, dst = replaces[0]
    assert src.endswith(".parquet.tmp"), f"temp file should end .parquet.tmp, got {src}"
    assert dst.endswith(".parquet") and not dst.endswith(".parquet.tmp"), f"dest should be .parquet, got {dst}"

    # No leftover .tmp files.
    tmp_files = list(tmp_path.rglob("*.parquet.tmp"))
    assert not tmp_files, f"leftover .tmp files found: {tmp_files}"

    # The final .parquet is readable — open the specific file, not a directory,
    # to avoid pyarrow merging schemas across multiple files.
    parquet_files = list(tmp_path.rglob("*.parquet"))
    assert parquet_files, "no .parquet file found after flush"
    table = pq.ParquetFile(parquet_files[0]).read()
    assert table.num_rows == 5
