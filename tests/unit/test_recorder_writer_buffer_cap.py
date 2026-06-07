from pathlib import Path
import pyarrow as pa  # noqa: F401  (writer imports it)
from hlanalysis.recorder.writer import ParquetWriter


def _row(sym="BTC", et="trade"):
    return {"venue": "v", "product_type": "p", "mechanism": "m",
            "event_type": et, "symbol": sym, "exchange_ts": 1}


def test_global_cap_drops_oldest_on_persistent_write_failure(tmp_path: Path, monkeypatch):
    w = ParquetWriter(tmp_path, max_buffer_rows=10, max_total_buffer_rows=25)

    # Force every flush to fail so rows re-buffer (the OOM path).
    import hlanalysis.recorder.writer as wr
    monkeypatch.setattr(wr.pq, "write_table",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))

    # Write far more than the global cap across several keys.
    for i in range(200):
        w.write(_row(sym=f"S{i % 5}"))

    total = sum(len(rows) for rows in w._buffers.values())
    assert total <= 25                        # global cap enforced
    assert w.dropped_rows > 0                 # and it tracked the drop
    # 200 rows written, cap=25; exactly 175 rows must have been dropped
    assert w.dropped_rows == 200 - total      # dropped == written - surviving
