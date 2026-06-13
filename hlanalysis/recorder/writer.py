from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

log = logging.getLogger(__name__)


def _hour_bucket(ts_ns: int) -> tuple[str, str]:
    dt = datetime.fromtimestamp(ts_ns / 1e9, tz=UTC)
    return dt.strftime("%Y-%m-%d"), dt.strftime("%H")


class ParquetWriter:
    """Buffered, partitioned parquet writer.

    Partition layout:
      data/venue=.../product_type=.../mechanism=.../event=.../symbol=.../date=YYYY-MM-DD/hour=HH/*.parquet

    Buffers events per (venue, product_type, mechanism, event_type, symbol, date, hour) and flushes
    when either the row count or the time-since-last-flush threshold is exceeded. The hour-keyed
    partition guarantees that no parquet file ever spans an hour boundary, even if the recorder
    has been buffering across the boundary.
    """

    def __init__(
        self,
        root: Path,
        flush_interval_s: float = 30.0,
        max_buffer_rows: int = 5000,
        max_total_buffer_rows: int = 500_000,  # global backstop (SHR-63)
    ) -> None:
        self.root = Path(root)
        self.flush_interval_s = flush_interval_s
        self.max_buffer_rows = max_buffer_rows
        self.max_total_buffer_rows = max_total_buffer_rows
        self.dropped_rows = 0
        self._buffers: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
        self._last_flush = time.monotonic()
        self._flush_seq = 0

    def write(self, event_dict: dict[str, Any]) -> None:
        ts_ns = event_dict.get("exchange_ts") or event_dict.get("local_recv_ts") or time.time_ns()
        date, hour = _hour_bucket(ts_ns)
        key = (
            event_dict["venue"],
            event_dict["product_type"],
            event_dict["mechanism"],
            event_dict["event_type"],
            event_dict["symbol"],
            date,
            hour,
        )
        self._buffers[key].append(event_dict)
        self._enforce_global_cap()
        if len(self._buffers[key]) >= self.max_buffer_rows:
            self._flush_key(key)

    def maybe_flush(self) -> None:
        if time.monotonic() - self._last_flush > self.flush_interval_s:
            self.flush_all()

    def flush_all(self) -> None:
        for key in list(self._buffers.keys()):
            self._flush_key(key)
        self._last_flush = time.monotonic()

    def _flush_key(self, key: tuple) -> None:
        rows = self._buffers.pop(key, None)
        if not rows:
            return
        venue, product_type, mechanism, event_type, symbol, date, hour = key
        partition = (
            self.root
            / f"venue={venue}"
            / f"product_type={product_type}"
            / f"mechanism={mechanism}"
            / f"event={event_type}"
            / f"symbol={self._safe_symbol(symbol)}"
            / f"date={date}"
            / f"hour={hour}"
        )
        partition.mkdir(parents=True, exist_ok=True)
        self._flush_seq += 1
        path = partition / f"{int(time.time() * 1000)}-{self._flush_seq:06d}.parquet"
        tmp_path = path.with_suffix(".parquet.tmp")
        try:
            table = pa.Table.from_pylist(rows)
            pq.write_table(table, tmp_path, compression="zstd")
            # SHR-64: fsync the temp file before the atomic rename so that
            # OOM-kills and power-loss after os.replace can't silently lose
            # un-fsync'd rows. High-value, low-volume keys (settlement,
            # strike-capture, question meta) are most at risk because they may
            # sit in the buffer for the full flush_interval_s. We open the
            # already-written tmp path in append mode (no truncation) purely to
            # obtain a file descriptor for fsync, then close it immediately.
            with open(tmp_path, "ab") as _fd:
                os.fsync(_fd.fileno())
            os.replace(tmp_path, path)
        except Exception:
            log.exception("failed to write %s (%d rows)", path, len(rows))
            # Clean up any partially written temp file so globs don't see it.
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            # Re-buffer so we don't lose the events; subsequent flush may succeed.
            self._buffers[key].extend(rows)
            self._enforce_global_cap()

    def _enforce_global_cap(self) -> None:
        """SHR-63: on a persistent write failure _flush_key re-buffers rows; on
        the OOMScore=-500 recorder that would OOM-kill the +500 live engine.
        Cap total buffered rows across all keys; drop the OLDEST rows (FIFO) and
        count them so a monitor/alert can see data loss instead of an OOM."""
        total = sum(len(r) for r in self._buffers.values())
        if total <= self.max_total_buffer_rows:
            return
        # Batch-drop oldest-first across keys until under the cap.  Buffers are
        # append-ordered so index 0 of each key list is oldest for that key.
        # Drain from the largest key first (bounded heuristic) to minimise the
        # number of max() scans.  Use slice deletion (O(n) once) rather than
        # pop(0) in a loop (O(n²) overall).
        overflow = total - self.max_total_buffer_rows
        dropped_this_call = 0
        while overflow > 0 and self._buffers:
            # Remove empty keys left over from prior iterations.
            empty = [k for k, v in self._buffers.items() if not v]
            for k in empty:
                del self._buffers[k]
            if not self._buffers:
                break
            key = max(self._buffers, key=lambda k: len(self._buffers[k]))
            rows = self._buffers[key]
            drop_k = min(overflow, len(rows))
            del rows[:drop_k]
            dropped_this_call += drop_k
            overflow -= drop_k
            if not rows:
                del self._buffers[key]
        self.dropped_rows += dropped_this_call
        if dropped_this_call:
            log.error(
                "RECORDER DATA LOSS: buffer cap hit; dropped=%d this_call,"
                " dropped_rows_total=%d — persistent write failure or runaway buffer;"
                " market-data records are being discarded",
                dropped_this_call,
                self.dropped_rows,
            )

    @staticmethod
    def _safe_symbol(symbol: str) -> str:
        # Hive partition values can't contain '/' or '='. HIP-4 binary symbols use only
        # ascii + dashes, but spot symbols may contain '/'.
        return symbol.replace("/", "_").replace("=", "_")
