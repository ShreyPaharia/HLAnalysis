from __future__ import annotations

import logging
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

log = logging.getLogger(__name__)


def _hour_bucket(ts_ns: int) -> tuple[str, str]:
    dt = datetime.fromtimestamp(ts_ns / 1e9, tz=timezone.utc)
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
    ) -> None:
        self.root = Path(root)
        self.flush_interval_s = flush_interval_s
        self.max_buffer_rows = max_buffer_rows
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
        try:
            table = pa.Table.from_pylist(rows)
            pq.write_table(table, path, compression="zstd")
        except Exception:
            log.exception("failed to write %s (%d rows)", path, len(rows))
            # Re-buffer so we don't lose the events; subsequent flush may succeed.
            self._buffers[key].extend(rows)

    @staticmethod
    def _safe_symbol(symbol: str) -> str:
        # Hive partition values can't contain '/' or '='. HIP-4 binary symbols use only
        # ascii + dashes, but spot symbols may contain '/'.
        return symbol.replace("/", "_").replace("=", "_")
