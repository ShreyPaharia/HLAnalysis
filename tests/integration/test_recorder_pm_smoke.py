"""End-to-end: recorder consuming a fake PolymarketAdapter writes parquet
with the expected schema. No network access; uses a stubbed adapter that
yields a known sequence of normalized events.
"""

from __future__ import annotations

import pyarrow.parquet as pq
import pytest

from hlanalysis.adapters.base import VenueAdapter
from hlanalysis.config import Subscription
from hlanalysis.events import (
    BookSnapshotEvent,
    Mechanism,
    ProductType,
    TradeEvent,
    to_record,
)
from hlanalysis.recorder.writer import ParquetWriter


_PM_TOKEN_ID = "71321045679252212594626385532706912750332728571942532289631379312455583992563"


class _StubAdapter(VenueAdapter):
    venue = "polymarket"

    def supports(self, *a, **k):
        return True

    async def stream(self, _subs):
        now = 1_716_000_000_000_000_000
        yield BookSnapshotEvent(
            venue="polymarket",
            product_type=ProductType.PREDICTION_BINARY,
            mechanism=Mechanism.CLOB,
            symbol=_PM_TOKEN_ID,
            exchange_ts=now,
            local_recv_ts=now,
            bid_px=[0.92],
            bid_sz=[100.0],
            ask_px=[0.93],
            ask_sz=[80.0],
        )
        yield TradeEvent(
            venue="polymarket",
            product_type=ProductType.PREDICTION_BINARY,
            mechanism=Mechanism.CLOB,
            symbol=_PM_TOKEN_ID,
            exchange_ts=now + 1,
            local_recv_ts=now + 1,
            price=0.927,
            size=10.0,
            side="buy",
            trade_id="t1",
        )


@pytest.mark.asyncio
async def test_recorder_writes_pm_book_and_trade(tmp_path):
    writer = ParquetWriter(tmp_path, flush_interval_s=0.01)
    adapter = _StubAdapter()
    sub = Subscription(
        venue="polymarket",
        product_type=ProductType.PREDICTION_BINARY,
        mechanism=Mechanism.CLOB,
        symbol="*",
        channels=("trades", "book"),
    )
    async for ev in adapter.stream([sub]):
        writer.write(to_record(ev))
    writer.flush_all()

    parquet_files = list(tmp_path.rglob("*.parquet"))
    assert parquet_files, "ParquetWriter produced no files"
    # Sanity: at least one file has the expected `symbol` column with the
    # 76-digit token-id string (i.e. ParquetWriter did NOT mis-infer it as int).
    # Use ParquetFile.read() rather than pq.read_table(path) to avoid pyarrow's
    # dataset-partition auto-detection (which mis-merges the in-file `venue`
    # column against the `venue=...` hive partition value).
    for path in parquet_files:
        tbl = pq.ParquetFile(path).read()
        assert "symbol" in tbl.schema.names
        symbols = tbl.column("symbol").to_pylist()
        for s in symbols:
            assert isinstance(s, str), f"symbol column must be string, got {type(s).__name__}={s!r}"
            assert s == _PM_TOKEN_ID
