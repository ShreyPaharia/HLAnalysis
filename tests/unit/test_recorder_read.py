"""Tests for hlanalysis.recorder.read.read_recorded.

The motivating bug (caught 2026-05-26): calling pq.read_table on a Hive-
partitioned recorder file triggers ParquetDataset auto-discovery, which
fails to merge the path-derived partition dict type with the in-file
string type. read_recorded uses ParquetFile (no partition inference) so
this is bypassed.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from hlanalysis.recorder.read import read_recorded


def _write_partition(root: Path, venue: str, event: str, symbol: str, rows: list[dict]) -> Path:
    """Mirror the recorder's partitioning shape."""
    partition = (
        root
        / f"venue={venue}"
        / f"product_type=prediction_binary"
        / f"mechanism=clob"
        / f"event={event}"
        / f"symbol={symbol}"
        / f"date=2026-05-26"
        / f"hour=03"
    )
    partition.mkdir(parents=True, exist_ok=True)
    path = partition / f"test-{event}.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def test_read_recorded_single_file(tmp_path):
    rows = [
        {"venue": "polymarket", "symbol": "tok-yes", "price": 0.62, "size": 5.0},
        {"venue": "polymarket", "symbol": "tok-yes", "price": 0.63, "size": 3.0},
    ]
    f = _write_partition(tmp_path, "polymarket", "trade", "tok-yes", rows)
    table = read_recorded(f)
    assert table.num_rows == 2
    assert table.column("price").to_pylist() == [0.62, 0.63]


def test_read_recorded_does_not_trip_partition_type_mismatch(tmp_path):
    """The bug we're working around: pq.read_table on a partitioned path
    fails with 'Field venue has incompatible types: string vs dictionary'.
    Confirm read_recorded does NOT raise."""
    rows = [{"venue": "polymarket", "symbol": "tok-yes", "v": 1}]
    f = _write_partition(tmp_path, "polymarket", "trade", "tok-yes", rows)
    # First: confirm the partition path is what triggers the issue. If the
    # underlying pyarrow version no longer raises, this still passes — the
    # helper just has nothing to work around. Either way our function works.
    try:
        pq.read_table(f)
        had_native_error = False
    except pa.ArrowTypeError:
        had_native_error = True
    table = read_recorded(f)
    assert table.num_rows == 1
    # Sanity: regardless of which pyarrow path is taken, the helper succeeds.
    # The boolean is recorded so the test self-documents the workaround.
    _ = had_native_error


def test_read_recorded_glob_concatenates(tmp_path):
    _write_partition(tmp_path, "polymarket", "trade", "tok-yes", [{"venue": "polymarket", "price": 0.5, "size": 1.0}])
    _write_partition(
        tmp_path,
        "polymarket",
        "trade",
        "tok-no",
        [{"venue": "polymarket", "price": 0.5, "size": 2.0}, {"venue": "polymarket", "price": 0.5, "size": 3.0}],
    )
    table = read_recorded(str(tmp_path) + "/venue=polymarket/**/event=trade/**/*.parquet")
    assert table.num_rows == 3


def test_read_recorded_directory_walks_recursively(tmp_path):
    _write_partition(tmp_path, "polymarket", "trade", "tok-a", [{"venue": "polymarket", "price": 0.5}])
    _write_partition(tmp_path, "binance", "trade", "BTCUSDT", [{"venue": "binance", "price": 60000.0}])
    # promote_options="default" lets the missing column become null.
    table = read_recorded(tmp_path)
    assert table.num_rows == 2


def test_read_recorded_raises_on_no_matches(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_recorded(tmp_path / "does-not-exist" / "*.parquet")


def test_read_recorded_iterable_of_paths(tmp_path):
    f1 = _write_partition(tmp_path, "polymarket", "trade", "a", [{"venue": "polymarket", "v": 1}])
    f2 = _write_partition(tmp_path, "polymarket", "trade", "b", [{"venue": "polymarket", "v": 2}])
    table = read_recorded([f1, f2])
    assert table.num_rows == 2
    assert sorted(table.column("v").to_pylist()) == [1, 2]
