"""Read recorder parquet output without ParquetDataset auto-discovery.

The recorder writes Hive-partitioned parquet:
  data/venue=.../product_type=.../mechanism=.../event=.../symbol=.../date=.../hour=.../*.parquet

Calling `pq.read_table(path)` on any of these triggers ParquetDataset's
schema discovery, which then tries to merge the path-derived partition
dictionary type with the in-file string type and fails:

    ArrowTypeError: Unable to merge: Field venue has incompatible types:
    string vs dictionary<values=string, indices=int32, ordered=0>

This module is the single hop everyone should go through to read recorded
data. It opens each file via `ParquetFile` (file mode, no partition
inference) and concatenates with schema promotion.

Usage:
    from hlanalysis.recorder.read import read_recorded

    # Single file
    table = read_recorded("data/venue=polymarket/.../foo.parquet")

    # Glob / list of files
    table = read_recorded("data/venue=polymarket/**/event=trade/**/*.parquet")

    # Pandas
    df = read_recorded(...).to_pandas()
"""

from __future__ import annotations

import glob
from collections.abc import Iterable
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def read_recorded(paths: str | Path | Iterable[str | Path]) -> pa.Table:
    """Read one or more recorder parquet files into a single pyarrow Table.

    Args:
        paths: Path / glob string / iterable of paths. Globs are expanded with
            recursive=True (supports `**`).

    Returns:
        A single concatenated Table. Schema-promoted so mixed-version files
        (e.g. a column added later) load cleanly.

    Raises:
        FileNotFoundError: when no files match the input.
    """
    expanded = _expand(paths)
    if not expanded:
        raise FileNotFoundError(f"no parquet files matched: {paths!r}")
    tables = [pq.ParquetFile(str(p)).read() for p in expanded]
    if len(tables) == 1:
        return tables[0]
    # promote_options="default" handles columns that exist in some files
    # but not others (returns null in the missing files) and resolves dict
    # vs string by promoting to string.
    return pa.concat_tables(tables, promote_options="default")


def _expand(paths: str | Path | Iterable[str | Path]) -> list[Path]:
    if isinstance(paths, (str, Path)):
        return _expand_one(paths)
    out: list[Path] = []
    for p in paths:
        out.extend(_expand_one(p))
    # De-dup preserving order.
    seen: set[str] = set()
    uniq: list[Path] = []
    for p in out:
        s = str(p)
        if s not in seen:
            seen.add(s)
            uniq.append(p)
    return uniq


def _expand_one(p: str | Path) -> list[Path]:
    s = str(p)
    if any(ch in s for ch in "*?["):
        return [Path(m) for m in sorted(glob.glob(s, recursive=True))]
    path = Path(s)
    if path.is_dir():
        return sorted(path.rglob("*.parquet"))
    return [path]
