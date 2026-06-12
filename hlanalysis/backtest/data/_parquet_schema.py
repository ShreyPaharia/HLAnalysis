"""Parquet column-name constants for the recorder's event partitions.

The recorder writes one parquet file per event type (book_snapshot, trade,
bbo, mark, settlement, …) via ``hlanalysis.events.to_record()``. The sim's
data loaders (hl_hip4, _hl_hip4_fastpath, _pm_fastpath, polymarket) then
SELECT specific columns from those files by name.

This module is the single source of truth for which columns each event type
writes and which subset the loaders SELECT. It does NOT change the parquet
schema (that stays defined by ``NormalizedEvent`` subclasses + ``to_record``)
— it only names the columns so that:

  1. A round-trip test can assert ``to_record(ev).keys()`` contains every
     column listed here (catching a rename in ``events.py`` that breaks
     loaders silently).
  2. Loader code can reference ``BOOK_SNAPSHOT_LOADER_COLS`` etc. instead of
     repeating the string literals, making future renames a one-place change.

The constants are intentionally NOT imported by the hot-path loaders today —
that would add an import with no perf benefit. They are the reference for
tests and for any future loader that wants to be explicit about its contract.

Column subsets vs. full schema
-------------------------------
``*_LOADER_COLS`` lists only the columns a loader actually SELECTs (the
minimal subset needed for simulation). The full event schema has additional
columns (e.g. ``local_recv_ts``, ``seq``, ``trade_id``, …) that the sim does
not need. Listing the loader subset keeps the contract tight — it fails the
moment a needed column is missing, not when an unused one is renamed.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Base columns present in every event type (from _BaseEvent)
# ---------------------------------------------------------------------------

#: Columns common to every NormalizedEvent parquet partition.
BASE_COLS: tuple[str, ...] = (
    "venue",
    "product_type",
    "mechanism",
    "symbol",
    "exchange_ts",
    "local_recv_ts",
    "seq",
    "event_type",
)

# ---------------------------------------------------------------------------
# Full event schemas (all columns written by to_record)
# ---------------------------------------------------------------------------

#: All columns written by BookSnapshotEvent → to_record.
BOOK_SNAPSHOT_COLS: tuple[str, ...] = BASE_COLS + (
    "bid_px",
    "bid_sz",
    "ask_px",
    "ask_sz",
)

#: All columns written by TradeEvent → to_record.
TRADE_COLS: tuple[str, ...] = BASE_COLS + (
    "price",
    "size",
    "side",
    "trade_id",
    "block_ts",
    "buyer",
    "seller",
    "block_hash",
)

#: All columns written by BboEvent → to_record.
BBO_COLS: tuple[str, ...] = BASE_COLS + (
    "bid_px",
    "bid_sz",
    "ask_px",
    "ask_sz",
)

#: All columns written by MarkEvent → to_record.
MARK_COLS: tuple[str, ...] = BASE_COLS + (
    "mark_px",
)

#: All columns written by SettlementEvent → to_record.
SETTLEMENT_COLS: tuple[str, ...] = BASE_COLS + (
    "settled_side_idx",
    "settle_price",
    "settle_ts",
    "keys",
    "values",
)

#: All columns written by QuestionMetaEvent → to_record.
QUESTION_META_COLS: tuple[str, ...] = BASE_COLS + (
    "question_idx",
    "named_outcome_idxs",
    "fallback_outcome_idx",
    "settled_named_outcome_idxs",
    "keys",
    "values",
)

# ---------------------------------------------------------------------------
# Loader subsets — what the sim's data loaders actually SELECT
# ---------------------------------------------------------------------------

#: Columns SELECTed from book_snapshot partitions by the sim loaders.
#: Source: hl_hip4._book_iter, _hl_hip4_fastpath._read_book_columns,
#:         _pm_fastpath.read_pm_book_columns.
BOOK_SNAPSHOT_LOADER_COLS: tuple[str, ...] = (
    "exchange_ts",
    "bid_px",
    "bid_sz",
    "ask_px",
    "ask_sz",
)

#: Columns SELECTed from trade partitions by the sim loaders.
#: Source: hl_hip4._trade_iter, _hl_hip4_fastpath._read_trade_columns.
TRADE_LOADER_COLS: tuple[str, ...] = (
    "exchange_ts",
    "price",
    "size",
    "side",
)

#: Columns SELECTed from bbo (reference) partitions by the sim loaders.
#: Source: hl_hip4._reference_rows (bbo branch).
BBO_REFERENCE_LOADER_COLS: tuple[str, ...] = (
    "exchange_ts",
    "bid_px",
    "ask_px",
)

#: Columns SELECTed from mark (reference) partitions by the sim loaders.
#: Source: hl_hip4._reference_rows (mark branch).
MARK_REFERENCE_LOADER_COLS: tuple[str, ...] = (
    "exchange_ts",
    "mark_px",
)

#: Columns SELECTed from settlement partitions by the sim loaders.
#: Source: hl_hip4._settlement_iter, _hl_hip4_fastpath._read_settlement_columns.
SETTLEMENT_LOADER_COLS: tuple[str, ...] = (
    "exchange_ts",
    "settle_ts",
    "settled_side_idx",
)

#: Columns SELECTed from question_meta partitions by hl_hip4.discover.
QUESTION_META_LOADER_COLS: tuple[str, ...] = (
    "symbol",
    "question_idx",
    "named_outcome_idxs",
    "fallback_outcome_idx",
    "keys",
    "values",
    "exchange_ts",
)


__all__ = [
    "BASE_COLS",
    "BOOK_SNAPSHOT_COLS",
    "TRADE_COLS",
    "BBO_COLS",
    "MARK_COLS",
    "SETTLEMENT_COLS",
    "QUESTION_META_COLS",
    "BOOK_SNAPSHOT_LOADER_COLS",
    "TRADE_LOADER_COLS",
    "BBO_REFERENCE_LOADER_COLS",
    "MARK_REFERENCE_LOADER_COLS",
    "SETTLEMENT_LOADER_COLS",
    "QUESTION_META_LOADER_COLS",
]
