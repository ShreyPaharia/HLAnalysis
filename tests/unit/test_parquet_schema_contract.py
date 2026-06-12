"""Round-trip contract: recorder column names vs. loader column expectations.

The recorder writes event parquet via ``to_record(ev)``.  The sim's data
loaders SELECT specific columns from those files by hardcoded name strings.

This test ensures that every column a loader expects to SELECT actually
exists in the recorder's output — i.e., a rename in ``events.py`` that
would cause a silent ``KeyError`` or DuckDB ``Referenced column not found``
at query time is caught here at import time.

Two contracts are checked for each event type:
  1. Full-schema round-trip: every column in ``*_COLS`` is present in
     ``to_record(ev)`` (no column silently dropped from the schema).
  2. Loader-subset round-trip: every column in ``*_LOADER_COLS`` is present
     in the full ``*_COLS`` (loaders can't SELECT something the recorder
     doesn't write).

Together these guarantee: if recorder renames a column that a loader reads,
at least one of these tests fails loudly before any parquet file is written
or queried.
"""
from __future__ import annotations

import pytest

from hlanalysis.events import (
    BboEvent,
    BookSnapshotEvent,
    MarkEvent,
    Mechanism,
    ProductType,
    QuestionMetaEvent,
    SettlementEvent,
    TradeEvent,
    to_record,
)
from hlanalysis.backtest.data._parquet_schema import (
    BASE_COLS,
    BBO_COLS,
    BBO_REFERENCE_LOADER_COLS,
    BOOK_SNAPSHOT_COLS,
    BOOK_SNAPSHOT_LOADER_COLS,
    MARK_COLS,
    MARK_REFERENCE_LOADER_COLS,
    QUESTION_META_COLS,
    QUESTION_META_LOADER_COLS,
    SETTLEMENT_COLS,
    SETTLEMENT_LOADER_COLS,
    TRADE_COLS,
    TRADE_LOADER_COLS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_kwargs() -> dict:
    """Minimal base fields shared by every event constructor."""
    return dict(
        venue="hyperliquid",
        product_type=ProductType.PREDICTION_BINARY,
        mechanism=Mechanism.CLOB,
        symbol="TEST",
        exchange_ts=1_000_000_000,
        local_recv_ts=1_000_000_001,
    )


# ---------------------------------------------------------------------------
# Full-schema round-trip: to_record() must contain every constant column
# ---------------------------------------------------------------------------

class TestFullSchemaRoundTrip:
    """Every column listed in *_COLS must appear in to_record() output."""

    def test_book_snapshot_cols_in_record(self):
        ev = BookSnapshotEvent(
            **_base_kwargs(),
            bid_px=[0.49], bid_sz=[100.0],
            ask_px=[0.51], ask_sz=[100.0],
        )
        record_keys = set(to_record(ev).keys())
        missing = set(BOOK_SNAPSHOT_COLS) - record_keys
        assert not missing, (
            f"BOOK_SNAPSHOT_COLS references columns absent from to_record(): {missing!r}. "
            "Update _parquet_schema.BOOK_SNAPSHOT_COLS to match events.BookSnapshotEvent."
        )

    def test_trade_cols_in_record(self):
        ev = TradeEvent(
            **_base_kwargs(),
            price=0.50, size=10.0, side="buy",
        )
        record_keys = set(to_record(ev).keys())
        missing = set(TRADE_COLS) - record_keys
        assert not missing, (
            f"TRADE_COLS references columns absent from to_record(): {missing!r}. "
            "Update _parquet_schema.TRADE_COLS to match events.TradeEvent."
        )

    def test_bbo_cols_in_record(self):
        ev = BboEvent(
            **_base_kwargs(),
            bid_px=0.49, bid_sz=50.0, ask_px=0.51, ask_sz=50.0,
        )
        record_keys = set(to_record(ev).keys())
        missing = set(BBO_COLS) - record_keys
        assert not missing, (
            f"BBO_COLS references columns absent from to_record(): {missing!r}. "
            "Update _parquet_schema.BBO_COLS to match events.BboEvent."
        )

    def test_mark_cols_in_record(self):
        ev = MarkEvent(**_base_kwargs(), mark_px=50000.0)
        record_keys = set(to_record(ev).keys())
        missing = set(MARK_COLS) - record_keys
        assert not missing, (
            f"MARK_COLS references columns absent from to_record(): {missing!r}. "
            "Update _parquet_schema.MARK_COLS to match events.MarkEvent."
        )

    def test_settlement_cols_in_record(self):
        ev = SettlementEvent(
            **_base_kwargs(),
            settled_side_idx=0, settle_price=1.0, settle_ts=1_100_000_000,
        )
        record_keys = set(to_record(ev).keys())
        missing = set(SETTLEMENT_COLS) - record_keys
        assert not missing, (
            f"SETTLEMENT_COLS references columns absent from to_record(): {missing!r}. "
            "Update _parquet_schema.SETTLEMENT_COLS to match events.SettlementEvent."
        )

    def test_question_meta_cols_in_record(self):
        ev = QuestionMetaEvent(
            **_base_kwargs(),
            question_idx=42,
            named_outcome_idxs=[0, 1],
        )
        record_keys = set(to_record(ev).keys())
        missing = set(QUESTION_META_COLS) - record_keys
        assert not missing, (
            f"QUESTION_META_COLS references columns absent from to_record(): {missing!r}. "
            "Update _parquet_schema.QUESTION_META_COLS to match events.QuestionMetaEvent."
        )

    def test_base_cols_present_in_every_event_type(self):
        """BASE_COLS must appear in every concrete event's to_record output."""
        events = [
            TradeEvent(**_base_kwargs(), price=0.5, size=1.0, side="buy"),
            BookSnapshotEvent(**_base_kwargs(), bid_px=[0.49], bid_sz=[1.0], ask_px=[0.51], ask_sz=[1.0]),
            BboEvent(**_base_kwargs(), bid_px=0.49, bid_sz=1.0, ask_px=0.51, ask_sz=1.0),
            MarkEvent(**_base_kwargs(), mark_px=50000.0),
            SettlementEvent(**_base_kwargs(), settled_side_idx=0, settle_ts=1_100_000_000),
            QuestionMetaEvent(**_base_kwargs(), question_idx=1, named_outcome_idxs=[0]),
        ]
        for ev in events:
            record_keys = set(to_record(ev).keys())
            missing = set(BASE_COLS) - record_keys
            assert not missing, (
                f"{type(ev).__name__}: BASE_COLS {missing!r} absent from to_record(). "
                "Update _parquet_schema.BASE_COLS."
            )


# ---------------------------------------------------------------------------
# Loader-subset contract: loader columns must be a subset of the full schema
# ---------------------------------------------------------------------------

class TestLoaderSubsetContract:
    """Every column a loader SELECTs must exist in the full event schema."""

    def test_book_snapshot_loader_cols_subset_of_full(self):
        extra = set(BOOK_SNAPSHOT_LOADER_COLS) - set(BOOK_SNAPSHOT_COLS)
        assert not extra, (
            f"BOOK_SNAPSHOT_LOADER_COLS contains columns not in BOOK_SNAPSHOT_COLS: {extra!r}. "
            "The loader would receive a DuckDB 'Referenced column not found' error."
        )

    def test_trade_loader_cols_subset_of_full(self):
        extra = set(TRADE_LOADER_COLS) - set(TRADE_COLS)
        assert not extra, (
            f"TRADE_LOADER_COLS contains columns not in TRADE_COLS: {extra!r}. "
            "The loader would receive a DuckDB 'Referenced column not found' error."
        )

    def test_bbo_reference_loader_cols_subset_of_full(self):
        extra = set(BBO_REFERENCE_LOADER_COLS) - set(BBO_COLS)
        assert not extra, (
            f"BBO_REFERENCE_LOADER_COLS contains columns not in BBO_COLS: {extra!r}. "
            "The loader would receive a DuckDB 'Referenced column not found' error."
        )

    def test_mark_reference_loader_cols_subset_of_full(self):
        extra = set(MARK_REFERENCE_LOADER_COLS) - set(MARK_COLS)
        assert not extra, (
            f"MARK_REFERENCE_LOADER_COLS contains columns not in MARK_COLS: {extra!r}. "
            "The loader would receive a DuckDB 'Referenced column not found' error."
        )

    def test_settlement_loader_cols_subset_of_full(self):
        extra = set(SETTLEMENT_LOADER_COLS) - set(SETTLEMENT_COLS)
        assert not extra, (
            f"SETTLEMENT_LOADER_COLS contains columns not in SETTLEMENT_COLS: {extra!r}. "
            "The loader would receive a DuckDB 'Referenced column not found' error."
        )

    def test_question_meta_loader_cols_subset_of_full(self):
        extra = set(QUESTION_META_LOADER_COLS) - set(QUESTION_META_COLS)
        assert not extra, (
            f"QUESTION_META_LOADER_COLS contains columns not in QUESTION_META_COLS: {extra!r}. "
            "The loader would receive a DuckDB 'Referenced column not found' error."
        )


# ---------------------------------------------------------------------------
# Exact value verification: ensure loader constants match the actual SQL used
# ---------------------------------------------------------------------------

class TestLoaderColsMatchActualSQL:
    """Cross-check _parquet_schema constants against the literal column names
    used in the loaders' SQL SELECT statements. This catches the case where
    a constant is updated but the loader's SQL is not (or vice versa).

    Each loader's column usage is verified by importing the module and checking
    that the SQL strings embedded in the loader contain each required column.
    This is a string-search based approach — robust enough to catch a rename
    in either direction without requiring a full DuckDB execution.
    """

    def test_hl_hip4_book_loader_uses_book_snapshot_loader_cols(self):
        import inspect
        from hlanalysis.backtest.data import hl_hip4
        src = inspect.getsource(hl_hip4)
        for col in BOOK_SNAPSHOT_LOADER_COLS:
            assert col in src, (
                f"Column {col!r} from BOOK_SNAPSHOT_LOADER_COLS not found in hl_hip4.py source. "
                "Either the loader no longer SELECTs this column (update the constant) "
                "or the column was renamed in events.py (update both)."
            )

    def test_hl_hip4_trade_loader_uses_trade_loader_cols(self):
        import inspect
        from hlanalysis.backtest.data import hl_hip4
        src = inspect.getsource(hl_hip4)
        for col in TRADE_LOADER_COLS:
            assert col in src, (
                f"Column {col!r} from TRADE_LOADER_COLS not found in hl_hip4.py source."
            )

    def test_hl_hip4_bbo_reference_loader_uses_bbo_reference_cols(self):
        import inspect
        from hlanalysis.backtest.data import hl_hip4
        src = inspect.getsource(hl_hip4)
        for col in BBO_REFERENCE_LOADER_COLS:
            assert col in src, (
                f"Column {col!r} from BBO_REFERENCE_LOADER_COLS not found in hl_hip4.py source."
            )

    def test_hl_hip4_mark_reference_loader_uses_mark_reference_cols(self):
        import inspect
        from hlanalysis.backtest.data import hl_hip4
        src = inspect.getsource(hl_hip4)
        for col in MARK_REFERENCE_LOADER_COLS:
            assert col in src, (
                f"Column {col!r} from MARK_REFERENCE_LOADER_COLS not found in hl_hip4.py source."
            )

    def test_hl_hip4_settlement_loader_uses_settlement_loader_cols(self):
        import inspect
        from hlanalysis.backtest.data import hl_hip4
        src = inspect.getsource(hl_hip4)
        for col in SETTLEMENT_LOADER_COLS:
            assert col in src, (
                f"Column {col!r} from SETTLEMENT_LOADER_COLS not found in hl_hip4.py source."
            )

    def test_fastpath_book_loader_uses_book_snapshot_loader_cols(self):
        import inspect
        from hlanalysis.backtest.data import _hl_hip4_fastpath
        src = inspect.getsource(_hl_hip4_fastpath)
        for col in BOOK_SNAPSHOT_LOADER_COLS:
            assert col in src, (
                f"Column {col!r} from BOOK_SNAPSHOT_LOADER_COLS not found in _hl_hip4_fastpath.py."
            )

    def test_fastpath_trade_loader_uses_trade_loader_cols(self):
        import inspect
        from hlanalysis.backtest.data import _hl_hip4_fastpath
        src = inspect.getsource(_hl_hip4_fastpath)
        for col in TRADE_LOADER_COLS:
            assert col in src, (
                f"Column {col!r} from TRADE_LOADER_COLS not found in _hl_hip4_fastpath.py."
            )

    def test_fastpath_settlement_loader_uses_settlement_loader_cols(self):
        import inspect
        from hlanalysis.backtest.data import _hl_hip4_fastpath
        src = inspect.getsource(_hl_hip4_fastpath)
        for col in SETTLEMENT_LOADER_COLS:
            assert col in src, (
                f"Column {col!r} from SETTLEMENT_LOADER_COLS not found in _hl_hip4_fastpath.py."
            )

    def test_pm_fastpath_book_loader_uses_book_snapshot_loader_cols(self):
        import inspect
        from hlanalysis.backtest.data import _pm_fastpath
        src = inspect.getsource(_pm_fastpath)
        for col in BOOK_SNAPSHOT_LOADER_COLS:
            assert col in src, (
                f"Column {col!r} from BOOK_SNAPSHOT_LOADER_COLS not found in _pm_fastpath.py."
            )


# ---------------------------------------------------------------------------
# Drift detection: to_record column ORDER must be stable
# ---------------------------------------------------------------------------

class TestColumnOrderStability:
    """The exact column order emitted by to_record() is part of the parquet
    contract — ``pa.Table.from_pylist`` derives the schema from key order.
    These tests pin the order so a refactor that permutes fields in
    ``_BASE_FIELD_ORDER`` or a struct's ``__struct_fields__`` is caught.

    ``BASE_COLS`` is ``(venue, product_type, mechanism, symbol, exchange_ts,
    local_recv_ts, seq, event_type)``; the first 8 columns of every event.
    Event-specific fields follow immediately after.
    """

    def test_trade_record_column_order(self):
        ev = TradeEvent(
            **_base_kwargs(),
            price=0.5, size=10.0, side="buy", trade_id="t1",
        )
        keys = list(to_record(ev).keys())
        # BASE_COLS already includes event_type as the last base column
        expected_prefix = list(BASE_COLS)
        assert keys[:len(expected_prefix)] == expected_prefix, (
            f"to_record(TradeEvent) base column order changed. "
            f"Got prefix: {keys[:len(expected_prefix)]!r}, "
            f"expected: {expected_prefix!r}"
        )
        # event-specific fields start immediately after the base prefix
        assert keys[len(expected_prefix)] == "price", (
            f"First event-specific column of TradeEvent must be 'price', got {keys!r}"
        )

    def test_settlement_record_column_order(self):
        ev = SettlementEvent(
            **_base_kwargs(),
            settled_side_idx=0, settle_price=1.0, settle_ts=1_100_000_000,
        )
        keys = list(to_record(ev).keys())
        expected_prefix = list(BASE_COLS)
        assert keys[:len(expected_prefix)] == expected_prefix, (
            f"to_record(SettlementEvent) base column order changed: {keys!r}"
        )
        assert keys[len(expected_prefix)] == "settled_side_idx", (
            f"First event-specific column of SettlementEvent must be 'settled_side_idx', got {keys!r}"
        )

    def test_book_snapshot_record_column_order(self):
        ev = BookSnapshotEvent(
            **_base_kwargs(),
            bid_px=[0.49], bid_sz=[1.0], ask_px=[0.51], ask_sz=[1.0],
        )
        keys = list(to_record(ev).keys())
        expected_prefix = list(BASE_COLS)
        assert keys[:len(expected_prefix)] == expected_prefix, (
            f"to_record(BookSnapshotEvent) base column order changed: {keys!r}"
        )
        assert keys[len(expected_prefix)] == "bid_px", (
            f"First event-specific column of BookSnapshotEvent must be 'bid_px', got {keys!r}"
        )
