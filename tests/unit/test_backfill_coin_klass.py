"""Tests for tools/backfill_coin_klass.py.

TDD: written BEFORE the implementation. Covers:
  1. Pure expansion — coin_klass_rows() emits correct (coin, klass, question_idx)
     triples for priceBinary (one outcome × 2 sides = 2 coins) and priceBucket
     (N outcomes × 2 sides = 2N coins).
  2. StateDAL integration — after backfill_from_questions() the coin_klass_map()
     on a real (tmp_path) StateDAL returns every expected mapping.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import NamedTuple

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

# Make the tools/ directory importable when running from the repo root.
_TOOLS = Path(__file__).parent.parent.parent / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from backfill_coin_klass import coin_klass_rows, backfill_from_questions
from hlanalysis.engine.state import StateDAL


# ---------------------------------------------------------------------------
# Minimal question-like object (same field names used by coin_klass_rows)
# ---------------------------------------------------------------------------


class _Q(NamedTuple):
    question_idx: int
    named_outcome_idxs: list[int]
    klass: str


# ---------------------------------------------------------------------------
# coin_klass_rows — pure expansion tests
# ---------------------------------------------------------------------------


def test_binary_two_coins():
    """priceBinary with one outcome → 2 coins: #{10o+0} and #{10o+1}."""
    q = _Q(question_idx=1000015, named_outcome_idxs=[15], klass="priceBinary")
    rows = coin_klass_rows([q])
    assert len(rows) == 2
    coins = {r[0] for r in rows}
    assert coins == {"#150", "#151"}
    for coin, klass, qidx in rows:
        assert klass == "priceBinary"
        assert qidx == 1000015


def test_bucket_four_coins():
    """priceBucket with two outcomes → 4 coins."""
    q = _Q(question_idx=7, named_outcome_idxs=[0, 1], klass="priceBucket")
    rows = coin_klass_rows([q])
    assert len(rows) == 4
    coins = {r[0] for r in rows}
    assert coins == {"#0", "#1", "#10", "#11"}
    for coin, klass, qidx in rows:
        assert klass == "priceBucket"
        assert qidx == 7


def test_bucket_six_coins():
    """priceBucket with three outcomes → 6 coins."""
    q = _Q(question_idx=42, named_outcome_idxs=[3, 4, 5], klass="priceBucket")
    rows = coin_klass_rows([q])
    assert len(rows) == 6
    coins = {r[0] for r in rows}
    assert coins == {"#30", "#31", "#40", "#41", "#50", "#51"}


def test_outcome_idxs_sorted():
    """named_outcome_idxs order must not matter — results sorted ascending."""
    q_unsorted = _Q(question_idx=99, named_outcome_idxs=[5, 2], klass="priceBucket")
    q_sorted = _Q(question_idx=99, named_outcome_idxs=[2, 5], klass="priceBucket")
    assert set(coin_klass_rows([q_unsorted])) == set(coin_klass_rows([q_sorted]))


def test_empty_questions_returns_empty():
    assert coin_klass_rows([]) == []


def test_skips_empty_klass():
    """Questions with an empty klass string must be skipped."""
    q = _Q(question_idx=1, named_outcome_idxs=[0], klass="")
    assert coin_klass_rows([q]) == []


def test_skips_empty_named_outcomes():
    """Questions with no named outcomes produce no rows (nothing to expand)."""
    q = _Q(question_idx=2, named_outcome_idxs=[], klass="priceBinary")
    assert coin_klass_rows([q]) == []


def test_multiple_questions():
    """Multiple questions produce the union of their rows."""
    qs = [
        _Q(1000015, [15], "priceBinary"),
        _Q(7, [0, 1], "priceBucket"),
    ]
    rows = coin_klass_rows(qs)
    assert len(rows) == 6  # 2 + 4
    binary_rows = [r for r in rows if r[1] == "priceBinary"]
    bucket_rows = [r for r in rows if r[1] == "priceBucket"]
    assert len(binary_rows) == 2
    assert len(bucket_rows) == 4


# ---------------------------------------------------------------------------
# StateDAL integration — backfill_from_questions writes to the DB
# ---------------------------------------------------------------------------


@pytest.fixture
def dal(tmp_path):
    db = tmp_path / "state.db"
    d = StateDAL(db)
    d.run_migrations()
    return d


def test_backfill_populates_coin_klass_map(dal):
    """After backfill_from_questions, coin_klass_map() covers all expanded coins."""
    qs = [
        _Q(1000015, [15], "priceBinary"),
        _Q(7, [0, 1], "priceBucket"),
    ]
    assert dal.coin_klass_map() == {}
    backfill_from_questions(qs, dal)
    m = dal.coin_klass_map()
    assert m["#150"] == "priceBinary"
    assert m["#151"] == "priceBinary"
    assert m["#0"] == "priceBucket"
    assert m["#1"] == "priceBucket"
    assert m["#10"] == "priceBucket"
    assert m["#11"] == "priceBucket"
    assert len(m) == 6


def test_backfill_idempotent(dal):
    """Running backfill twice must not error or duplicate rows."""
    qs = [_Q(1000015, [15], "priceBinary")]
    backfill_from_questions(qs, dal)
    backfill_from_questions(qs, dal)
    m = dal.coin_klass_map()
    assert m == {"#150": "priceBinary", "#151": "priceBinary"}


def test_backfill_empty_does_nothing(dal):
    backfill_from_questions([], dal)
    assert dal.coin_klass_map() == {}


# ---------------------------------------------------------------------------
# Parquet source reader — parse_question_meta_parquet
# ---------------------------------------------------------------------------


def test_parse_binary_from_parquet(tmp_path):
    """parse_question_meta_parquet reads a real parquet file and returns
    questions with correct klass/named_outcome_idxs."""
    from backfill_coin_klass import parse_question_meta_parquet

    # Minimal parquet that mirrors recorder output.
    # keys/values store class, plus other fields. PyArrow needs list columns.
    rows = [
        {
            "venue": "hyperliquid",
            "product_type": "prediction_binary",
            "mechanism": "clob",
            "symbol": "Q1000015",
            "exchange_ts": 1_700_000_000_000_000_000,
            "local_recv_ts": 1_700_000_000_000_000_000,
            "seq": None,
            "event_type": "question_meta",
            "question_idx": 1000015,
            "named_outcome_idxs": [15],
            "fallback_outcome_idx": None,
            "settled_named_outcome_idxs": [],
            "keys": ["question_name", "class", "underlying"],
            "values": ["Recurring", "priceBinary", "BTC"],
        },
    ]
    schema = pa.schema(
        [
            pa.field("venue", pa.string()),
            pa.field("product_type", pa.string()),
            pa.field("mechanism", pa.string()),
            pa.field("symbol", pa.string()),
            pa.field("exchange_ts", pa.int64()),
            pa.field("local_recv_ts", pa.int64()),
            pa.field("seq", pa.int32()),
            pa.field("event_type", pa.string()),
            pa.field("question_idx", pa.int64()),
            pa.field("named_outcome_idxs", pa.list_(pa.int64())),
            pa.field("fallback_outcome_idx", pa.int32()),
            pa.field("settled_named_outcome_idxs", pa.list_(pa.int32())),
            pa.field("keys", pa.list_(pa.string())),
            pa.field("values", pa.list_(pa.string())),
        ]
    )
    table = pa.Table.from_pylist(rows, schema=schema)
    f = tmp_path / "fixture.parquet"
    pq.write_table(table, str(f), compression="zstd")

    qs = parse_question_meta_parquet(str(tmp_path / "*.parquet"))
    assert len(qs) == 1
    assert qs[0].question_idx == 1000015
    assert qs[0].klass == "priceBinary"
    assert sorted(qs[0].named_outcome_idxs) == [15]


def test_parse_deduplicates_same_question(tmp_path):
    """parse_question_meta_parquet deduplicates rows for the same question_idx
    (recorder re-emits QuestionMetaEvent on every engine restart)."""
    from backfill_coin_klass import parse_question_meta_parquet

    row = {
        "venue": "hyperliquid",
        "product_type": "prediction_binary",
        "mechanism": "clob",
        "symbol": "Q1000015",
        "exchange_ts": 1_700_000_000_000_000_000,
        "local_recv_ts": 1_700_000_000_000_000_000,
        "seq": None,
        "event_type": "question_meta",
        "question_idx": 1000015,
        "named_outcome_idxs": [15],
        "fallback_outcome_idx": None,
        "settled_named_outcome_idxs": [],
        "keys": ["class"],
        "values": ["priceBinary"],
    }
    schema = pa.schema(
        [
            pa.field("venue", pa.string()),
            pa.field("product_type", pa.string()),
            pa.field("mechanism", pa.string()),
            pa.field("symbol", pa.string()),
            pa.field("exchange_ts", pa.int64()),
            pa.field("local_recv_ts", pa.int64()),
            pa.field("seq", pa.int32()),
            pa.field("event_type", pa.string()),
            pa.field("question_idx", pa.int64()),
            pa.field("named_outcome_idxs", pa.list_(pa.int64())),
            pa.field("fallback_outcome_idx", pa.int32()),
            pa.field("settled_named_outcome_idxs", pa.list_(pa.int32())),
            pa.field("keys", pa.list_(pa.string())),
            pa.field("values", pa.list_(pa.string())),
        ]
    )
    # Write the same row twice (simulates two parquet flushes of the same question)
    table = pa.Table.from_pylist([row, row], schema=schema)
    f = tmp_path / "dup.parquet"
    pq.write_table(table, str(f), compression="zstd")

    qs = parse_question_meta_parquet(str(tmp_path / "*.parquet"))
    assert len(qs) == 1  # deduplicated to one question


def test_parse_skips_rows_without_class(tmp_path):
    """Rows with no 'class' key in keys/values are skipped — can't classify."""
    from backfill_coin_klass import parse_question_meta_parquet

    row = {
        "venue": "hyperliquid",
        "product_type": "prediction_binary",
        "mechanism": "clob",
        "symbol": "Q9",
        "exchange_ts": 1_700_000_000_000_000_000,
        "local_recv_ts": 1_700_000_000_000_000_000,
        "seq": None,
        "event_type": "question_meta",
        "question_idx": 9,
        "named_outcome_idxs": [9],
        "fallback_outcome_idx": None,
        "settled_named_outcome_idxs": [],
        "keys": ["underlying"],  # no "class" key
        "values": ["BTC"],
    }
    schema = pa.schema(
        [
            pa.field("venue", pa.string()),
            pa.field("product_type", pa.string()),
            pa.field("mechanism", pa.string()),
            pa.field("symbol", pa.string()),
            pa.field("exchange_ts", pa.int64()),
            pa.field("local_recv_ts", pa.int64()),
            pa.field("seq", pa.int32()),
            pa.field("event_type", pa.string()),
            pa.field("question_idx", pa.int64()),
            pa.field("named_outcome_idxs", pa.list_(pa.int64())),
            pa.field("fallback_outcome_idx", pa.int32()),
            pa.field("settled_named_outcome_idxs", pa.list_(pa.int32())),
            pa.field("keys", pa.list_(pa.string())),
            pa.field("values", pa.list_(pa.string())),
        ]
    )
    table = pa.Table.from_pylist([row], schema=schema)
    f = tmp_path / "noclass.parquet"
    pq.write_table(table, str(f), compression="zstd")

    qs = parse_question_meta_parquet(str(tmp_path / "*.parquet"))
    assert qs == []
