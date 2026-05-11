"""Unit tests for the HL HIP-4 recorded-data DataSource.

Exercises the §3 contract against the committed
``tests/fixtures/hl_hip4/`` slice — see ``tests/fixtures/hl_hip4/README.md`` for
its provenance and capture command.
"""
from __future__ import annotations

import random
from pathlib import Path

import duckdb
import pytest

from hlanalysis.backtest.core.data_source import QuestionDescriptor
from hlanalysis.backtest.core.events import (
    BookSnapshot,
    ReferenceEvent,
    SettlementEvent,
    TradeEvent,
)
from hlanalysis.backtest.data.hl_hip4 import (
    HLHip4DataSource,
    _expiry_ns,
    _iso_to_ns,
    _leg_for,
    _leg_outcome_side,
)


FIXTURE_ROOT = Path(__file__).resolve().parents[2] / "fixtures" / "hl_hip4"


@pytest.fixture(scope="module")
def source() -> HLHip4DataSource:
    return HLHip4DataSource(data_root=FIXTURE_ROOT)


@pytest.fixture(scope="module")
def discovered(source: HLHip4DataSource) -> QuestionDescriptor:
    qs = source.discover(start="2026-05-09", end="2026-05-11", underlying="BTC")
    assert len(qs) == 1, qs
    return qs[0]


# ---------------------------------------------------------------------------
# Helper-function unit tests (pure, no I/O).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sym,want",
    [
        ("#150", (15, 0)),
        ("#151", (15, 1)),
        ("#220", (22, 0)),
        ("#221", (22, 1)),
        ("#0", (0, 0)),
    ],
)
def test_leg_outcome_side_decodes(sym: str, want: tuple[int, int]) -> None:
    assert _leg_outcome_side(sym) == want


def test_leg_outcome_side_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        _leg_outcome_side("BTC")


@pytest.mark.parametrize(
    "outcome,side,want",
    [(22, 0, "#220"), (22, 1, "#221"), (1000, 0, "#10000"), (0, 1, "#1")],
)
def test_leg_for_encodes(outcome: int, side: int, want: str) -> None:
    assert _leg_for(outcome, side) == want


def test_expiry_ns_parses_hl_format() -> None:
    # 2026-05-10 06:00 UTC
    assert _expiry_ns("20260510-0600") == 1778392800 * 1_000_000_000


def test_iso_to_ns_round_trip() -> None:
    assert _iso_to_ns("2026-05-10") == 1778371200 * 1_000_000_000


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def test_discover_finds_fixture_question(source: HLHip4DataSource) -> None:
    qs = source.discover(start="2026-05-09", end="2026-05-11", underlying="BTC")
    assert len(qs) == 1
    q = qs[0]
    assert q.question_id == "Q1000015"
    assert q.question_idx == 1_000_015
    assert q.klass == "priceBinary"
    assert q.underlying == "BTC"
    assert q.leg_symbols == ("#150", "#151")
    # expiry is 2026-05-10 06:00 UTC
    assert q.end_ts_ns == _expiry_ns("20260510-0600")


def test_discover_filter_excludes_wrong_underlying(source: HLHip4DataSource) -> None:
    qs = source.discover(start="2026-05-09", end="2026-05-11", underlying="ETH")
    assert qs == []


def test_discover_filter_excludes_wrong_window(source: HLHip4DataSource) -> None:
    qs = source.discover(start="2026-05-20", end="2026-05-21", underlying="BTC")
    assert qs == []


# ---------------------------------------------------------------------------
# events() — ordering + L2 reconstruction + reference stream
# ---------------------------------------------------------------------------


def test_events_monotonic_ts_ns(
    source: HLHip4DataSource, discovered: QuestionDescriptor
) -> None:
    prev = -1
    n = 0
    for ev in source.events(discovered):
        assert ev.ts_ns >= prev, (prev, ev)
        prev = ev.ts_ns
        n += 1
    # The fixture is dense — expect ≥10k events.
    assert n >= 10_000, n


def test_events_contain_all_kinds(
    source: HLHip4DataSource, discovered: QuestionDescriptor
) -> None:
    kinds: set[str] = set()
    for ev in source.events(discovered):
        kinds.add(type(ev).__name__)
        if {"BookSnapshot", "TradeEvent", "ReferenceEvent"} <= kinds:
            break
    assert "BookSnapshot" in kinds
    assert "TradeEvent" in kinds
    assert "ReferenceEvent" in kinds


def test_events_book_snapshots_per_leg(
    source: HLHip4DataSource, discovered: QuestionDescriptor
) -> None:
    counts = {s: 0 for s in discovered.leg_symbols}
    for ev in source.events(discovered):
        if isinstance(ev, BookSnapshot):
            counts[ev.symbol] = counts.get(ev.symbol, 0) + 1
    # Both legs are recorded at ~equal density.
    for leg, n in counts.items():
        assert n >= 1_000, (leg, n)


def test_events_trade_sides_normalised(
    source: HLHip4DataSource, discovered: QuestionDescriptor
) -> None:
    sides: set[str] = set()
    for ev in source.events(discovered):
        if isinstance(ev, TradeEvent):
            sides.add(ev.side)
    assert sides.issubset({"buy", "sell"})


def test_events_reference_mid_in_range(
    source: HLHip4DataSource, discovered: QuestionDescriptor
) -> None:
    for ev in source.events(discovered):
        if isinstance(ev, ReferenceEvent):
            # BTC trades in the high-$70k–$80k band over the fixture window.
            assert 40_000.0 < ev.close < 200_000.0
            assert ev.high == ev.low == ev.close  # we collapse H/L to mid


def test_events_no_settlement_in_fixture(
    source: HLHip4DataSource, discovered: QuestionDescriptor
) -> None:
    # The fixture's window doesn't include the question's settle moment.
    for ev in source.events(discovered):
        assert not isinstance(ev, SettlementEvent)


def test_l2_reconstruction_matches_raw_parquet(
    source: HLHip4DataSource, discovered: QuestionDescriptor
) -> None:
    """Spec acceptance: L2 reconstruction matches recorded `book_snapshot` rows
    on at least one sampled tick."""
    leg = "#150"
    glob = str(
        FIXTURE_ROOT
        / "venue=hyperliquid/product_type=prediction_binary/mechanism=clob"
        / "event=book_snapshot" / f"symbol={leg}" / "**" / "*.parquet"
    )
    con = duckdb.connect()
    # Sample 5 random ticks from the raw parquet.
    total = con.sql(
        f"SELECT COUNT(*) FROM read_parquet('{glob}', hive_partitioning=1)"
    ).fetchone()[0]
    assert total > 100, total
    random.seed(0)
    offsets = sorted(random.sample(range(total), k=5))
    raw_rows = {}
    for off in offsets:
        row = con.sql(
            f"""
            SELECT exchange_ts, bid_px, bid_sz, ask_px, ask_sz
            FROM read_parquet('{glob}', hive_partitioning=1)
            ORDER BY exchange_ts OFFSET {off} LIMIT 1
            """
        ).fetchone()
        raw_rows[int(row[0])] = (
            tuple((float(p), float(s)) for p, s in zip(row[1], row[2])),
            tuple((float(p), float(s)) for p, s in zip(row[3], row[4])),
        )

    hits: dict[int, BookSnapshot] = {}
    for ev in source.events(discovered):
        if isinstance(ev, BookSnapshot) and ev.symbol == leg and ev.ts_ns in raw_rows:
            hits[ev.ts_ns] = ev
        if len(hits) == len(raw_rows):
            break

    assert set(hits) == set(raw_rows), (set(raw_rows) - set(hits))
    for ts, (bids, asks) in raw_rows.items():
        assert hits[ts].bids == bids, ts
        assert hits[ts].asks == asks, ts


# ---------------------------------------------------------------------------
# question_view + resolved_outcome
# ---------------------------------------------------------------------------


def test_question_view_binary(
    source: HLHip4DataSource, discovered: QuestionDescriptor
) -> None:
    qv = source.question_view(
        discovered, now_ns=discovered.start_ts_ns, settled=False
    )
    assert qv.klass == "priceBinary"
    assert qv.yes_symbol == "#150"
    assert qv.no_symbol == "#151"
    assert qv.strike == 80_354.0  # targetPrice from market_meta
    assert qv.underlying == "BTC"
    assert qv.expiry_ns == discovered.end_ts_ns
    assert qv.period == "1d"
    assert qv.settled is False
    assert qv.settled_side is None
    kv = dict(qv.kv)
    assert kv.get("class") == "priceBinary"
    assert kv.get("targetPrice") == "80354"


def test_resolved_outcome_falls_back_to_mark_vs_strike(
    source: HLHip4DataSource, discovered: QuestionDescriptor
) -> None:
    # No settlement in the fixture; binary fallback compares last BTC mid vs strike.
    # Strike=80354. The 2h before expiry sat at ~$80k+, so the fallback should
    # return either "yes" or "no" — never "unknown" since we have BTC data.
    outcome = source.resolved_outcome(discovered)
    assert outcome in ("yes", "no")


def test_question_view_bucket_synthetic(source: HLHip4DataSource) -> None:
    """The bucket path of question_view doesn't run against the fixture's binary,
    so unit-test it with a hand-crafted descriptor + mocked meta cache."""
    src = HLHip4DataSource(data_root=FIXTURE_ROOT)
    q = QuestionDescriptor(
        question_id="Q-synthetic-bucket",
        question_idx=99,
        start_ts_ns=1_000_000_000_000_000_000,
        end_ts_ns=2_000_000_000_000_000_000,
        leg_symbols=("#220", "#221", "#230", "#231", "#240", "#241", "#210", "#211"),
        klass="priceBucket",
        underlying="BTC",
    )
    # Inject meta into the cache so the test doesn't depend on real parquet rows.
    from hlanalysis.backtest.data.hl_hip4 import _QuestionMeta
    src._meta_cache[q.question_id] = _QuestionMeta(
        name="Recurring",
        kv={
            "question_name": "Recurring",
            "class": "priceBucket",
            "priceThresholds": "79043,82270",
            "period": "1d",
        },
    )
    qv = src.question_view(q, now_ns=q.start_ts_ns, settled=False)
    assert qv.klass == "priceBucket"
    assert qv.yes_symbol == ""
    assert qv.no_symbol == ""
    assert qv.strike == 79_043.0  # lowest threshold
    assert qv.leg_symbols == q.leg_symbols
    assert qv.period == "1d"
