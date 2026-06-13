"""Unit tests for the pure planning logic of tools/backfill_daily_compaction.py."""

from __future__ import annotations

from tools.backfill_daily_compaction import plan_migration

STREAM = "venue=binance/product_type=perp/mechanism=clob/event=bbo/symbol=BTCUSDT"
TODAY = "2026-06-05"


def _k(date: str, hour: str) -> str:
    return f"{STREAM}/date={date}/hour={hour}/compacted.parquet"


def test_groups_hourly_into_one_partition() -> None:
    keys = [_k("2026-06-01", "00"), _k("2026-06-01", "01"), _k("2026-06-01", "23")]
    plan = plan_migration(keys, TODAY)
    assert len(plan) == 1
    p = plan[0]
    assert p.date == "2026-06-01"
    assert p.date_prefix == f"{STREAM}/date=2026-06-01"
    assert p.sources == sorted(keys)
    assert p.target_key == f"{STREAM}/date=2026-06-01/hour=all/compacted.parquet"


def test_skips_today_and_future() -> None:
    keys = [_k(TODAY, "00"), _k("2026-06-06", "00")]
    assert plan_migration(keys, TODAY) == []


def test_skips_already_migrated_daily_only() -> None:
    # Only an hour=all file present -> nothing to do.
    keys = [_k("2026-06-01", "all")]
    assert plan_migration(keys, TODAY) == []


def test_partial_migration_merges_only_hourly_sources() -> None:
    # hour=all already exists AND stale hourly files remain (interrupted run):
    # the merge must use ONLY the hourly sources, never fold hour=all back in.
    keys = [_k("2026-06-01", "all"), _k("2026-06-01", "07"), _k("2026-06-01", "08")]
    plan = plan_migration(keys, TODAY)
    assert len(plan) == 1
    assert plan[0].sources == sorted([_k("2026-06-01", "07"), _k("2026-06-01", "08")])


def test_multiple_streams_and_dates() -> None:
    other = STREAM.replace("event=bbo", "event=trade")
    keys = [
        _k("2026-06-01", "00"),
        _k("2026-06-02", "00"),
        f"{other}/date=2026-06-01/hour=00/compacted.parquet",
    ]
    plan = plan_migration(keys, TODAY)
    assert len(plan) == 3
    # sorted by (date, prefix)
    assert [p.date for p in plan] == ["2026-06-01", "2026-06-01", "2026-06-02"]


def test_ignores_non_parquet_and_unparseable_keys() -> None:
    keys = [_k("2026-06-01", "00"), "logs/recorder.log", "random/key.txt", STREAM + "/_SUCCESS"]
    plan = plan_migration(keys, TODAY)
    assert len(plan) == 1
    assert plan[0].sources == [_k("2026-06-01", "00")]
