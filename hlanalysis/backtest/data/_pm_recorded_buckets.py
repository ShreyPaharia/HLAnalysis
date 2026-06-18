"""Build PM ``priceBucket`` questions directly from RECORDED ``question_meta``.

The Gamma-fetched cache manifest holds historical "above-on-DATE" bucket markets
whose leg token IDs do NOT match the tokens the recorder actually captured books
for (overlap was 15 of 131 for ``btc-multi-strikes-weekly``). So manifest bucket
questions backtest to 0 decisions — every leg's recorded book is empty in-window.

The recorder, however, writes a ``question_meta`` event for every multistrike
leg (recorded as an individual ``priceBinary`` question) carrying the exact
token IDs that DO have recorded books, plus the strike (in ``question_name``),
the (yes,no) token pair, the condition_id, and the expiry. This module groups
those per-leg recordings back into ``priceBucket`` questions whose leg tokens
match the recorded books BY CONSTRUCTION — so bucket backtests run on real data.

Output entries use the SAME schema as the manifest's ``kind="bucket"`` entries,
so the data source merges them into the manifest in-memory and every downstream
consumer (``discover`` / ``events_arrays`` / ``question_view`` / ``leg_payoff``)
works unchanged.

Resolution is PURE ORACLE: a leg "above $X" wins (YES) iff the reference close at
expiry > X. Deterministic from the dense recorded Binance feed; recorded
settlement covers only ~42% of the YES legs, so a uniform oracle rule is both
more complete and reproducible (and matches the repo's documented bucket
win-label convention — see research/outcome_markets.py).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from glob import glob as _glob

# "Will the price of Bitcoin be above $56,000 on June 6?" -> 56000.0
# "...above $1.60 on June 17?" -> 1.6
_THRESHOLD_RE = re.compile(r"above\s+\$?\s*([\d,]+(?:\.\d+)?)", re.IGNORECASE)


def _parse_threshold(question_name: str | None) -> float | None:
    """Parse the dollar strike from a leg's ``question_name``.

    Parsed from the human question (robust across coins) rather than the
    ``targetPrice`` field, which the recorder populates inconsistently (e.g. an
    index ``2`` for XRP vs the dollar value ``56000`` for BTC).
    """
    if not question_name:
        return None
    m = _THRESHOLD_RE.search(question_name)
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except ValueError:
        return None


def read_recorded_bucket_legs(question_meta_glob: str, series_slug: str) -> list[dict]:
    """Return one row per recorded leg for ``series_slug`` from ``question_meta``.

    Keys: ``expiry``, ``expiry_ns``, ``yes_tok``, ``no_tok``, ``cond``, ``qn``.
    Empty list when no parquet matches the glob (graceful degrade).
    """
    if not _glob(question_meta_glob, recursive=True):
        return []
    import duckdb

    con = duckdb.connect()
    try:
        rows = con.sql(
            f"""
            WITH m AS (
              SELECT DISTINCT
                list_extract(values, list_position(keys, 'series_slug')) AS slug,
                list_extract(values, list_position(keys, 'expiry')) AS expiry,
                list_extract(values, list_position(keys, 'expiry_ns')) AS expiry_ns,
                list_extract(values, list_position(keys, 'yes_token_id')) AS yes_tok,
                list_extract(values, list_position(keys, 'no_token_id')) AS no_tok,
                list_extract(values, list_position(keys, 'condition_id')) AS cond,
                list_extract(values, list_position(keys, 'question_name')) AS qn
              FROM read_parquet('{question_meta_glob}', hive_partitioning=1)
            )
            SELECT expiry, expiry_ns, yes_tok, no_tok, cond, qn
            FROM m
            WHERE slug = '{series_slug}' AND expiry_ns IS NOT NULL
            """
        ).fetchall()
    finally:
        con.close()
    return [
        {
            "expiry": str(r[0]),
            "expiry_ns": int(r[1]),
            "yes_tok": str(r[2]),
            "no_tok": str(r[3]),
            "cond": str(r[4]),
            "qn": r[5],
        }
        for r in rows
    ]


def _min_book_ts_before(book_glob_for: Callable[[str], str], tokens: list[str], end_ns: int) -> int | None:
    """Earliest recorded book ``exchange_ts`` across ``tokens`` that is BEFORE
    ``end_ns`` (the bucket's expiry), or None when no leg has a pre-expiry book
    snapshot.

    The ``< end_ns`` filter is load-bearing: the recorder also captures a
    handful of post-expiry/settlement snapshots, so the unfiltered minimum can
    fall AFTER expiry — which would invert the question window (start > end) and
    the scan loop would break before its first tick. None signals an untradeable
    bucket (no pre-expiry liquidity) → the caller skips it.
    """
    existing = [g for t in tokens if _glob((g := book_glob_for(t)), recursive=True)]
    if not existing:
        return None
    import duckdb

    rels = ",".join(f"'{g}'" for g in existing)
    con = duckdb.connect()
    try:
        row = con.sql(
            f"SELECT min(exchange_ts) FROM read_parquet([{rels}], hive_partitioning=1, union_by_name=true) "
            f"WHERE exchange_ts < {int(end_ns)}"
        ).fetchone()
    finally:
        con.close()
    return int(row[0]) if row and row[0] is not None else None


def build_recorded_bucket_entries(
    *,
    question_meta_glob: str,
    series_slug: str,
    book_glob_for: Callable[[str], str],
    oracle_close_at: Callable[[int], float | None],
) -> dict[str, dict]:
    """Build manifest-schema ``kind="bucket"`` entries from recorded metadata.

    Groups the recorded per-leg ``priceBinary`` rows of ``series_slug`` by expiry
    into one bucket each, ordered by ascending strike (the "above ladder"), with
    ``leg_tokens = [[yes, no], ...]`` (even index = YES, matching the strategy's
    bucket candidate restriction). Leg resolutions are oracle-derived via
    ``oracle_close_at(expiry_ns)``. ``start_ts_ns`` is the earliest recorded book
    ts BEFORE expiry across the bucket's legs (a tight window over real coverage).

    Returns ``{question_id: entry}`` keyed ``f"{series_slug}:{expiry}"``. Buckets
    with fewer than 2 strike legs, no parseable strikes, no oracle close, or no
    pre-expiry book coverage on any leg are skipped (untradeable).
    """
    legs = read_recorded_bucket_legs(question_meta_glob, series_slug)
    by_expiry: dict[str, list[dict]] = {}
    for leg in legs:
        th = _parse_threshold(leg["qn"])
        if th is None:
            continue
        by_expiry.setdefault(leg["expiry"], []).append({**leg, "threshold": th})

    out: dict[str, dict] = {}
    for expiry, group in by_expiry.items():
        # Sort by strike and dedupe (a leg can recur across daily question_meta
        # snapshots; DISTINCT already collapsed identical rows, but two strikes
        # could parse equal — keep the first).
        group.sort(key=lambda x: x["threshold"])
        seen: set[float] = set()
        legs2 = [g for g in group if not (g["threshold"] in seen or seen.add(g["threshold"]))]
        if len(legs2) < 2:
            continue
        expiry_ns = legs2[0]["expiry_ns"]
        close = oracle_close_at(expiry_ns)
        if close is None:
            continue
        thresholds = [g["threshold"] for g in legs2]
        leg_tokens = [[g["yes_tok"], g["no_tok"]] for g in legs2]
        leg_condition_ids = [g["cond"] for g in legs2]
        leg_resolutions = ["yes" if close > th else "no" for th in thresholds]
        tokens = [t for pair in leg_tokens for t in pair]
        mb = _min_book_ts_before(book_glob_for, tokens, expiry_ns)
        if mb is None:
            # No pre-expiry book snapshots on any leg → untradeable (e.g. an
            # expiry the recorder only caught post-settlement, or a not-yet-
            # resolved future expiry). Skip rather than emit an inverted window.
            continue
        start_ts_ns = mb
        qid = f"{series_slug}:{expiry}"
        out[qid] = {
            "kind": "bucket",
            "bucket": {
                "event_slug": qid,
                "start_ts_ns": int(start_ts_ns),
                "end_ts_ns": int(expiry_ns),
                "thresholds": thresholds,
                "leg_tokens": leg_tokens,
                "leg_condition_ids": leg_condition_ids,
                "leg_resolutions": leg_resolutions,
            },
        }
    return out


__all__ = [
    "build_recorded_bucket_entries",
    "read_recorded_bucket_legs",
    "_parse_threshold",
]
