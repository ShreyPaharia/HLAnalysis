"""Resolver: HL outcome-market symbol ↔ question ↔ class ↔ expiry ↔ strike/thresholds.

Symbol encoding (verified from data):
  #NNN  where NNN = outcome_idx * 10 + side_idx
  e.g.  #100  -> outcome_idx=10, side_idx=0 (Yes)
        #101  -> outcome_idx=10, side_idx=1 (No)
        #1050 -> outcome_idx=105, side_idx=0 (Yes)

question_meta links:
  priceBucket: symbol=QNNNN (question_idx 0..N), named_outcome_idxs list of outcomes
  priceBinary: symbol=Q1000NNN (question_idx 1000000+), named_outcome_idxs=[outcome_idx]

Settlement encoding:
  Settlement records exist only for *bucket* band Yes-legs (outcome_name='Recurring Named
  Outcome').  ALL 99 records have settled_side_idx=0, which is a constant artifact — NOT
  the winner.  Binary markets emit NO settlement event.

Win-label logic (CORRECT):
  Binary: yes_won = oracle_px_at_expiry > targetPrice
    oracle_px_at_expiry = ASOF-joined HL perp oracle (product_type=perp, event=oracle,
    symbol=BTC) as-of the expiry timestamp in nanoseconds.

  Bucket: exactly one named band wins per question/expiry.
    Band boundary mapping for 2 thresholds [lo, hi] and 3 named bands (index 0,1,2):
      band 0 (index=0): oracle_px <= lo
      band 1 (index=1): lo < oracle_px <= hi
      band 2 (index=2): oracle_px > hi
    Cross-check: terminal BBO mid of the band's Yes-leg ≥ 0.5 → winner.
    oracle_mid_agree flag: True when oracle and terminal-mid agree.

DO NOT use settled_side_idx for the win label.
"""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

import duckdb
import pandas as pd

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _data_root(data_root: str) -> Path:
    return Path(data_root)


def _glob(data_root: str, event: str) -> str:
    root = _data_root(data_root)
    return str(
        root
        / "venue=hyperliquid"
        / "product_type=prediction_binary"
        / "mechanism=clob"
        / f"event={event}"
        / "symbol=*"
        / "date=*"
        / "hour=all"
        / "*.parquet"
    )


def _oracle_glob(data_root: str) -> str:
    root = _data_root(data_root)
    return str(
        root
        / "venue=hyperliquid"
        / "product_type=perp"
        / "mechanism=clob"
        / "event=oracle"
        / "symbol=BTC"
        / "date=*"
        / "hour=all"
        / "*.parquet"
    )


def _parse_expiry(expiry_str: str) -> dt.datetime:
    """Parse 'YYYYMMDD-HHMM' or 'YYYYMMDD-HH' into a UTC datetime."""
    try:
        return dt.datetime.strptime(expiry_str, "%Y%m%d-%H%M").replace(tzinfo=dt.UTC)
    except ValueError:
        return dt.datetime.strptime(expiry_str, "%Y%m%d-%H").replace(tzinfo=dt.UTC)


def band_index_to_range(
    band_index: int,
    lo: float,
    hi: float,
    n_bands: int = 3,
) -> tuple[float | None, float | None]:
    """Map a band index to its (lo_bound, hi_bound) price range.

    For 2 thresholds [lo, hi] and 3 named bands:
      band 0: (-inf, lo]   -> (None, lo)
      band 1: (lo,   hi]   -> (lo, hi)
      band 2: (hi,  +inf)  -> (hi, None)

    Boundary convention: lo < price <= hi (band_won = oracle_px > lo AND oracle_px <= hi).
    For band 0: oracle_px <= lo.
    For last band: oracle_px > hi_of_prev.

    Parameters
    ----------
    band_index : int
        0-based index of the band (from market_meta 'index' key).
    lo : float
        First (lower) price threshold.
    hi : float
        Second (upper) price threshold.
    n_bands : int
        Total number of named bands (default 3 for 2 thresholds).

    Returns
    -------
    (lo_bound, hi_bound) where None means ±infinity.
    """
    if n_bands != 3:
        raise NotImplementedError("Only 3-band (2-threshold) markets supported")
    if band_index == 0:
        return (None, lo)
    elif band_index == 1:
        return (lo, hi)
    elif band_index == 2:
        return (hi, None)
    else:
        raise ValueError(f"band_index {band_index} out of range for n_bands={n_bands}")


def band_index_wins(band_index: int, oracle_px: float, lo: float, hi: float) -> bool:
    """Return True if the oracle price falls in the given band.

    Boundary convention:
      band 0 (index=0): oracle_px <= lo
      band 1 (index=1): lo < oracle_px <= hi
      band 2 (index=2): oracle_px > hi
    """
    if band_index == 0:
        return oracle_px <= lo
    elif band_index == 1:
        return lo < oracle_px <= hi
    elif band_index == 2:
        return oracle_px > hi
    else:
        raise ValueError(f"band_index {band_index} out of range")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve_binary_outcomes(
    con: duckdb.DuckDBPyConnection,
    data_root: str,
) -> pd.DataFrame:
    """Return per-expiry binary outcome resolution using oracle price.

    Returns one row per Yes-leg (one per expiry date).

    Columns
    -------
    symbol          str    — Yes-leg symbol (side_idx=0)
    outcome_idx     int
    expiry_str      str    — raw expiry string e.g. '20260509-0600'
    expiry          datetime (UTC)
    target_price    float
    oracle_px_at_expiry  float | None  — HL perp oracle price ASOF expiry
    yes_won         bool | None       — True when oracle_px > target_price; None if no oracle
    winner_source   str    — always 'oracle' when oracle data is available
    """
    meta_glob = _glob(data_root, "market_meta")
    oracle_glob_path = _oracle_glob(data_root)

    sql = f"""
        WITH binary_yes AS (
            SELECT DISTINCT
                symbol,
                list_element(values, list_position(keys, 'outcome_idx'))::BIGINT AS outcome_idx,
                list_element(values, list_position(keys, 'expiry'))               AS expiry_str,
                list_element(values, list_position(keys, 'targetPrice'))::DOUBLE  AS target_price,
                epoch_ns(
                    strptime(
                        list_element(values, list_position(keys, 'expiry')),
                        '%Y%m%d-%H%M'
                    ) AT TIME ZONE 'UTC'
                ) AS exp_ns
            FROM read_parquet('{meta_glob}', union_by_name=true)
            WHERE list_contains(keys, 'class')
              AND list_element(values, list_position(keys, 'class'))     = 'priceBinary'
              AND list_contains(keys, 'side_name')
              AND list_element(values, list_position(keys, 'side_name')) = 'Yes'
        ),
        oracle AS (
            SELECT oracle_px, local_recv_ts
            FROM read_parquet('{oracle_glob_path}', union_by_name=true)
        )
        SELECT
            b.symbol,
            b.outcome_idx,
            b.expiry_str,
            b.target_price,
            o.oracle_px   AS oracle_px_at_expiry,
            o.local_recv_ts AS oracle_ts,
            (o.oracle_px > b.target_price) AS yes_won
        FROM binary_yes b
        ASOF JOIN oracle o ON b.exp_ns >= o.local_recv_ts
        ORDER BY b.expiry_str
    """
    try:
        df = con.execute(sql).df()
    except duckdb.IOException:
        return pd.DataFrame()

    if df.empty:
        return df

    df["expiry"] = df["expiry_str"].apply(_parse_expiry)
    df["yes_won"] = df["yes_won"].astype(bool)
    df["winner_source"] = "oracle"
    df.drop(columns=["oracle_ts"], inplace=True)

    yes_win_rate = df["yes_won"].mean()
    _log.info(
        "resolve_binary_outcomes: %d expiries, Yes-win rate=%.1f%%",
        len(df),
        yes_win_rate * 100,
    )
    return df.reset_index(drop=True)


def resolve_bucket_outcomes(
    con: duckdb.DuckDBPyConnection,
    data_root: str,
) -> pd.DataFrame:
    """Return per-band bucket outcome resolution using oracle price + terminal-mid cross-check.

    Returns one row per named bucket Yes-leg (band_index in {0,1,2}).
    Fallback legs are excluded (they have no band_index).

    Columns
    -------
    symbol             str    — Yes-leg symbol
    outcome_idx        int
    band_index         int    — 0,1,2
    q_symbol           str    — question symbol (QNNN)
    expiry_str         str
    expiry             datetime (UTC)
    lo                 float  — lower price threshold
    hi                 float  — upper price threshold
    lo_bound           float | None  — band's lower price bound (None = -inf)
    hi_bound           float | None  — band's upper price bound (None = +inf)
    oracle_px_at_expiry float | None
    terminal_mid       float | None  — BBO mid just before settlement (cross-check only)
    band_won_oracle    bool | None   — authoritative: oracle in band range
    band_won_mid       bool | None   — cross-check: terminal_mid >= 0.5
    oracle_mid_agree   bool | None   — True when oracle and mid agree
    winner_source      str    — 'oracle'
    """
    meta_glob = _glob(data_root, "market_meta")
    q_glob = _glob(data_root, "question_meta")
    oracle_glob_path = _oracle_glob(data_root)
    bbo_glob = _glob(data_root, "bbo")
    settlement_glob = _glob(data_root, "settlement")

    sql = f"""
        WITH bucket_legs AS (
            SELECT DISTINCT
                mm.symbol,
                list_element(mm.values, list_position(mm.keys, 'outcome_idx'))::BIGINT AS outcome_idx,
                list_element(mm.values, list_position(mm.keys, 'index'))::INT          AS band_index
            FROM read_parquet('{meta_glob}', union_by_name=true) mm
            WHERE list_contains(mm.keys, 'index')
              AND list_contains(mm.keys, 'side_idx')
              AND list_element(mm.values, list_position(mm.keys, 'side_idx')) = '0'
        ),
        questions AS (
            SELECT DISTINCT
                q.symbol AS q_symbol,
                unnest(q.named_outcome_idxs) AS outcome_idx,
                list_element(q.values, list_position(q.keys, 'expiry'))              AS expiry_str,
                split_part(
                    list_element(q.values, list_position(q.keys, 'priceThresholds')),
                    ',', 1
                )::DOUBLE AS lo,
                split_part(
                    list_element(q.values, list_position(q.keys, 'priceThresholds')),
                    ',', 2
                )::DOUBLE AS hi,
                epoch_ns(
                    strptime(
                        list_element(q.values, list_position(q.keys, 'expiry')),
                        '%Y%m%d-%H%M'
                    ) AT TIME ZONE 'UTC'
                ) AS exp_ns
            FROM read_parquet('{q_glob}', union_by_name=true) q
            WHERE list_contains(q.keys, 'priceThresholds')
        ),
        oracle AS (
            SELECT oracle_px, local_recv_ts
            FROM read_parquet('{oracle_glob_path}', union_by_name=true)
        ),
        settlements AS (
            SELECT DISTINCT symbol, local_recv_ts AS settle_ts
            FROM read_parquet('{settlement_glob}', union_by_name=true)
        ),
        terminal_bbo_ranked AS (
            SELECT
                b.symbol,
                b.bid_px,
                b.ask_px,
                (b.bid_px + b.ask_px) / 2.0 AS mid,
                b.local_recv_ts,
                ROW_NUMBER() OVER (
                    PARTITION BY b.symbol
                    ORDER BY b.local_recv_ts DESC
                ) AS rn
            FROM read_parquet('{bbo_glob}', union_by_name=true) b
            JOIN settlements s
              ON b.symbol = s.symbol
             AND b.local_recv_ts < s.settle_ts
        ),
        last_mid AS (
            SELECT symbol, mid FROM terminal_bbo_ranked WHERE rn = 1
        ),
        joined AS (
            SELECT
                bl.symbol,
                bl.outcome_idx,
                bl.band_index,
                qs.q_symbol,
                qs.expiry_str,
                qs.lo,
                qs.hi,
                qs.exp_ns,
                o.oracle_px AS oracle_px_at_expiry
            FROM bucket_legs bl
            JOIN questions qs ON bl.outcome_idx = qs.outcome_idx
            ASOF JOIN oracle o ON qs.exp_ns >= o.local_recv_ts
        )
        SELECT
            j.*,
            lm.mid AS terminal_mid,
            CASE
                WHEN j.band_index = 0 THEN (j.oracle_px_at_expiry <= j.lo)
                WHEN j.band_index = 2 THEN (j.oracle_px_at_expiry > j.hi)
                ELSE (j.oracle_px_at_expiry > j.lo AND j.oracle_px_at_expiry <= j.hi)
            END AS band_won_oracle,
            (lm.mid >= 0.5) AS band_won_mid
        FROM joined j
        LEFT JOIN last_mid lm ON j.symbol = lm.symbol
        ORDER BY j.expiry_str, j.band_index
    """
    try:
        df = con.execute(sql).df()
    except duckdb.IOException:
        return pd.DataFrame()

    if df.empty:
        return df

    df["expiry"] = df["expiry_str"].apply(_parse_expiry)
    df["band_won_oracle"] = df["band_won_oracle"].astype(bool)
    df["band_won_mid"] = df["band_won_mid"].where(df["terminal_mid"].notna())

    # Compute lo_bound / hi_bound columns from band_index
    lo_bounds: list[float | None] = []
    hi_bounds: list[float | None] = []
    for _, row in df.iterrows():
        lb, hb = band_index_to_range(int(row["band_index"]), float(row["lo"]), float(row["hi"]))
        lo_bounds.append(lb)
        hi_bounds.append(hb)
    df["lo_bound"] = lo_bounds
    df["hi_bound"] = hi_bounds

    # oracle_mid_agree: only defined when terminal_mid is available
    has_mid = df["terminal_mid"].notna()
    df["oracle_mid_agree"] = None
    df.loc[has_mid, "oracle_mid_agree"] = df.loc[has_mid, "band_won_oracle"] == df.loc[has_mid, "band_won_mid"]

    df["winner_source"] = "oracle"

    # Drop internal column
    df.drop(columns=["exp_ns"], inplace=True)

    # Log diagnostics
    n_q = df.groupby(["q_symbol", "expiry_str"])["band_won_oracle"].sum()
    not_exactly_one = (n_q != 1).sum()
    if not_exactly_one > 0:
        _log.warning(
            "resolve_bucket_outcomes: %d question/expiry pairs do NOT have exactly 1 oracle winner",
            not_exactly_one,
        )

    n_agree = int(df.loc[has_mid, "oracle_mid_agree"].sum())
    n_mid_total = int(has_mid.sum())
    agree_pct = n_agree / n_mid_total * 100 if n_mid_total > 0 else float("nan")
    _log.info(
        "resolve_bucket_outcomes: %d band rows, oracle/mid agreement=%.1f%% (%d/%d)",
        len(df),
        agree_pct,
        n_agree,
        n_mid_total,
    )

    return df.reset_index(drop=True)


def load_market_reference(con: duckdb.DuckDBPyConnection, data_root: str) -> pd.DataFrame:
    """Return one row per leg symbol with resolved metadata.

    Columns
    -------
    symbol          str    — leg symbol, e.g. '#100'
    outcome_idx     int    — HL outcome index
    side_idx        int    — 0=Yes, 1=No
    side_name       str    — 'Yes' or 'No'
    market_class    str    — 'priceBinary' or 'priceBucket'
    expiry          datetime (UTC) — market expiry
    target_price    float | None   — binary strike (priceBinary only)
    lo_threshold    float | None   — lower bucket boundary (priceBucket only)
    hi_threshold    float | None   — upper bucket boundary (priceBucket only)
    is_yes          bool   — True when side_idx=0
    question_symbol str    — QNNNN from question_meta (None if not found)

    Duplicate market_meta rows (same symbol from multiple dates) are de-duped
    by taking the first occurrence.
    """
    glob = _glob(data_root, "market_meta")

    sql = f"""
        SELECT DISTINCT
            symbol,
            keys,
            values
        FROM read_parquet('{glob}', union_by_name=true)
        WHERE array_contains(keys, 'outcome_idx')
          AND array_contains(keys, 'side_idx')
          AND array_contains(keys, 'side_name')
    """
    try:
        raw = con.execute(sql).df()
    except duckdb.IOException:
        return pd.DataFrame()

    if raw.empty:
        return pd.DataFrame()

    # Parse keys/values arrays into columns
    records = []
    for _, row in raw.iterrows():
        keys = list(row["keys"])
        values = list(row["values"])
        kv = dict(zip(keys, values))

        symbol = row["symbol"]
        outcome_idx = int(kv.get("outcome_idx", -1))
        side_idx = int(kv.get("side_idx", -1))
        side_name = kv.get("side_name", "")
        market_class = kv.get("class")
        expiry_str = kv.get("expiry")
        target_price_str = kv.get("targetPrice")

        expiry = _parse_expiry(expiry_str) if expiry_str else None
        target_price = float(target_price_str) if target_price_str else None

        # For priceBinary, lo/hi are None
        lo_threshold = None
        hi_threshold = None
        # priceBucket legs have priceThresholds in the containing question
        # They will be joined from question_meta below

        records.append(
            {
                "symbol": symbol,
                "outcome_idx": outcome_idx,
                "side_idx": side_idx,
                "side_name": side_name,
                "market_class": market_class,
                "expiry": expiry,
                "target_price": target_price,
                "lo_threshold": lo_threshold,
                "hi_threshold": hi_threshold,
                "is_yes": side_idx == 0,
            }
        )

    df = pd.DataFrame(records)
    # De-duplicate: keep first occurrence per symbol
    df = df.drop_duplicates(subset=["symbol"]).reset_index(drop=True)

    # Join question_meta for question_symbol and bucket thresholds
    q_df = _load_question_meta(con, data_root)
    if not q_df.empty:
        df = df.merge(q_df, on="outcome_idx", how="left")
        # For priceBucket legs, fill lo/hi from question_meta
        mask = df["market_class"].isna() & df["q_lo"].notna()
        df.loc[mask, "market_class"] = "priceBucket"
        df["lo_threshold"] = df["lo_threshold"].fillna(df.get("q_lo", pd.Series(dtype=float)))
        df["hi_threshold"] = df["hi_threshold"].fillna(df.get("q_hi", pd.Series(dtype=float)))
        df["question_symbol"] = df.get("q_symbol", pd.Series(dtype=str))
        # For bucket legs, expiry may come from question_meta if missing in market_meta
        mask_exp = df["expiry"].isna() & df.get("q_expiry", pd.Series(dtype=object)).notna()
        if "q_expiry" in df.columns:
            df.loc[mask_exp, "expiry"] = df.loc[mask_exp, "q_expiry"]
        df.drop(columns=[c for c in ["q_lo", "q_hi", "q_symbol", "q_expiry"] if c in df.columns], inplace=True)
    else:
        df["question_symbol"] = None

    return df.reset_index(drop=True)


def _load_question_meta(
    con: duckdb.DuckDBPyConnection,
    data_root: str,
) -> pd.DataFrame:
    """Load question_meta and return one row per outcome_idx with question context.

    Returns columns: outcome_idx, q_symbol, q_lo, q_hi, q_expiry
    """
    glob = _glob(data_root, "question_meta")
    sql = f"""
        SELECT DISTINCT
            symbol AS q_symbol,
            question_idx,
            named_outcome_idxs,
            fallback_outcome_idx,
            keys,
            values
        FROM read_parquet('{glob}', union_by_name=true)
        WHERE named_outcome_idxs IS NOT NULL
    """
    try:
        raw = con.execute(sql).df()
    except duckdb.IOException:
        return pd.DataFrame()

    if raw.empty:
        return pd.DataFrame()

    # De-dup per (q_symbol, question_idx)
    raw = raw.drop_duplicates(subset=["q_symbol", "question_idx"]).reset_index(drop=True)

    records = []
    for _, row in raw.iterrows():
        keys = list(row["keys"]) if row["keys"] is not None else []
        values = list(row["values"]) if row["values"] is not None else []
        kv = dict(zip(keys, values))

        q_symbol = row["q_symbol"]
        named_idxs = list(row["named_outcome_idxs"]) if row["named_outcome_idxs"] is not None else []
        fallback_idx = row["fallback_outcome_idx"]

        # Parse thresholds for bucket questions
        price_thresholds_str = kv.get("priceThresholds")
        lo, hi = None, None
        if price_thresholds_str:
            parts = str(price_thresholds_str).replace("'", "").split(",")
            if len(parts) == 2:
                try:
                    lo = float(parts[0].strip())
                    hi = float(parts[1].strip())
                except ValueError:
                    pass

        expiry_str = kv.get("expiry")
        q_expiry = _parse_expiry(expiry_str) if expiry_str else None

        # Emit one row per named outcome_idx (+ fallback)
        all_idxs = [int(i) for i in named_idxs]
        if fallback_idx is not None and not pd.isna(fallback_idx):
            all_idxs.append(int(fallback_idx))

        for oidx in all_idxs:
            records.append(
                {
                    "outcome_idx": oidx,
                    "q_symbol": q_symbol,
                    "q_lo": lo,
                    "q_hi": hi,
                    "q_expiry": q_expiry,
                }
            )

    if not records:
        return pd.DataFrame()

    q_df = pd.DataFrame(records)
    # De-dup by outcome_idx (take first)
    q_df = q_df.drop_duplicates(subset=["outcome_idx"]).reset_index(drop=True)
    return q_df


def load_settlements(con: duckdb.DuckDBPyConnection, data_root: str) -> pd.DataFrame:
    """Return one row per settlement event.

    Columns
    -------
    symbol          str   — Yes-leg symbol (e.g. '#1070'), always ends in '0'
    settled_at      int   — settlement timestamp in nanoseconds (local_recv_ts)
    won             bool  — always True (settlement records only exist for bucket
                            band Yes-legs; binary markets emit no settlement event)
    settlement_price float | None  — settle_price if present (usually NULL in recorded data)

    Notes
    -----
    Settlement records in the recorder only contain entries for bucket band Yes-leg
    symbols (outcome_name='Recurring Named Outcome').  ALL records have
    settled_side_idx=0, which is a constant artifact — NOT the winner.
    Use resolve_bucket_outcomes() for authoritative oracle-based win labels.

    Binary markets do NOT produce settlement events; use resolve_binary_outcomes()
    for binary win labels.
    """
    glob = _glob(data_root, "settlement")
    sql = f"""
        SELECT DISTINCT
            symbol,
            local_recv_ts AS settled_at,
            settled_side_idx,
            settle_price
        FROM read_parquet('{glob}', union_by_name=true)
    """
    try:
        raw = con.execute(sql).df()
    except duckdb.IOException:
        return pd.DataFrame(
            {
                "symbol": pd.Series([], dtype="object"),
                "settled_at": pd.Series([], dtype="int64"),
                "won": pd.Series([], dtype="bool"),
                "settlement_price": pd.Series([], dtype="float64"),
            }
        )

    if raw.empty:
        return pd.DataFrame(
            {
                "symbol": pd.Series([], dtype="object"),
                "settled_at": pd.Series([], dtype="int64"),
                "won": pd.Series([], dtype="bool"),
                "settlement_price": pd.Series([], dtype="float64"),
            }
        )

    # Settlement records are for bucket band Yes-legs that settled (price converged to ~1).
    # settled_side_idx is a constant=0 artifact; do NOT use it as a win label.
    # won=True here means the record exists (this band settled), not that it won.
    raw["won"] = True
    raw["settlement_price"] = raw["settle_price"].astype("float64") if "settle_price" in raw.columns else float("nan")
    raw["settled_at"] = raw["settled_at"].astype("int64")

    # De-dup by symbol (keep first)
    result = raw[["symbol", "settled_at", "won", "settlement_price"]].copy()
    result = result.drop_duplicates(subset=["symbol"]).reset_index(drop=True)
    return result
