"""Calibrate per-price-bucket liquidity parameters from recorded PM L2 book snapshots.

Reads the recorded ``book_snapshot`` parquet partitions (coverage from
2026-05-27), computes the median half-spread and depth per price bucket, and
writes a JSON profile consumable by ``LiquidityProfile`` / ``trade_to_l2``.

Usage::

    python -m scripts.calibrate_pm_liquidity \\
        --cache-root data/sim \\
        --book-root data \\
        --bucket-width 0.05 \\
        --out config/pm_liquidity_profile.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Iterator


# ---------------------------------------------------------------------------
# Pure calibration function (unit-testable)
# ---------------------------------------------------------------------------

def calibrate_from_snapshots(
    rows: list[dict],
    *,
    bucket_width: float,
) -> dict:
    """Calibrate per-bucket liquidity from a list of snapshot dicts.

    Parameters
    ----------
    rows:
        Each dict must have keys ``"p"`` (mid price in [0, 1]),
        ``"half_spread"`` (float), ``"depth"`` (float).
    bucket_width:
        Width of each price bucket (e.g. 0.05 → 20 buckets covering [0, 1)).

    Returns
    -------
    dict with keys:
        ``bucket_width``, ``half_spread`` (per-bucket median or None),
        ``depth`` (per-bucket median or None),
        ``global_half_spread`` (median of all, fallback 0.005),
        ``global_depth`` (median of all, fallback 10000.0),
        and a ``_meta`` dict with ``n_snapshots``.
    """
    n_buckets = round(1.0 / bucket_width)

    # Collect values per bucket.
    hs_per_bucket: list[list[float]] = [[] for _ in range(n_buckets)]
    depth_per_bucket: list[list[float]] = [[] for _ in range(n_buckets)]
    all_hs: list[float] = []
    all_depth: list[float] = []

    for row in rows:
        p = float(row["p"])
        hs = float(row["half_spread"])
        d = float(row["depth"])
        idx = min(int(p / bucket_width), n_buckets - 1)
        hs_per_bucket[idx].append(hs)
        depth_per_bucket[idx].append(d)
        all_hs.append(hs)
        all_depth.append(d)

    global_hs = statistics.median(all_hs) if all_hs else 0.005
    global_depth = statistics.median(all_depth) if all_depth else 10000.0

    half_spread_out: list[float | None] = []
    depth_out: list[float | None] = []
    for i in range(n_buckets):
        if hs_per_bucket[i]:
            half_spread_out.append(statistics.median(hs_per_bucket[i]))
        else:
            half_spread_out.append(None)
        if depth_per_bucket[i]:
            depth_out.append(statistics.median(depth_per_bucket[i]))
        else:
            depth_out.append(None)

    return {
        "bucket_width": bucket_width,
        "half_spread": half_spread_out,
        "depth": depth_out,
        "global_half_spread": global_hs,
        "global_depth": global_depth,
        "_meta": {"n_snapshots": len(rows)},
    }


# ---------------------------------------------------------------------------
# Manifest reader
# ---------------------------------------------------------------------------

def _crypto_token_ids(manifest_path: Path) -> set[str]:
    """Extract all yes/no token IDs from the sim manifest JSON.

    For ``kind=="binary"`` entries: ``market.yes_token_id`` + ``no_token_id``.
    For ``kind=="bucket"`` entries: every token in ``bucket.leg_tokens``
    (list of [yes_tok, no_tok] pairs).
    """
    if not manifest_path.exists():
        return set()
    with open(manifest_path) as f:
        manifest: dict = json.load(f)

    token_ids: set[str] = set()
    for entry in manifest.values():
        kind = entry.get("kind", "binary")
        if kind == "binary":
            mk = entry.get("market", {})
            yes = mk.get("yes_token_id")
            no = mk.get("no_token_id")
            if yes:
                token_ids.add(str(yes))
            if no:
                token_ids.add(str(no))
        elif kind == "bucket":
            b = entry.get("bucket", {})
            leg_tokens: list[list[str]] = b.get("leg_tokens", [])
            for pair in leg_tokens:
                for tok in pair:
                    token_ids.add(str(tok))
    return token_ids


# ---------------------------------------------------------------------------
# Recorded book reader
# ---------------------------------------------------------------------------

_PM_BOOK_DATA_SUBPATH = (
    "venue=polymarket/product_type=prediction_binary/mechanism=clob/event=book_snapshot"
)


def _normalize_levels(
    px: list[float] | None, sz: list[float] | None, *, descending: bool,
) -> list[tuple[float, float]]:
    """Sort (px, sz) level pairs; descending=True for bids (best = max)."""
    if not px:
        return []
    sz = sz or []
    levels = [
        (float(px[i]), float(sz[i]) if i < len(sz) else 0.0)
        for i in range(len(px))
    ]
    levels.sort(key=lambda lv: lv[0], reverse=descending)
    return levels


def _iter_snapshots(
    book_root: Path, token_ids: set[str],
) -> Iterator[dict]:
    """Yield ``{"p": mid, "half_spread": half_spread, "depth": top_depth}``
    for each recorded book snapshot row across all supplied token IDs.

    Uses the same duckdb query pattern as ``_load_recorded_book``.  Skips
    rows where best bid / ask are missing or ask <= bid (crossed book).
    Top-level depth = size at best bid (the first level after normalization).

    The parquet columns are:
        exchange_ts  int64
        bid_px       list<float>   (may be None / empty)
        bid_sz       list<float>
        ask_px       list<float>
        ask_sz       list<float>
    """
    import duckdb
    from glob import glob as _glob

    for token_id in token_ids:
        book_glob = str(
            book_root / _PM_BOOK_DATA_SUBPATH / f"symbol={token_id}" / "**" / "*.parquet"
        )
        if not _glob(book_glob, recursive=True):
            continue
        con = duckdb.connect()
        try:
            rows = con.sql(
                f"""
                SELECT bid_px, bid_sz, ask_px, ask_sz
                FROM read_parquet('{book_glob}', hive_partitioning=1)
                ORDER BY exchange_ts
                """
            ).fetchall()
        finally:
            con.close()

        for bid_px_raw, bid_sz_raw, ask_px_raw, ask_sz_raw in rows:
            bids = _normalize_levels(bid_px_raw, bid_sz_raw, descending=True)
            asks = _normalize_levels(ask_px_raw, ask_sz_raw, descending=False)
            if not bids or not asks:
                continue
            best_bid_px, best_bid_sz = bids[0]
            best_ask_px, best_ask_sz = asks[0]
            if best_ask_px <= best_bid_px:
                continue
            mid = (best_bid_px + best_ask_px) / 2.0
            half_spread = (best_ask_px - best_bid_px) / 2.0
            # The synthetic book is symmetric (bid_sz == ask_sz == depth) and a
            # taker walks it from either side, so use the conservative min of the
            # two top-of-book sizes as the representative depth (N2).
            depth = min(best_bid_sz, best_ask_sz)
            yield {"p": mid, "half_spread": half_spread, "depth": depth}


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate PM synthetic-liquidity profile from recorded book snapshots."
    )
    parser.add_argument(
        "--cache-root", default="data/sim",
        help="PM cache root containing manifest.json (default: data/sim).",
    )
    parser.add_argument(
        "--book-root", default="data",
        help="Root directory containing recorded book_snapshot parquet "
        "(default: data).",
    )
    parser.add_argument(
        "--bucket-width", type=float, default=0.05,
        help="Price bucket width for calibration (default: 0.05 → 20 buckets).",
    )
    parser.add_argument(
        "--out", default="config/pm_liquidity_profile.json",
        help="Output JSON path (default: config/pm_liquidity_profile.json).",
    )
    args = parser.parse_args()

    cache_root = Path(args.cache_root)
    book_root = Path(args.book_root)
    out_path = Path(args.out)

    manifest_path = cache_root / "manifest.json"
    token_ids = _crypto_token_ids(manifest_path)
    print(f"[calibrate] found {len(token_ids)} token IDs from manifest")

    rows = list(_iter_snapshots(book_root, token_ids))
    print(f"[calibrate] loaded {len(rows)} book snapshots")

    profile = calibrate_from_snapshots(rows, bucket_width=args.bucket_width)
    profile["_meta"]["n_tokens"] = len(token_ids)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(profile, f, indent=2)

    print(
        f"[calibrate] wrote profile → {out_path}  "
        f"(global_half_spread={profile['global_half_spread']:.4f}, "
        f"global_depth={profile['global_depth']:.1f}, "
        f"n_tokens={profile['_meta']['n_tokens']}, "
        f"n_snapshots={profile['_meta']['n_snapshots']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
