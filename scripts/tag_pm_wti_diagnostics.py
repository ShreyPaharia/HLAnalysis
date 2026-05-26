#!/usr/bin/env python3
"""Tag each PM WTI market with spanned_weekend / spanned_eia booleans.

`spanned_weekend`: market window [start_ts_ns, end_ts_ns] overlaps the
WTI closed window — Fri 21:00 UTC (5pm ET) → Sun 22:00 UTC (6pm ET).
During this window the reference price is frozen → σ stale → p_model
degrades.

`spanned_eia`: EIA Weekly Petroleum Status Report releases at 10:30am
ET on Wednesday (14:30 UTC during US DST). PM WTI corpus is entirely
in DST (Mar 25 → ongoing 2026). Tag true if the market window contains
the Wed 14:00–15:00 UTC hour.

Usage:
    python scripts/tag_pm_wti_diagnostics.py \\
        --cache-root data/sim/pm_wti \\
        --out data/sim/runs/pm_wti_hl_tuned/diagnostics_tags.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


_HOUR_NS = 3600 * 1_000_000_000


def spans_weekend(start_ns: int, end_ns: int) -> bool:
    """Any hour in [start_ns, end_ns] falling inside Fri 21:00Z → Sun 22:00Z."""
    t = start_ns
    while t <= end_ns:
        dt = datetime.fromtimestamp(t / 1e9, tz=timezone.utc)
        wd, h = dt.weekday(), dt.hour  # Mon=0..Sun=6
        if (wd == 4 and h >= 21) or (wd == 5) or (wd == 6 and h < 22):
            return True
        t += _HOUR_NS
    return False


def spans_eia(start_ns: int, end_ns: int) -> bool:
    """Wed 14:00–14:59 UTC (EIA release window during US DST)."""
    t = start_ns
    while t <= end_ns:
        dt = datetime.fromtimestamp(t / 1e9, tz=timezone.utc)
        if dt.weekday() == 2 and dt.hour == 14:
            return True
        t += _HOUR_NS
    return False


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-root", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    manifest = json.loads((Path(args.cache_root) / "manifest.json").read_text())

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["question_id", "end_date", "spanned_weekend", "spanned_eia"])
        for qid, entry in manifest.items():
            if entry.get("kind") != "binary":
                continue
            mk = entry["market"]
            s, e = int(mk["start_ts_ns"]), int(mk["end_ts_ns"])
            end_date = datetime.fromtimestamp(e / 1e9, tz=timezone.utc).strftime("%Y-%m-%d")
            w.writerow([qid, end_date, int(spans_weekend(s, e)), int(spans_eia(s, e))])
    return 0


if __name__ == "__main__":
    sys.exit(main())
