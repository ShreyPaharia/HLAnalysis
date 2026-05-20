"""Micro-benchmarks for the HL HIP-4 backtest hot paths.

Times each phase in isolation so we know where the remaining wall-clock goes
after the first round of fixes. No pyinstrument involved — the numbers are
raw ``time.perf_counter`` deltas around already-warmed code paths.

Usage::

    HLBT_HL_DATA_ROOT=/path/to/data \
    python scripts/perf/micro_bench.py --kind binary
    python scripts/perf/micro_bench.py --kind bucket
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

from hlanalysis.backtest.core.events import (
    BookSnapshot,
    ReferenceEvent,
    SettlementEvent,
    TradeEvent,
)
from hlanalysis.backtest.data.hl_hip4 import HLHip4DataSource
from hlanalysis.backtest.runner.hftbt_runner import _build_leg_event_array


def _bench_events(ds: HLHip4DataSource, q, repeats: int = 1) -> tuple[float, dict[str, int]]:
    """Time ds.events(q): fetch from parquet + build BookSnapshot/Trade/Ref objects.

    Returns (wall_seconds, count_by_type).
    """
    t0 = time.perf_counter()
    counts = {"book": 0, "trade": 0, "ref": 0, "settle": 0}
    for _ in range(repeats):
        counts = {"book": 0, "trade": 0, "ref": 0, "settle": 0}
        for ev in ds.events(q):
            if isinstance(ev, BookSnapshot):
                counts["book"] += 1
            elif isinstance(ev, TradeEvent):
                counts["trade"] += 1
            elif isinstance(ev, ReferenceEvent):
                counts["ref"] += 1
            elif isinstance(ev, SettlementEvent):
                counts["settle"] += 1
    return (time.perf_counter() - t0) / repeats, counts


def _collect_per_leg(ds: HLHip4DataSource, q) -> tuple[dict, dict, list, list]:
    book_events: dict[str, list[BookSnapshot]] = {s: [] for s in q.leg_symbols}
    trade_events: dict[str, list[TradeEvent]] = {s: [] for s in q.leg_symbols}
    ref_events: list[ReferenceEvent] = []
    settle_events: list[SettlementEvent] = []
    for ev in ds.events(q):
        if isinstance(ev, BookSnapshot):
            if ev.symbol in book_events:
                book_events[ev.symbol].append(ev)
        elif isinstance(ev, TradeEvent):
            if ev.symbol in trade_events:
                trade_events[ev.symbol].append(ev)
        elif isinstance(ev, ReferenceEvent):
            ref_events.append(ev)
        elif isinstance(ev, SettlementEvent):
            settle_events.append(ev)
    return book_events, trade_events, ref_events, settle_events


def _bench_build_arrays(book_events: dict, trade_events: dict, repeats: int = 3) -> tuple[float, int]:
    """Time _build_leg_event_array for every leg, averaged over repeats."""
    total = 0.0
    n_rows_total = 0
    for r in range(repeats):
        n_rows_total = 0
        t0 = time.perf_counter()
        for sym in book_events.keys():
            arr = _build_leg_event_array(book_events[sym], trade_events[sym])
            n_rows_total += arr.shape[0]
        total += time.perf_counter() - t0
    return total / repeats, n_rows_total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["binary", "bucket"], required=True)
    ap.add_argument("--data-root", default=os.environ.get("HLBT_HL_DATA_ROOT", "data"))
    ap.add_argument("--start", default="2026-05-13")
    ap.add_argument("--end", default="2026-05-15")
    args = ap.parse_args()

    ds = HLHip4DataSource(data_root=Path(args.data_root))
    kinds = ("priceBinary",) if args.kind == "binary" else ("priceBucket",)
    qs = ds.discover(start=args.start, end=args.end, kinds=kinds)
    if not qs:
        print(f"No questions for kind={args.kind} in [{args.start}, {args.end})")
        return 1
    q = qs[0]
    print(f"Question {q.question_id} ({q.klass}), legs={len(q.leg_symbols)}")

    # Phase 1: time ds.events()
    t_events, counts = _bench_events(ds, q)
    print(f"\n[1] ds.events(q): {t_events:.2f}s  counts={counts}")

    # Collect once for phase 2.
    book_events, trade_events, _, _ = _collect_per_leg(ds, q)
    n_book = sum(len(v) for v in book_events.values())
    n_trade = sum(len(v) for v in trade_events.values())
    print(f"    collected per-leg: book={n_book}, trade={n_trade}")

    # Phase 2: time _build_leg_event_array
    t_build, n_rows = _bench_build_arrays(book_events, trade_events, repeats=3)
    print(f"[2] _build_leg_event_array (avg of 3): {t_build:.2f}s  out_rows={n_rows}")

    # Phase 3: profile internals of build to see where time goes
    # Time the inner steps for one leg
    if book_events:
        sym0 = next(iter(book_events))
        snaps = book_events[sym0]
        trades = trade_events[sym0]
        if snaps:
            t0 = time.perf_counter()
            for _ in range(5):
                _build_leg_event_array(snaps, trades)
            t_one_leg = (time.perf_counter() - t0) / 5
            print(f"    one leg ({sym0}) snaps={len(snaps)} trades={len(trades)}: {t_one_leg:.3f}s")

    print()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
