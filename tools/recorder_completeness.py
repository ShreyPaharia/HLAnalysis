"""Recorder completeness / seq-gap reconciliation tool (SHR-84).

Read-only audit over the recorder's Hive-partitioned parquet that answers one
question: *is the recorded feed the sim replays actually complete?* A unified
MarketState core still computes wrong σ / volume if its input event stream is
missing events, so before a sim run trusts a window we want to assert the
corpus for that window is whole.

For each symbol/day (over `event=book_snapshot|trade` by default) it checks:

  * **seq monotonicity** — per-symbol `seq` gaps: how many integers are missing
    from the recorded sequence, the largest single hole, and duplicate counts.
  * **time gaps** — the longest inter-event gap per symbol/hour.
  * **message-count / notional sanity** — flags suspiciously quiet windows
    (fewer than `min_events`, or trade notional below `min_notional`).
  * **per-question leg coverage** — given each question's `[start,end]` window
    and its leg symbols, whether every leg has recorded data spanning the
    window (HL HIP-4 questions are discovered opportunistically from the same
    recorded corpus when available).

The result is emitted as a machine-readable JSON report plus a human summary,
so a downstream caller can gate on ``report["summary"]["complete"]``.

This module is import-safe (no work at import time) and lives under ``tools/``
which CLAUDE.md notes is imported as a module by tests — keep it that way.

Run:
    uv run python tools/recorder_completeness.py --data-root data --json report.json
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import json
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa

from hlanalysis.recorder.read import read_recorded

NS_PER_S = 1_000_000_000

# Events that carry a meaningful per-symbol `seq` and drive the sim's book/σ/
# volume inputs. Other event types (mark/funding/...) are low-cadence and
# checked only on request.
DEFAULT_EVENTS: tuple[str, ...] = ("book_snapshot", "trade")


# --------------------------------------------------------------------------- #
# Result records (all dataclasses so the report serialises cleanly to JSON).
# --------------------------------------------------------------------------- #
@dataclasses.dataclass(frozen=True)
class SeqGapStats:
    """Pure seq-monotonicity summary for one ordered (or unordered) seq stream."""

    n_total: int
    n_with_seq: int
    n_no_seq: int
    n_gaps: int
    n_missing: int
    largest_gap: int
    n_dup: int
    first_seq: int | None
    last_seq: int | None

    @property
    def complete(self) -> bool:
        # Completeness == no missing sequence numbers. Duplicates (HL re-publishes
        # recent trades after a reconnect) and null seq are recorded but do not
        # mark the window incomplete.
        return self.n_missing == 0


@dataclasses.dataclass(frozen=True)
class TimeGap:
    longest_gap_s: float
    gap_start_ns: int | None
    gap_end_ns: int | None
    n_events: int


@dataclasses.dataclass(frozen=True)
class SeqGapReport:
    event: str
    symbol: str
    n_total: int
    n_with_seq: int
    n_no_seq: int
    n_gaps: int
    n_missing: int
    largest_gap: int
    n_dup: int
    complete: bool


@dataclasses.dataclass(frozen=True)
class TimeGapReport:
    event: str
    symbol: str
    hour: str
    n_events: int
    longest_gap_s: float
    gap_start_ns: int | None
    gap_end_ns: int | None


@dataclasses.dataclass(frozen=True)
class VolumeReport:
    event: str
    symbol: str
    hour: str
    n_events: int
    notional: float | None
    quiet: bool


@dataclasses.dataclass(frozen=True)
class QuestionSpec:
    """Minimal question descriptor for the leg-coverage cross-check."""

    question_id: str
    start_ts_ns: int
    end_ts_ns: int
    leg_symbols: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class LegCoverageReport:
    question_id: str
    leg: str
    present: bool
    covered: bool
    first_ts_ns: int | None
    last_ts_ns: int | None
    gap_after_start_s: float
    gap_before_end_s: float


# --------------------------------------------------------------------------- #
# Pure analysis primitives.
# --------------------------------------------------------------------------- #
def seq_gap_stats(seqs: Iterable[int | None]) -> SeqGapStats:
    """Summarise sequence-number completeness over a stream of `seq` values.

    Order-independent: gaps are computed from the set of observed sequence
    numbers, so out-of-order arrival (or concatenation across files) is not
    mistaken for a hole. ``None`` values (events recorded without a seq) are
    counted but excluded from gap math.
    """
    items = list(seqs)  # materialise once so we can count Nones and gaps
    vals = [int(s) for s in items if s is not None]
    n_total = len(items)
    n_with_seq = len(vals)
    n_no_seq = n_total - n_with_seq

    uniq = sorted(set(vals))
    n_dup = n_with_seq - len(uniq)
    n_gaps = 0
    n_missing = 0
    largest_gap = 0
    for a, b in zip(uniq, uniq[1:]):
        delta = b - a
        if delta > 1:
            missing = delta - 1
            n_gaps += 1
            n_missing += missing
            largest_gap = max(largest_gap, missing)
    return SeqGapStats(
        n_total=n_total,
        n_with_seq=n_with_seq,
        n_no_seq=n_no_seq,
        n_gaps=n_gaps,
        n_missing=n_missing,
        largest_gap=largest_gap,
        n_dup=n_dup,
        first_seq=uniq[0] if uniq else None,
        last_seq=uniq[-1] if uniq else None,
    )


def longest_time_gap(ts_ns: Sequence[int]) -> TimeGap:
    """Longest inter-event gap (seconds) over a set of event timestamps."""
    ts = sorted(int(t) for t in ts_ns if t is not None)
    if len(ts) < 2:
        return TimeGap(longest_gap_s=0.0, gap_start_ns=None, gap_end_ns=None, n_events=len(ts))
    best = 0
    start = end = None
    for a, b in zip(ts, ts[1:]):
        d = b - a
        if d > best:
            best = d
            start, end = a, b
    return TimeGap(
        longest_gap_s=best / NS_PER_S,
        gap_start_ns=start,
        gap_end_ns=end,
        n_events=len(ts),
    )


def question_leg_coverage(
    questions: Iterable[QuestionSpec],
    spans: dict[str, tuple[int, int]],
    *,
    edge_tolerance_s: float = 5.0,
) -> list[LegCoverageReport]:
    """Cross-check that every leg of each question has data over ``[start,end]``.

    ``spans`` maps symbol -> (first_ts_ns, last_ts_ns) of recorded events. A leg
    is *covered* when its data begins no later than ``start + tolerance`` and
    ends no earlier than ``end - tolerance``.
    """
    tol_ns = int(edge_tolerance_s * NS_PER_S)
    out: list[LegCoverageReport] = []
    for q in questions:
        for leg in q.leg_symbols:
            span = spans.get(leg)
            if span is None:
                out.append(
                    LegCoverageReport(
                        question_id=q.question_id, leg=leg, present=False, covered=False,
                        first_ts_ns=None, last_ts_ns=None,
                        gap_after_start_s=0.0, gap_before_end_s=0.0,
                    )
                )
                continue
            first, last = span
            gap_after_start = max(0, first - q.start_ts_ns)
            gap_before_end = max(0, q.end_ts_ns - last)
            covered = (first <= q.start_ts_ns + tol_ns) and (last >= q.end_ts_ns - tol_ns)
            out.append(
                LegCoverageReport(
                    question_id=q.question_id, leg=leg, present=True, covered=covered,
                    first_ts_ns=first, last_ts_ns=last,
                    gap_after_start_s=gap_after_start / NS_PER_S,
                    gap_before_end_s=gap_before_end / NS_PER_S,
                )
            )
    return out


# --------------------------------------------------------------------------- #
# Table-level analysis.
# --------------------------------------------------------------------------- #
def _column(table: pa.Table, name: str) -> list[Any] | None:
    if name not in table.column_names:
        return None
    return table.column(name).to_pylist()


def _event_ts(exchange: int | None, local: int | None) -> int | None:
    """Pick the timestamp the recorder partitions on: exchange_ts when set
    (>0), else local_recv_ts. Binance spot has no exchange ts (recorded 0)."""
    if exchange:
        return int(exchange)
    if local:
        return int(local)
    return None


def _hour_key(ts_ns: int) -> str:
    dt = _dt.datetime.fromtimestamp(ts_ns / NS_PER_S, tz=_dt.timezone.utc)
    return dt.strftime("%Y-%m-%dT%H")


def analyze_table(
    table: pa.Table,
    *,
    event: str,
    min_events: int = 1,
    min_notional: float = 0.0,
) -> tuple[list[SeqGapReport], list[TimeGapReport], list[VolumeReport]]:
    """Compute seq-gap, time-gap and volume reports for one event's table.

    Groups by the in-file ``symbol`` column (the recorder writes it as a real
    column, not just a path partition). Time-gap and volume reports are per
    (symbol, hour); seq reports are per symbol.
    """
    n = table.num_rows
    symbols = _column(table, "symbol") or [""] * n
    seqs = _column(table, "seq") or [None] * n
    exch = _column(table, "exchange_ts") or [None] * n
    local = _column(table, "local_recv_ts") or [None] * n
    prices = _column(table, "price")
    sizes = _column(table, "size")

    by_symbol_seq: dict[str, list[int | None]] = defaultdict(list)
    by_hour_ts: dict[tuple[str, str], list[int]] = defaultdict(list)
    by_hour_notional: dict[tuple[str, str], float] = defaultdict(float)
    by_hour_has_notional: dict[tuple[str, str], bool] = defaultdict(bool)

    for i in range(n):
        sym = symbols[i]
        by_symbol_seq[sym].append(seqs[i])
        ts = _event_ts(exch[i], local[i])
        if ts is None:
            continue
        hour = _hour_key(ts)
        key = (sym, hour)
        by_hour_ts[key].append(ts)
        if prices is not None and sizes is not None:
            p, s = prices[i], sizes[i]
            if p is not None and s is not None:
                by_hour_notional[key] += float(p) * float(s)
                by_hour_has_notional[key] = True

    seq_reports = [
        _to_seq_report(event, sym, seq_gap_stats(vals))
        for sym, vals in sorted(by_symbol_seq.items())
    ]

    time_reports: list[TimeGapReport] = []
    volume_reports: list[VolumeReport] = []
    for (sym, hour), ts_list in sorted(by_hour_ts.items()):
        g = longest_time_gap(ts_list)
        time_reports.append(
            TimeGapReport(
                event=event, symbol=sym, hour=hour, n_events=g.n_events,
                longest_gap_s=g.longest_gap_s,
                gap_start_ns=g.gap_start_ns, gap_end_ns=g.gap_end_ns,
            )
        )
        n_events = len(ts_list)
        has_ntl = by_hour_has_notional[(sym, hour)]
        notional = by_hour_notional[(sym, hour)] if has_ntl else None
        quiet = (n_events < min_events) or (
            notional is not None and notional < min_notional
        )
        volume_reports.append(
            VolumeReport(
                event=event, symbol=sym, hour=hour, n_events=n_events,
                notional=notional, quiet=quiet,
            )
        )
    return seq_reports, time_reports, volume_reports


def _to_seq_report(event: str, symbol: str, s: SeqGapStats) -> SeqGapReport:
    return SeqGapReport(
        event=event, symbol=symbol,
        n_total=s.n_total, n_with_seq=s.n_with_seq, n_no_seq=s.n_no_seq,
        n_gaps=s.n_gaps, n_missing=s.n_missing, largest_gap=s.largest_gap,
        n_dup=s.n_dup, complete=s.complete,
    )


# --------------------------------------------------------------------------- #
# Corpus-level driver.
# --------------------------------------------------------------------------- #
def _event_glob(data_root: Path, event: str) -> str:
    return str(
        data_root
        / "venue=*" / "product_type=*" / "mechanism=*"
        / f"event={event}" / "symbol=*" / "date=*" / "hour=*" / "*.parquet"
    )


def build_completeness_report(
    data_root: Path | str,
    *,
    events: Sequence[str] = DEFAULT_EVENTS,
    min_events: int = 1,
    min_notional: float = 0.0,
    questions: Iterable[QuestionSpec] | None = None,
) -> dict[str, Any]:
    """Read the recorded parquet under ``data_root`` and assemble the report.

    Returns a JSON-serialisable dict with ``seq_gaps``, ``time_gaps``,
    ``volume``, ``question_coverage`` lists and a ``summary`` block whose
    ``complete`` flag is the single gate a sim run can assert on.
    """
    data_root = Path(data_root)
    seq_reports: list[SeqGapReport] = []
    time_reports: list[TimeGapReport] = []
    volume_reports: list[VolumeReport] = []
    spans: dict[str, tuple[int, int]] = {}

    for event in events:
        glob = _event_glob(data_root, event)
        try:
            table = read_recorded(glob)
        except FileNotFoundError:
            continue
        s_reps, t_reps, v_reps = analyze_table(
            table, event=event, min_events=min_events, min_notional=min_notional
        )
        seq_reports.extend(s_reps)
        time_reports.extend(t_reps)
        volume_reports.extend(v_reps)
        _accumulate_spans(spans, table)

    coverage: list[LegCoverageReport] = []
    if questions is not None:
        coverage = question_leg_coverage(questions, spans)

    symbols = {r.symbol for r in seq_reports}
    n_seq_incomplete = sum(1 for r in seq_reports if not r.complete)
    n_quiet = sum(1 for r in volume_reports if r.quiet)
    n_uncovered = sum(1 for r in coverage if not r.covered)
    complete = n_seq_incomplete == 0 and n_uncovered == 0

    return {
        "data_root": str(data_root),
        "events": list(events),
        "seq_gaps": [dataclasses.asdict(r) for r in seq_reports],
        "time_gaps": [dataclasses.asdict(r) for r in time_reports],
        "volume": [dataclasses.asdict(r) for r in volume_reports],
        "question_coverage": [dataclasses.asdict(r) for r in coverage],
        "summary": {
            "complete": complete,
            "n_symbols": len(symbols),
            "n_seq_incomplete": n_seq_incomplete,
            "n_seq_missing_total": sum(r.n_missing for r in seq_reports),
            "n_quiet_windows": n_quiet,
            "n_questions": len(list(questions)) if questions is not None else 0,
            "n_uncovered_legs": n_uncovered,
        },
    }


def _accumulate_spans(spans: dict[str, tuple[int, int]], table: pa.Table) -> None:
    n = table.num_rows
    symbols = _column(table, "symbol") or [""] * n
    exch = _column(table, "exchange_ts") or [None] * n
    local = _column(table, "local_recv_ts") or [None] * n
    for i in range(n):
        ts = _event_ts(exch[i], local[i])
        if ts is None:
            continue
        sym = symbols[i]
        cur = spans.get(sym)
        if cur is None:
            spans[sym] = (ts, ts)
        else:
            spans[sym] = (min(cur[0], ts), max(cur[1], ts))


def discover_hl_questions(
    data_root: Path | str, *, start: str, end: str, underlying: str = "BTC"
) -> list[QuestionSpec]:
    """Best-effort discovery of HL HIP-4 question windows from the recorded
    corpus, reusing the backtest data source. Returns ``[]`` if the source or
    its metadata is unavailable so the tool stays usable without it."""
    try:
        from hlanalysis.backtest.data.hl_hip4 import HLHip4DataSource
    except Exception:  # pragma: no cover - optional dependency path
        return []
    try:
        src = HLHip4DataSource(data_root)
        descs = src.discover(start=start, end=end, underlying=underlying)
    except Exception:  # pragma: no cover - no/partial metadata
        return []
    return [
        QuestionSpec(
            question_id=d.question_id,
            start_ts_ns=d.start_ts_ns,
            end_ts_ns=d.end_ts_ns,
            leg_symbols=tuple(d.leg_symbols),
        )
        for d in descs
    ]


# --------------------------------------------------------------------------- #
# Summary formatting + CLI.
# --------------------------------------------------------------------------- #
def format_summary(report: dict[str, Any]) -> str:
    s = report["summary"]
    lines: list[str] = []
    verdict = "COMPLETE" if s["complete"] else "INCOMPLETE"
    lines.append(f"Recorder completeness: {verdict}")
    lines.append(f"  data_root        : {report['data_root']}")
    lines.append(f"  events           : {', '.join(report['events'])}")
    lines.append(f"  symbols          : {s['n_symbols']}")
    lines.append(
        f"  seq-incomplete   : {s['n_seq_incomplete']} symbol(s), "
        f"{s['n_seq_missing_total']} missing seq total"
    )
    lines.append(f"  quiet windows    : {s['n_quiet_windows']}")
    lines.append(
        f"  question legs    : {s['n_uncovered_legs']} uncovered "
        f"of {s['n_questions']} question(s)"
    )

    incomplete = [r for r in report["seq_gaps"] if not r["complete"]]
    if incomplete:
        lines.append("  --- seq gaps ---")
        for r in sorted(incomplete, key=lambda r: -r["n_missing"])[:20]:
            lines.append(
                f"    {r['event']:<14} {r['symbol']:<16} "
                f"missing={r['n_missing']} largest={r['largest_gap']} gaps={r['n_gaps']}"
            )

    longest = sorted(report["time_gaps"], key=lambda r: -r["longest_gap_s"])[:10]
    if longest:
        lines.append("  --- longest time gaps ---")
        for r in longest:
            lines.append(
                f"    {r['longest_gap_s']:>8.1f}s  {r['event']:<14} "
                f"{r['symbol']:<16} hour={r['hour']}"
            )

    uncovered = [r for r in report["question_coverage"] if not r["covered"]]
    if uncovered:
        lines.append("  --- uncovered question legs ---")
        for r in uncovered[:20]:
            why = "absent" if not r["present"] else (
                f"starts +{r['gap_after_start_s']:.0f}s / ends -{r['gap_before_end_s']:.0f}s"
            )
            lines.append(f"    {r['question_id']:<22} {r['leg']:<16} {why}")

    return "\n".join(lines)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default="data", help="recorder data root (default: data)")
    p.add_argument(
        "--events", nargs="+", default=list(DEFAULT_EVENTS),
        help=f"event types to check (default: {' '.join(DEFAULT_EVENTS)})",
    )
    p.add_argument("--min-events", type=int, default=1, help="flag hourly windows with fewer events")
    p.add_argument("--min-notional", type=float, default=0.0, help="flag trade windows below this notional")
    p.add_argument("--json", default=None, help="write machine-readable report to this path")
    p.add_argument("--start", default=None, help="ISO start for HL question leg-coverage cross-check")
    p.add_argument("--end", default=None, help="ISO end for HL question leg-coverage cross-check")
    p.add_argument("--underlying", default="BTC", help="underlying for HL question discovery")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    questions: list[QuestionSpec] | None = None
    if args.start and args.end:
        questions = discover_hl_questions(
            args.data_root, start=args.start, end=args.end, underlying=args.underlying
        )
    report = build_completeness_report(
        args.data_root,
        events=tuple(args.events),
        min_events=args.min_events,
        min_notional=args.min_notional,
        questions=questions,
    )
    print(format_summary(report))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.json}")
    # Exit non-zero when the corpus is incomplete so callers can gate on it.
    return 0 if report["summary"]["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
