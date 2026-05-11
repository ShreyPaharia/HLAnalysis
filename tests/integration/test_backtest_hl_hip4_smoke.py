"""End-to-end smoke test for the HL HIP-4 DataSource.

Spec acceptance (§4, Task B):
- Smoke test runs end-to-end against the committed fixture and produces a
  non-empty ``report.md``.

Until task A's ``hl-bt`` CLI + runner lands, we exercise the DataSource via a
tiny inline harness that tallies event types and writes a minimal report. The
harness is a stand-in for the real runner — both consume the same §3 interface
so the test won't need behavioural changes once A merges; only the harness
swaps out for ``hl-bt run`` invocation.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from hlanalysis.backtest.core.events import (
    BookSnapshot,
    MarketEvent,
    ReferenceEvent,
    SettlementEvent,
    TradeEvent,
)
from hlanalysis.backtest.data.hl_hip4 import HLHip4DataSource


FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "hl_hip4"


def _write_report(out_dir: Path, descriptor_summary: str, counts: Counter, n: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / "report.md"
    lines = [
        "# HL HIP-4 fixture backtest (smoke)",
        "",
        f"- Question: {descriptor_summary}",
        f"- Total events: {n}",
        "- Counts by kind:",
    ]
    for kind in ("BookSnapshot", "TradeEvent", "ReferenceEvent", "SettlementEvent"):
        lines.append(f"  - {kind}: {counts.get(kind, 0)}")
    lines.append("")
    lines.append("This report is a placeholder produced by the smoke harness while task A's runner is in flight.")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def test_hl_hip4_smoke_end_to_end(tmp_path: Path) -> None:
    src = HLHip4DataSource(data_root=FIXTURE_ROOT)
    questions = src.discover(start="2026-05-09", end="2026-05-11", underlying="BTC")
    assert len(questions) == 1, questions
    q = questions[0]
    qv = src.question_view(q, now_ns=q.start_ts_ns, settled=False)

    counts: Counter[str] = Counter()
    n = 0
    prev = -1
    seen_ref = False
    seen_book = False
    seen_trade = False
    for ev in src.events(q):
        n += 1
        assert ev.ts_ns >= prev
        prev = ev.ts_ns
        counts[type(ev).__name__] += 1
        if isinstance(ev, BookSnapshot):
            seen_book = True
        elif isinstance(ev, TradeEvent):
            seen_trade = True
        elif isinstance(ev, ReferenceEvent):
            seen_ref = True
    assert n >= 1_000, n
    assert seen_book and seen_trade and seen_ref, counts

    descriptor_summary = f"{q.question_id} ({qv.klass}, strike={qv.strike}, legs={list(q.leg_symbols)})"
    report = _write_report(tmp_path, descriptor_summary, counts, n)
    contents = report.read_text(encoding="utf-8")
    assert contents.strip(), "report.md must be non-empty"
    assert "HL HIP-4 fixture backtest" in contents
    assert str(q.question_id) in contents


def test_hl_hip4_resolved_outcome_against_fixture() -> None:
    src = HLHip4DataSource(data_root=FIXTURE_ROOT)
    q = src.discover(start="2026-05-09", end="2026-05-11", underlying="BTC")[0]
    # No settlement event captured in the fixture; binary fallback should
    # produce a definite yes/no using the last HL perp BTC mid in-window.
    assert src.resolved_outcome(q) in ("yes", "no")
