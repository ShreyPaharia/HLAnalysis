"""Per-decision sim-vs-live replay report (decision-granularity fidelity).

IO shell over ``hlanalysis.parity.decision_replay``. Loads the LIVE trade journal
(SHR-83 ``trade_journal`` in a slot's ``state.db``) and the SIM evaluate()
diagnostics (``diagnostics.parquet`` from an ``hl-bt run``), aligns them on
``(question_idx, ts_ns)``, and reports:

  * decision-match rate — of the live orders, how many the sim also emitted
    (same action, same question, within --ts-tol-seconds);
  * per-field input skew (σ / reference_price / p_model / edge) at the moments
    live actually decided — the decision-layer fidelity number;
  * phantom sim actions (sim entered/exited where live never did) and unmatched
    live decisions.

The state.db is read strictly read-only (never touches the live engine). Pull it
SSM-free with ``scripts/pull-engine.sh`` (the daily engine→S3 snapshot).

Usage:
    uv run python tools/decision_replay_report.py \
        --state-db data/engine/date=2026-06-11/v31/state.db \
        --sim-run  data/sim/runs/m_v31_binary_0610 \
        [--ts-tol-seconds 2.0] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import duckdb

from hlanalysis.parity.decision_replay import (
    LiveDecision,
    SimTick,
    format_report,
    replay,
)

_SENTINEL = -1e8  # edge_yes/edge_no use -1e9 for "not the favorite side"


def _favorite_edge(edge_yes: float | None, edge_no: float | None) -> float | None:
    """The edge of the side the strategy actually considered (drops the sentinel)."""
    cands = [e for e in (edge_yes, edge_no) if e is not None and e == e and e > _SENTINEL]
    return max(cands) if cands else None


def _parse_journal_diag(
    diag_json: str | None,
) -> tuple[float | None, float | None, float | None]:
    """Pull (p_model, favorite_edge, sigma) out of a journal row's diagnostics_json.

    The 'edge' diagnostic carries [p_model, edge_yes, edge_no, sigma, ...] as
    (name, formatted-str) pairs. σ here is the ANNUALIZED evaluate() σ — the
    apples-to-apples value vs the sim's diagnostics σ. (The journal's ``sigma``
    *column* is NOT comparable: it falls back to raw returns-std on exit rows,
    which carry no 'edge' block.) Exit rows → (None, None, None)."""
    if not diag_json:
        return None, None, None
    try:
        blocks = json.loads(diag_json)
    except (ValueError, TypeError):
        return None, None, None
    for b in blocks:
        if b.get("message") == "edge":
            kv = {k: v for k, v in b.get("fields", [])}
            def _f(x):
                try:
                    return float(x)
                except (TypeError, ValueError):
                    return None
            return (
                _f(kv.get("p_model")),
                _favorite_edge(_f(kv.get("edge_yes")), _f(kv.get("edge_no"))),
                _f(kv.get("sigma")),
            )
    return None, None, None


def _book_event_ts(book_json: str | None) -> int | None:
    """The book-event timestamp the engine reacted to (``last_l2_ts_ns``).

    This is the SAME clock as the sim's ts_ns (recorded book exchange_ts). The
    journal's ``decision_ts_ns`` is instead the engine's wall-clock at scan time —
    ~1.4 s later (ingest→evaluate latency) — so aligning on it misjoins the sim
    by tens of seconds. Align on the book event instead."""
    if not book_json:
        return None
    try:
        b = json.loads(book_json)
    except (ValueError, TypeError):
        return None
    v = b.get("last_l2_ts_ns")
    return int(v) if v is not None else None


def load_live(state_db: Path, question_idxs: set[int], ts_lo: int, ts_hi: int) -> list[LiveDecision]:
    con = sqlite3.connect(f"file:{state_db}?mode=ro", uri=True)
    qmarks = ",".join("?" for _ in question_idxs)
    # Filter on decision_ts (the indexed column) with slack, then re-key each row
    # to its book-event ts for the actual sim alignment.
    slack = 10 * 1_000_000_000
    rows = con.execute(
        f"""SELECT question_idx, decision_ts_ns, action, symbol, reference_price,
                   sigma, diagnostics_json, book_json
            FROM trade_journal
            WHERE question_idx IN ({qmarks})
              AND decision_ts_ns >= ? AND decision_ts_ns <= ?
              AND action IN ('enter','exit')
            ORDER BY decision_ts_ns""",
        (*question_idxs, ts_lo - slack, ts_hi + slack),
    ).fetchall()
    con.close()
    out: list[LiveDecision] = []
    for qi, dts, action, symbol, ref, _col_sigma, diag, book in rows:
        p_model, edge, sigma = _parse_journal_diag(diag)
        ts = _book_event_ts(book)
        if ts is None:
            ts = int(dts)  # fallback: no book snapshot on this row
        out.append(LiveDecision(
            question_idx=int(qi), ts_ns=ts, action=action, symbol=symbol or "",
            sigma=sigma, reference_price=ref, p_model=p_model, edge=edge,
        ))
    return out


def load_sim(sim_run: Path) -> list[SimTick]:
    diag = sim_run / "diagnostics.parquet"
    if not diag.exists():
        raise SystemExit(f"sim diagnostics not found: {diag}")
    # Isolated connection: a failure here must not poison a shared global
    # duckdb transaction used by other callers (e.g. the daily report).
    df = duckdb.connect().execute(
        f"""SELECT ts_ns, question_idx, action, sigma, ref_price,
                   p_model, edge_yes, edge_no
            FROM read_parquet('{diag}') ORDER BY ts_ns"""
    ).df()
    out: list[SimTick] = []
    for r in df.itertuples(index=False):
        def _n(x):
            return None if x is None or x != x else float(x)
        out.append(SimTick(
            question_idx=int(r.question_idx), ts_ns=int(r.ts_ns), action=str(r.action),
            sigma=_n(r.sigma), reference_price=_n(r.ref_price), p_model=_n(r.p_model),
            edge=_favorite_edge(_n(r.edge_yes), _n(r.edge_no)),
        ))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--state-db", required=True, type=Path, help="slot state.db (read-only)")
    p.add_argument("--sim-run", required=True, type=Path, help="hl-bt run dir (has diagnostics.parquet)")
    p.add_argument("--ts-tol-seconds", type=float, default=2.0,
                   help="match tolerance between a live decision and a sim action (default 2.0)")
    p.add_argument("--json", type=Path, default=None, help="optional machine-readable output path")
    args = p.parse_args(argv)

    sim = load_sim(args.sim_run)
    if not sim:
        raise SystemExit("sim diagnostics empty")
    qidxs = {t.question_idx for t in sim}
    ts_lo = min(t.ts_ns for t in sim)
    ts_hi = max(t.ts_ns for t in sim)
    live = load_live(args.state_db, qidxs, ts_lo, ts_hi)

    rep = replay(live, sim, ts_tol_ns=int(args.ts_tol_seconds * 1e9))
    print(f"# decision-replay  slot_db={args.state_db}  sim_run={args.sim_run.name}")
    print(f"# questions={sorted(qidxs)}  window=[{ts_lo},{ts_hi}]  tol={args.ts_tol_seconds}s")
    print(format_report(rep))
    if rep.unmatched_live:
        print("unmatched live decisions (sim did NOT reproduce):")
        for d in rep.unmatched_live[:20]:
            print(f"  q{d.question_idx} {d.symbol} {d.action} @ {d.ts_ns}")
    if rep.phantom_sim:
        print(f"phantom sim actions (no live counterpart): {rep.n_sim_phantom}")
        for t in rep.phantom_sim[:20]:
            print(f"  q{t.question_idx} {t.action} @ {t.ts_ns}")
    if args.json:
        args.json.write_text(json.dumps(rep.to_dict(), indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
