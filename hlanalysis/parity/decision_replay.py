"""Per-decision sim-vs-live replay — pure core (decision-granularity fidelity).

SHR-90 (`validation.py`) reconciles **market-level PnL** and triages the residual
with one representative σ/reference per market. That is too coarse to tell a
decision-layer bug from an execution one: a single first-decision σ within 5 %
can hide plenty of per-order divergence. This module goes one level finer.

For each moment LIVE actually decided — a trade-journal row, carrying the exact
evaluate() inputs the engine saw (σ / reference_price / p_model / edge) and the
action it took — it asks:

* **Did the sim see the same inputs?** Compare the live decision's captured
  inputs against the sim's evaluate() diagnostics at the nearest tick. Because
  sim and live run the *same* unified evaluate() (SHR-97), a *decision*
  divergence can only come from an *input* divergence — so the per-field input
  skew at matched decision points IS the decision-layer fidelity measurement.
* **Did the sim take the same action?** A live `enter`/`exit` is *matched* if the
  sim emitted the same action for the same question within a timestamp tolerance.
  A sim action with no live counterpart is a **phantom** (the over-entry /
  SHR-91 signature); a live action the sim never took is an **unmatched live
  decision** (a gate the sim evaluated differently).

This is the PURE core: it takes already-extracted :class:`LiveDecision`s and
:class:`SimTick`s and emits the report. The IO that loads the live trade journal
(sqlite ``state.db``) and the sim ``diagnostics.parquet`` lives in the ``tools``
shell so the matching/attribution logic stays unit-testable on synthetic inputs.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import Any

# Fields compared at each matched decision point. ``rel`` fields also report a
# relative divergence (|a-b|/|live|); absolute-only fields (already in [0,1] or
# a difference of rates) report abs alone.
_REL_FIELDS = ("sigma", "reference_price")
_ABS_ONLY_FIELDS = ("p_model", "edge")
_ALL_FIELDS = _REL_FIELDS + _ABS_ONLY_FIELDS


@dataclass(frozen=True, slots=True)
class LiveDecision:
    """One live trade-journal decision with its captured evaluate() inputs."""

    question_idx: int
    ts_ns: int
    action: str  # 'enter' | 'exit'
    symbol: str
    sigma: float | None = None
    reference_price: float | None = None
    p_model: float | None = None
    edge: float | None = None


@dataclass(frozen=True, slots=True)
class SimTick:
    """One sim evaluate() diagnostic row (every scan tick, action may be 'hold')."""

    question_idx: int
    ts_ns: int
    action: str  # 'enter' | 'exit' | 'hold'
    sigma: float | None = None
    reference_price: float | None = None
    p_model: float | None = None
    edge: float | None = None


@dataclass(frozen=True, slots=True)
class FieldSkew:
    """Distribution of |sim − live| for one input field over matched pairs."""

    field: str
    n: int
    median_abs: float
    p90_abs: float
    max_abs: float
    median_rel: float | None  # None when the field has no meaningful relative form

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "n": self.n,
            "median_abs": self.median_abs,
            "p90_abs": self.p90_abs,
            "max_abs": self.max_abs,
            "median_rel": self.median_rel,
        }


@dataclass(frozen=True, slots=True)
class MatchedDecision:
    """A live decision joined to the nearest sim tick + whether the sim acted."""

    live: LiveDecision
    sim_tick: SimTick | None  # nearest sim tick in the same question within tol
    sim_action_match: bool  # sim emitted the SAME action in-window
    dt_ns: int | None  # |live.ts − sim_tick.ts|, None if no sim tick in tol


@dataclass
class DecisionReplayReport:
    n_live: int
    n_sim_actions: int
    n_live_matched: int
    field_skews: dict[str, FieldSkew]
    matched: list[MatchedDecision] = field(default_factory=list)
    unmatched_live: list[LiveDecision] = field(default_factory=list)
    phantom_sim: list[SimTick] = field(default_factory=list)

    @property
    def n_sim_phantom(self) -> int:
        return len(self.phantom_sim)

    @property
    def n_input_comparable(self) -> int:
        """Live decisions that had a sim evaluate-tick within tol (σ comparable).

        Low coverage here means the sim and live SCANNED at different instants
        (cadence divergence), not that the inputs disagree — read it alongside the
        per-field skews, which are computed only over these comparable points."""
        return sum(1 for m in self.matched if m.sim_tick is not None)

    def decision_match_rate(self) -> float:
        """Fraction of live decisions the sim reproduced (same action, in-window)."""
        return self.n_live_matched / self.n_live if self.n_live else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_live": self.n_live,
            "n_sim_actions": self.n_sim_actions,
            "n_live_matched": self.n_live_matched,
            "n_sim_phantom": self.n_sim_phantom,
            "n_input_comparable": self.n_input_comparable,
            "decision_match_rate": self.decision_match_rate(),
            "field_skews": {k: v.to_dict() for k, v in self.field_skews.items()},
            "unmatched_live": [
                {"question_idx": d.question_idx, "ts_ns": d.ts_ns, "action": d.action, "symbol": d.symbol}
                for d in self.unmatched_live
            ],
            "phantom_sim": [
                {"question_idx": t.question_idx, "ts_ns": t.ts_ns, "action": t.action} for t in self.phantom_sim
            ],
        }


def _percentile(sorted_vals: list[float], q: float) -> float:
    """Nearest-rank percentile on an already-sorted list (q in [0,1])."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = min(len(sorted_vals) - 1, max(0, round(q * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


def _nearest_tick(ticks: list[SimTick], ts_list: list[int], ts: int, tol: int) -> tuple[SimTick | None, int | None]:
    """Nearest sim tick to ``ts`` within ``tol`` (ticks sorted by ts)."""
    if not ticks:
        return None, None
    i = bisect.bisect_left(ts_list, ts)
    best: SimTick | None = None
    best_dt: int | None = None
    for j in (i - 1, i):
        if 0 <= j < len(ticks):
            dt = abs(ts_list[j] - ts)
            if dt <= tol and (best_dt is None or dt < best_dt):
                best, best_dt = ticks[j], dt
    return best, best_dt


def _action_in_window(action_ts: list[int], ts: int, tol: int) -> bool:
    """True if any action timestamp lies within [ts-tol, ts+tol]."""
    if not action_ts:
        return False
    i = bisect.bisect_left(action_ts, ts - tol)
    return i < len(action_ts) and action_ts[i] <= ts + tol


def replay(
    live: list[LiveDecision],
    sim: list[SimTick],
    *,
    ts_tol_ns: int,
) -> DecisionReplayReport:
    """Match live decisions to sim ticks and attribute per-field input skew.

    A live decision is *matched* iff the sim emitted the SAME action for the
    SAME question within ``ts_tol_ns``. Field skews are computed over every live
    decision that has any sim tick within tolerance (matched or not) so the input
    comparison covers unmatched decisions too. Sim actions with no same-action
    live counterpart in-window are phantoms.
    """
    # Index sim ticks per question for the input comparison. The sim emits a
    # diagnostics row on every book/reference tick, but only populates the
    # evaluate() model fields (σ / p_model / edge) on actual evaluate ticks — the
    # in-between book-tick rows carry NaN→None there. Snap the input comparison to
    # the nearest EVALUATED tick (σ present) so we compare the sim's real decision
    # state, not a bare book refresh. Falls back to all ticks when none is
    # evaluated (e.g. a question the sim only ever held with σ unset).
    eval_by_q: dict[int, list[SimTick]] = {}
    for t in sim:
        if t.sigma is not None:
            eval_by_q.setdefault(t.question_idx, []).append(t)
    all_by_q: dict[int, list[SimTick]] = {}
    for t in sim:
        all_by_q.setdefault(t.question_idx, []).append(t)
    for ticks in eval_by_q.values():
        ticks.sort(key=lambda t: t.ts_ns)
    for ticks in all_by_q.values():
        ticks.sort(key=lambda t: t.ts_ns)
    cmp_by_q = {q: (eval_by_q.get(q) or all_by_q.get(q, [])) for q in all_by_q}
    ts_by_q = {q: [t.ts_ns for t in ticks] for q, ticks in cmp_by_q.items()}
    # Per (question, action) sorted timestamps of sim action events.
    action_ts_by_qa: dict[tuple[int, str], list[int]] = {}
    sim_actions: list[SimTick] = []
    for t in sim:
        if t.action in ("enter", "exit"):
            sim_actions.append(t)
            action_ts_by_qa.setdefault((t.question_idx, t.action), []).append(t.ts_ns)
    for v in action_ts_by_qa.values():
        v.sort()

    diffs: dict[str, list[float]] = {f: [] for f in _ALL_FIELDS}
    rel_diffs: dict[str, list[float]] = {f: [] for f in _REL_FIELDS}
    matched: list[MatchedDecision] = []
    unmatched_live: list[LiveDecision] = []
    n_live_matched = 0

    for d in live:
        ticks = cmp_by_q.get(d.question_idx, [])
        ts_list = ts_by_q.get(d.question_idx, [])
        tick, dt = _nearest_tick(ticks, ts_list, d.ts_ns, ts_tol_ns)
        act_match = _action_in_window(action_ts_by_qa.get((d.question_idx, d.action), []), d.ts_ns, ts_tol_ns)
        matched.append(MatchedDecision(live=d, sim_tick=tick, sim_action_match=act_match, dt_ns=dt))
        if act_match:
            n_live_matched += 1
        else:
            unmatched_live.append(d)
        if tick is not None:
            for f in _ALL_FIELDS:
                a, b = getattr(d, f), getattr(tick, f)
                if a is None or b is None:
                    continue
                diffs[f].append(abs(a - b))
                if f in _REL_FIELDS and a != 0:
                    rel_diffs[f].append(abs(a - b) / abs(a))

    field_skews: dict[str, FieldSkew] = {}
    for f in _ALL_FIELDS:
        vals = sorted(diffs[f])
        rels = sorted(rel_diffs[f]) if f in _REL_FIELDS else []
        field_skews[f] = FieldSkew(
            field=f,
            n=len(vals),
            median_abs=_percentile(vals, 0.5),
            p90_abs=_percentile(vals, 0.9),
            max_abs=vals[-1] if vals else 0.0,
            median_rel=(_percentile(rels, 0.5) if rels else (None if f in _REL_FIELDS else None)),
        )

    # Phantom sim actions: a sim enter/exit with no same-action live decision in-window.
    live_ts_by_qa: dict[tuple[int, str], list[int]] = {}
    for d in live:
        live_ts_by_qa.setdefault((d.question_idx, d.action), []).append(d.ts_ns)
    for v in live_ts_by_qa.values():
        v.sort()
    phantom_sim: list[SimTick] = []
    for t in sim_actions:
        if not _action_in_window(live_ts_by_qa.get((t.question_idx, t.action), []), t.ts_ns, ts_tol_ns):
            phantom_sim.append(t)

    return DecisionReplayReport(
        n_live=len(live),
        n_sim_actions=len(sim_actions),
        n_live_matched=n_live_matched,
        field_skews=field_skews,
        matched=matched,
        unmatched_live=unmatched_live,
        phantom_sim=phantom_sim,
    )


def format_report(r: DecisionReplayReport) -> str:
    """Human-readable one-screen summary."""
    lines = [
        f"decisions: live={r.n_live}  sim_actions={r.n_sim_actions}",
        f"decision-match: {r.n_live_matched}/{r.n_live} "
        f"({r.decision_match_rate():.1%})   phantom sim actions: {r.n_sim_phantom}",
        f"input-comparable: {r.n_input_comparable}/{r.n_live} live decisions had a "
        f"sim eval-tick in tol (low ⇒ scan-cadence divergence, not input skew)",
        "per-field input skew at those comparable points:",
    ]
    for f in _ALL_FIELDS:
        sk = r.field_skews[f]
        rel = f"  rel_med={sk.median_rel:.2%}" if sk.median_rel is not None else ""
        lines.append(f"  {f:16} n={sk.n:4}  |Δ| med={sk.median_abs:.5g} p90={sk.p90_abs:.5g} max={sk.max_abs:.5g}{rel}")
    return "\n".join(lines)


__all__ = [
    "LiveDecision",
    "SimTick",
    "FieldSkew",
    "MatchedDecision",
    "DecisionReplayReport",
    "replay",
    "format_report",
]
