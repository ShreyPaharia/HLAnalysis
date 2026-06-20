"""SHR-150: cross-question fidelity dashboard.

Reconciliation is per-question; fidelity — does the sim faithfully reproduce live
across the corpus? — is an aggregate property a single lucky question can hide.
This module aggregates many :class:`ReconcileResult` into the distributions that
characterise sim↔live fidelity and renders them as an HTML card reusing the
research :class:`~hlanalysis.research.report.Report` builder.

Reported aggregates (overall and broken out per class / track):

* decision-agreement rate — mean Layer-1 ``match_rate`` (and verdict pass-rate);
* slippage distribution — live − sim VWAP over matched episodes. A predominantly
  *adverse* slippage signals a model/routing problem (live consistently fills
  worse than the sim assumes);
* per-pair latency distribution — matched live↔sim episode start-time gap (s);
* hit-rate delta — live vs sim fraction of profitable questions;
* PnL-difference distribution — per-question ``pnl_diff``.

Usage::

    from hlanalysis.research.reconcile.fidelity import aggregate_fidelity, render_fidelity_html
    report = aggregate_fidelity(results, labels)
    render_fidelity_html(report, "docs/research/reconcile-fidelity.html")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from hlanalysis.research.reconcile.reconcile import ReconcileResult
from hlanalysis.research.report import Report


@dataclass
class QuestionLabel:
    """Class/track label for one reconciled question (for per-class breakout).

    Parameters
    ----------
    klass:
        Market class, e.g. ``"binary"`` or ``"bucket"``.
    track:
        Market track, ``"HL"`` or ``"PM"``.
    """

    klass: str = "unknown"
    track: str = "HL"


@dataclass
class Distribution:
    """Summary statistics for one distribution of values.

    All fields are ``0.0`` / ``n=0`` for an empty input.
    """

    n: int
    mean: float
    median: float
    std: float
    p10: float
    p90: float
    min: float
    max: float


@dataclass
class FidelityGroup:
    """Aggregated fidelity metrics for one group of questions.

    Parameters
    ----------
    label:
        Group name (``"all"``, a class, or a track).
    n_questions:
        Number of reconciled questions in the group.
    decision_agreement_rate:
        Mean Layer-1 decision ``match_rate``.
    verdict_pass_rate:
        Fraction of questions whose overall verdict was PASS.
    hit_rate_live, hit_rate_sim:
        Fraction of questions with strictly-positive live / sim realized PnL.
    hit_rate_delta:
        ``hit_rate_live − hit_rate_sim``.
    adverse_slippage_frac:
        Fraction of matched episodes where live filled *worse* than sim (a BUY
        above / a SELL below the sim VWAP).
    slippage:
        Distribution of live − sim VWAP over matched episodes.
    latency_seconds:
        Distribution of matched live↔sim start-time gaps (seconds).
    pnl_diff:
        Distribution of per-question ``pnl_diff`` (live − sim realized).
    """

    label: str
    n_questions: int
    decision_agreement_rate: float
    verdict_pass_rate: float
    hit_rate_live: float
    hit_rate_sim: float
    hit_rate_delta: float
    adverse_slippage_frac: float
    slippage: Distribution
    latency_seconds: Distribution
    pnl_diff: Distribution


@dataclass
class FidelityReport:
    """Overall fidelity plus per-class and per-track breakouts."""

    overall: FidelityGroup
    by_class: dict[str, FidelityGroup] = field(default_factory=dict)
    by_track: dict[str, FidelityGroup] = field(default_factory=dict)


def summarize(values: list[float]) -> Distribution:
    """Summarise a list of floats into a :class:`Distribution`.

    Parameters
    ----------
    values:
        The values; an empty list yields an all-zero, ``n=0`` distribution.
    """
    if not values:
        return Distribution(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    arr = np.asarray(values, dtype=float)
    return Distribution(
        n=int(arr.size),
        mean=float(arr.mean()),
        median=float(np.median(arr)),
        std=float(arr.std(ddof=0)),
        p10=float(np.percentile(arr, 10)),
        p90=float(np.percentile(arr, 90)),
        min=float(arr.min()),
        max=float(arr.max()),
    )


def _matched_episodes(result: ReconcileResult) -> pd.DataFrame:
    """Return the matched rows of a result's episode table (may be empty)."""
    et = result.layer2.episode_table
    if et is None or et.empty or "match_status" not in et.columns:
        return pd.DataFrame()
    return et[et["match_status"] == "matched"]


def _is_adverse(side: str, slip: float) -> bool:
    """Whether ``slip`` (= live − sim VWAP) is adverse for ``side``.

    A BUY filled *above* the sim VWAP, or a SELL filled *below* it, is worse for
    live — the signature of a model/routing slippage problem.
    """
    s = str(side).strip().upper()
    if s == "BUY":
        return slip > 0
    if s == "SELL":
        return slip < 0
    return False


def _group(label: str, results: list[ReconcileResult]) -> FidelityGroup:
    """Aggregate one group of results into a :class:`FidelityGroup`."""
    n = len(results)
    match_rates = [r.layer1.match_rate for r in results]
    pass_count = sum(1 for r in results if r.verdict == "PASS")
    live_wins = sum(1 for r in results if r.layer3.live_realized > 0)
    sim_wins = sum(1 for r in results if r.layer3.sim_realized > 0)
    pnl_diffs = [r.layer3.pnl_diff for r in results]

    slippage: list[float] = []
    latency: list[float] = []
    adverse = 0
    n_episodes = 0
    for r in results:
        em = _matched_episodes(r)
        for _, row in em.iterrows():
            lv = row.get("live_vwap")
            sv = row.get("sim_vwap")
            if lv is not None and sv is not None and pd.notna(lv) and pd.notna(sv):
                slip = float(lv) - float(sv)
                slippage.append(slip)
                n_episodes += 1
                if _is_adverse(row.get("live_side", ""), slip):
                    adverse += 1
            lat = row.get("latency_ns")
            if lat is not None and pd.notna(lat):
                latency.append(float(lat) / 1e9)

    hit_live = live_wins / n if n else 0.0
    hit_sim = sim_wins / n if n else 0.0
    return FidelityGroup(
        label=label,
        n_questions=n,
        decision_agreement_rate=(sum(match_rates) / n if n else 0.0),
        verdict_pass_rate=(pass_count / n if n else 0.0),
        hit_rate_live=hit_live,
        hit_rate_sim=hit_sim,
        hit_rate_delta=hit_live - hit_sim,
        adverse_slippage_frac=(adverse / n_episodes if n_episodes else 0.0),
        slippage=summarize(slippage),
        latency_seconds=summarize(latency),
        pnl_diff=summarize(pnl_diffs),
    )


def aggregate_fidelity(
    results: list[ReconcileResult],
    labels: list[QuestionLabel] | None = None,
) -> FidelityReport:
    """Aggregate per-question reconciliations into a cross-question fidelity report.

    Parameters
    ----------
    results:
        Per-question :class:`ReconcileResult` objects.
    labels:
        Optional per-result class/track labels (aligned by index). When omitted
        every question is labelled ``unknown`` / ``HL``.

    Returns
    -------
    A :class:`FidelityReport` with overall, per-class and per-track groups.
    """
    if labels is None:
        labels = [QuestionLabel() for _ in results]
    if len(labels) != len(results):
        raise ValueError(f"labels ({len(labels)}) must align with results ({len(results)})")

    by_class: dict[str, list[ReconcileResult]] = {}
    by_track: dict[str, list[ReconcileResult]] = {}
    for r, lab in zip(results, labels):
        by_class.setdefault(lab.klass, []).append(r)
        by_track.setdefault(lab.track, []).append(r)

    return FidelityReport(
        overall=_group("all", results),
        by_class={k: _group(k, rs) for k, rs in by_class.items()},
        by_track={k: _group(k, rs) for k, rs in by_track.items()},
    )


# ── HTML rendering ───────────────────────────────────────────────────────────


def _dist_row(name: str, d: Distribution) -> str:
    return (
        f"<tr><td>{name}</td><td>{d.n}</td><td>{d.mean:+.4f}</td><td>{d.median:+.4f}</td>"
        f"<td>{d.std:.4f}</td><td>{d.p10:+.4f}</td><td>{d.p90:+.4f}</td>"
        f"<td>{d.min:+.4f}</td><td>{d.max:+.4f}</td></tr>"
    )


def _group_html(g: FidelityGroup) -> str:
    """Render one group as an HTML body fragment."""
    head = (
        f"<p><b>{g.n_questions}</b> questions · decision-agreement "
        f"<b>{g.decision_agreement_rate:.1%}</b> · verdict-pass <b>{g.verdict_pass_rate:.1%}</b></p>"
        f"<p>hit-rate live <b>{g.hit_rate_live:.1%}</b> vs sim <b>{g.hit_rate_sim:.1%}</b> "
        f"(Δ <b>{g.hit_rate_delta:+.1%}</b>) · adverse-slippage fraction "
        f"<b>{g.adverse_slippage_frac:.1%}</b></p>"
    )
    table = (
        "<table><tr><th>Distribution</th><th>n</th><th>mean</th><th>median</th><th>std</th>"
        "<th>p10</th><th>p90</th><th>min</th><th>max</th></tr>"
        + _dist_row("Slippage (live − sim VWAP)", g.slippage)
        + _dist_row("Latency (s)", g.latency_seconds)
        + _dist_row("PnL diff ($)", g.pnl_diff)
        + "</table>"
    )
    return head + table


def render_fidelity_html(
    report: FidelityReport,
    path: str | Path,
    title: str = "Sim↔Live Reconciliation Fidelity",
) -> None:
    """Render a :class:`FidelityReport` to a standalone HTML dashboard.

    Parameters
    ----------
    report:
        The aggregated fidelity report.
    path:
        Output HTML file path (parent dirs are created).
    title:
        Page title.
    """
    rep = Report(title=title)
    rep.add_card(
        "Overall fidelity",
        _group_html(report.overall),
        notes=(
            "Slippage = live − sim VWAP over matched episodes; a predominantly adverse "
            "fraction (>50%) points to a model/routing problem, not noise."
        ),
    )
    if report.by_class:
        body = "".join(f"<h3>{k}</h3>{_group_html(g)}" for k, g in sorted(report.by_class.items()))
        rep.add_card("By class (binary / bucket)", body)
    if report.by_track:
        body = "".join(f"<h3>{k}</h3>{_group_html(g)}" for k, g in sorted(report.by_track.items()))
        rep.add_card("By track (HL / PM)", body)
    rep.render(path)
