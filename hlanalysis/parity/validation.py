"""Standing sim-vs-live validation pipeline — pure core (SHR-90 / Spec 3 T10).

Turns *"is the sim trustworthy?"* into a monitored number. For each settled
market it compares the sim's realized PnL/fills against live (venue user_fills +
settlement, joined with the SHR-83 trade journal) and **attributes** the per-market
PnL residual ``residual = sim_pnl − live_pnl`` to one of three causes:

* **input-skew** — the sim's :class:`MarketState` inputs (σ / returns / book /
  reference) differed from the journal's recorded evaluate() inputs, so the sim
  made a *different decision* than live. The divergence is at the decision layer.
* **execution** — sim and live made the *same decision*, but the realized fill
  (price / size / timing — real venue vs simulated) differed. The divergence is
  at the fill layer.
* **unmodeled-halt** — a live halt/cap gate (stale_data / daily_loss / reject_breaker
  / restart_blocked) was active at decision time, so live could not trade where the
  sim did. The divergence is a gate the sim didn't replay.

The buckets PARTITION the residual exactly (``input_skew + execution +
unmodeled_halt == residual`` for every market), so a per-bucket total is an
honest decomposition of total drift — the residual is never silently dropped.

This module is the PURE core: it takes already-matched :class:`DecisionPair`s
and emits per-market + aggregate reports (machine-readable :meth:`to_dict` +
:func:`format_report` human summary). The IO that loads pairs from the live
journal / venue fills / sim ``RunResult`` lives in the ``tools`` CLI shell so
the attribution logic stays unit-testable on synthetic inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Residual attribution bucket keys (also the JSON field names).
BUCKET_INPUT_SKEW = "input_skew"
BUCKET_EXECUTION = "execution"
BUCKET_UNMODELED_HALT = "unmodeled_halt"

# Default gate threshold: the attributed |residual| may be at most this fraction
# of the live PnL magnitude for the sim to be called "trustworthy". Conservative
# starting point; tune as the fidelity program closes the execution residual.
DEFAULT_RESIDUAL_RATIO_THRESHOLD = 0.20


@dataclass(frozen=True, slots=True)
class TradeLeg:
    """One side's (sim or live) realized outcome for a single matched decision.

    ``realized_pnl`` is the trade's realized PnL on this market (settlement
    payout included). ``fill_price`` / ``fill_size`` describe the executed fill
    (``fill_price`` is ``None`` and ``fill_size`` ``0.0`` for a decision that
    produced no fill — e.g. a rejected order recorded by the journal)."""

    realized_pnl: float
    fill_price: float | None = None
    fill_size: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "realized_pnl": self.realized_pnl,
            "fill_price": self.fill_price,
            "fill_size": self.fill_size,
        }


@dataclass(frozen=True, slots=True)
class DecisionPair:
    """A sim decision matched to its live counterpart for one market.

    Exactly one side is ``None`` for a *one-sided* decision (only sim or only
    live traded); both present is a *matched* decision. The two context flags
    are read from the live journal:

    * ``live_halt_active`` — a halt/cap gate was live at decision time.
    * ``inputs_diverged`` — the sim's evaluate() inputs differed materially from
      the journal's recorded inputs for this decision.

    The matching itself (join key, timestamp tolerance) is the caller's job; the
    ``key`` is opaque to the core and only used for reporting/debugging."""

    key: str
    sim: TradeLeg | None = None
    live: TradeLeg | None = None
    live_halt_active: bool = False
    inputs_diverged: bool = False

    @property
    def delta(self) -> float:
        """This decision's signed PnL residual: ``sim_pnl − live_pnl``."""
        sim_pnl = self.sim.realized_pnl if self.sim is not None else 0.0
        live_pnl = self.live.realized_pnl if self.live is not None else 0.0
        return sim_pnl - live_pnl

    @property
    def is_matched(self) -> bool:
        return self.sim is not None and self.live is not None


@dataclass(frozen=True, slots=True)
class ResidualAttribution:
    """The per-market PnL residual decomposed into its three causes. The three
    fields sum to :attr:`total` exactly (the partition invariant)."""

    input_skew: float = 0.0
    execution: float = 0.0
    unmodeled_halt: float = 0.0

    @property
    def total(self) -> float:
        return self.input_skew + self.execution + self.unmodeled_halt

    @property
    def dominant_bucket(self) -> str | None:
        """The bucket with the largest absolute contribution, or ``None`` when
        there is no residual to attribute."""
        buckets = {
            BUCKET_INPUT_SKEW: self.input_skew,
            BUCKET_EXECUTION: self.execution,
            BUCKET_UNMODELED_HALT: self.unmodeled_halt,
        }
        key, val = max(buckets.items(), key=lambda kv: abs(kv[1]))
        return key if abs(val) > 0.0 else None

    def __add__(self, other: ResidualAttribution) -> ResidualAttribution:
        return ResidualAttribution(
            input_skew=self.input_skew + other.input_skew,
            execution=self.execution + other.execution,
            unmodeled_halt=self.unmodeled_halt + other.unmodeled_halt,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            BUCKET_INPUT_SKEW: self.input_skew,
            BUCKET_EXECUTION: self.execution,
            BUCKET_UNMODELED_HALT: self.unmodeled_halt,
            "total": self.total,
            "dominant_bucket": self.dominant_bucket,
        }


def attribute_residual(pairs: list[DecisionPair]) -> ResidualAttribution:
    """Partition the PnL residual of ``pairs`` into the three buckets.

    Per decision, with ``delta = sim_pnl − live_pnl``:

    * **matched** (both traded) → ``execution`` (same decision, fill differs).
    * **one-sided + halt active** → ``unmodeled_halt`` (the gate is the proximate
      cause; it takes precedence over input divergence — live could not have
      traded regardless of its inputs).
    * **one-sided + inputs diverged** → ``input_skew`` (different MarketState →
      different decision).
    * **one-sided, no halt, same inputs** → ``execution`` (a reject / marketability
      difference: same intent, different fill outcome).

    Because every decision lands in exactly one bucket and contributes its full
    ``delta``, the returned buckets sum to the total residual exactly."""
    input_skew = 0.0
    execution = 0.0
    unmodeled_halt = 0.0
    for p in pairs:
        d = p.delta
        if p.is_matched:
            execution += d
        elif p.live_halt_active:
            unmodeled_halt += d
        elif p.inputs_diverged:
            input_skew += d
        else:
            execution += d
    return ResidualAttribution(
        input_skew=input_skew, execution=execution, unmodeled_halt=unmodeled_halt
    )


@dataclass(frozen=True, slots=True)
class MarketParity:
    """The reconciliation of one settled market: sim vs live PnL, the residual,
    and its attribution + fill counts."""

    question_idx: int
    symbol: str
    sim_pnl: float
    live_pnl: float
    residual: float
    attribution: ResidualAttribution
    n_matched: int
    n_sim_only: int
    n_live_only: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_idx": self.question_idx,
            "symbol": self.symbol,
            "sim_pnl": self.sim_pnl,
            "live_pnl": self.live_pnl,
            "residual": self.residual,
            "attribution": self.attribution.to_dict(),
            "n_matched": self.n_matched,
            "n_sim_only": self.n_sim_only,
            "n_live_only": self.n_live_only,
        }


def reconcile_market(
    *, question_idx: int, symbol: str, pairs: list[DecisionPair]
) -> MarketParity:
    """Reconcile one settled market from its matched decision pairs."""
    sim_pnl = sum(p.sim.realized_pnl for p in pairs if p.sim is not None)
    live_pnl = sum(p.live.realized_pnl for p in pairs if p.live is not None)
    n_matched = sum(1 for p in pairs if p.is_matched)
    n_sim_only = sum(1 for p in pairs if p.sim is not None and p.live is None)
    n_live_only = sum(1 for p in pairs if p.live is not None and p.sim is None)
    return MarketParity(
        question_idx=question_idx,
        symbol=symbol,
        sim_pnl=sim_pnl,
        live_pnl=live_pnl,
        residual=sim_pnl - live_pnl,
        attribution=attribute_residual(pairs),
        n_matched=n_matched,
        n_sim_only=n_sim_only,
        n_live_only=n_live_only,
    )


@dataclass(frozen=True, slots=True)
class FidelityReport:
    """The aggregate sim-fidelity report over a set of settled markets — the
    standing regression gate. ``residual_ratio`` is the monitored number;
    ``trustworthy`` is the gate verdict."""

    run_ts_ns: int
    markets: list[MarketParity]
    total_sim_pnl: float
    total_live_pnl: float
    total_residual: float
    abs_residual: float
    attribution: ResidualAttribution
    residual_ratio: float
    residual_ratio_threshold: float
    trustworthy: bool

    @property
    def n_markets(self) -> int:
        return len(self.markets)

    @property
    def dominant_bucket(self) -> str | None:
        return self.attribution.dominant_bucket

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": {
                "run_ts_ns": self.run_ts_ns,
                "n_markets": self.n_markets,
                "total_sim_pnl": self.total_sim_pnl,
                "total_live_pnl": self.total_live_pnl,
                "total_residual": self.total_residual,
                "abs_residual": self.abs_residual,
                "residual_ratio": self.residual_ratio,
                "residual_ratio_threshold": self.residual_ratio_threshold,
                "trustworthy": self.trustworthy,
                "dominant_bucket": self.dominant_bucket,
                "attribution": self.attribution.to_dict(),
            },
            "markets": [m.to_dict() for m in self.markets],
        }

    def timeseries_row(self) -> dict[str, Any]:
        """The flat one-line record appended to the monitoring time series — the
        per-run datapoint for tracking fidelity drift over time."""
        return {
            "run_ts_ns": self.run_ts_ns,
            "n_markets": self.n_markets,
            "total_residual": self.total_residual,
            "abs_residual": self.abs_residual,
            "residual_ratio": self.residual_ratio,
            BUCKET_INPUT_SKEW: self.attribution.input_skew,
            BUCKET_EXECUTION: self.attribution.execution,
            BUCKET_UNMODELED_HALT: self.attribution.unmodeled_halt,
            "dominant_bucket": self.dominant_bucket,
            "trustworthy": self.trustworthy,
        }


def build_report(
    markets: list[MarketParity],
    *,
    run_ts_ns: int,
    residual_ratio_threshold: float = DEFAULT_RESIDUAL_RATIO_THRESHOLD,
) -> FidelityReport:
    """Aggregate per-market parities into the standing fidelity report.

    The monitored number is ``residual_ratio = Σ|residual| / Σ|live_pnl|`` — the
    attributed sim-vs-live drift as a fraction of the live PnL magnitude (0.0 ==
    perfect parity). The gate passes (``trustworthy``) when it is at or below
    ``residual_ratio_threshold``. A market set with no live PnL is vacuously
    trustworthy (ratio 0.0) — there is no live truth to disagree with."""
    total_sim_pnl = sum(m.sim_pnl for m in markets)
    total_live_pnl = sum(m.live_pnl for m in markets)
    total_residual = sum(m.residual for m in markets)
    abs_residual = sum(abs(m.residual) for m in markets)
    attribution = ResidualAttribution()
    for m in markets:
        attribution = attribution + m.attribution
    gross_live = sum(abs(m.live_pnl) for m in markets)
    residual_ratio = abs_residual / gross_live if gross_live > 0.0 else 0.0
    return FidelityReport(
        run_ts_ns=run_ts_ns,
        markets=list(markets),
        total_sim_pnl=total_sim_pnl,
        total_live_pnl=total_live_pnl,
        total_residual=total_residual,
        abs_residual=abs_residual,
        attribution=attribution,
        residual_ratio=residual_ratio,
        residual_ratio_threshold=residual_ratio_threshold,
        trustworthy=residual_ratio <= residual_ratio_threshold,
    )


def format_report(report: FidelityReport) -> str:
    """Render a human-readable summary of a :class:`FidelityReport`."""
    a = report.attribution
    verdict = "TRUSTWORTHY" if report.trustworthy else "DRIFT"
    lines = [
        f"=== Sim-fidelity report  ({verdict}) ===",
        f"markets:          {report.n_markets}",
        f"sim PnL:          {report.total_sim_pnl:+.2f}",
        f"live PnL:         {report.total_live_pnl:+.2f}",
        f"residual:         {report.total_residual:+.2f}  "
        f"(|Σ|={report.abs_residual:.2f})",
        f"residual ratio:   {report.residual_ratio:.4f}  "
        f"(threshold {report.residual_ratio_threshold:.4f})",
        "attribution:",
        f"  input-skew:     {a.input_skew:+.2f}",
        f"  execution:      {a.execution:+.2f}",
        f"  unmodeled-halt: {a.unmodeled_halt:+.2f}",
        f"  dominant:       {report.dominant_bucket or '—'}",
    ]
    if report.markets:
        lines.append("per-market:")
        for m in report.markets:
            lines.append(
                f"  q{m.question_idx} {m.symbol}: "
                f"sim {m.sim_pnl:+.2f} / live {m.live_pnl:+.2f} "
                f"→ resid {m.residual:+.2f} "
                f"[skew {m.attribution.input_skew:+.2f} "
                f"exec {m.attribution.execution:+.2f} "
                f"halt {m.attribution.unmodeled_halt:+.2f}]"
            )
    return "\n".join(lines)


__all__ = [
    "BUCKET_INPUT_SKEW",
    "BUCKET_EXECUTION",
    "BUCKET_UNMODELED_HALT",
    "DEFAULT_RESIDUAL_RATIO_THRESHOLD",
    "TradeLeg",
    "DecisionPair",
    "ResidualAttribution",
    "attribute_residual",
    "MarketParity",
    "reconcile_market",
    "FidelityReport",
    "build_report",
    "format_report",
]
