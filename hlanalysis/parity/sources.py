"""SHR-90 — normalize the live (trade journal + venue fills + settlement) and
sim (``RunResult``) sources into per-market records, then match them into the
:class:`~hlanalysis.parity.validation.DecisionPair`s the pure core attributes.

Reconciliation granularity is the **settled market** (``question_idx``): the
live realized PnL is venue truth (Σ venue ``closed_pnl`` + settlement payout),
the sim realized PnL is the run's ``realized_pnl_usd``, and the journal supplies
the two attribution flags — was a halt gate active, and did the sim's σ /
reference inputs diverge from the journal's recorded inputs. A market only one
side traded becomes a one-sided pair; both traded becomes a matched pair (its
residual is fill-level → execution).

The DB read is strictly **read-only** (a fresh sqlite connection over the engine
state DB) so this never touches the live engine or its DAL write-path — SHR-90 is
a pure consumer of SHR-83's journal and the venue mirror.
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .validation import DecisionPair, MarketParity, TradeLeg, reconcile_market

if TYPE_CHECKING:
    from hlanalysis.backtest.runner.result import RunResult

# The halt_json gate keys whose truth means "live could not have traded here".
_HALT_GATE_KEYS = (
    "restart_blocked",
    "daily_loss_halted",
    "reject_breaker_tripped",
    "stale_reference",
)

# Default relative tolerances for declaring an evaluate()-input divergence.
DEFAULT_SIGMA_REL_TOL = 0.05
DEFAULT_REF_REL_TOL = 0.01


@dataclass(frozen=True, slots=True)
class LiveMarket:
    """Live (venue-truth) reconciliation record for one settled market."""

    question_idx: int
    symbol: str
    realized_pnl: float  # Σ venue closed_pnl + settlement payout
    traded: bool  # ≥1 non-rejected order reached a fill
    halt_active: bool  # a halt/cap gate was active on any decision
    sigma: float | None  # representative journal σ input
    reference_price: float | None
    n_fills: int


@dataclass(frozen=True, slots=True)
class SimMarket:
    """Sim reconciliation record for one settled market (from a ``RunResult``)."""

    question_idx: int
    symbol: str
    realized_pnl: float
    traded: bool
    sigma: float | None
    reference_price: float | None
    n_fills: int


def _rel_diverged(a: float | None, b: float | None, rel_tol: float) -> bool:
    """True when both values are present and differ by more than ``rel_tol``
    (relative to the larger magnitude). Missing on either side → not detectable,
    so not a divergence."""
    if a is None or b is None:
        return False
    scale = max(abs(a), abs(b))
    if scale == 0.0:
        return False
    return abs(a - b) / scale > rel_tol


def _leg(realized_pnl: float, fill_price: float | None, fill_size: float) -> TradeLeg:
    return TradeLeg(realized_pnl=realized_pnl, fill_price=fill_price, fill_size=fill_size)


def _pair_for_market(
    q: int,
    lm: LiveMarket | None,
    sm: SimMarket | None,
    *,
    sigma_rel_tol: float,
    ref_rel_tol: float,
) -> tuple[str, DecisionPair]:
    """Build the (symbol, DecisionPair) for one market from its live/sim records."""
    live_leg = _leg(lm.realized_pnl, None, float(lm.n_fills)) if lm is not None and lm.traded else None
    sim_leg = _leg(sm.realized_pnl, None, float(sm.n_fills)) if sm is not None and sm.traded else None
    halt_active = bool(lm is not None and lm.halt_active)
    inputs_diverged = False
    if lm is not None and sm is not None:
        inputs_diverged = _rel_diverged(lm.sigma, sm.sigma, sigma_rel_tol) or _rel_diverged(
            lm.reference_price, sm.reference_price, ref_rel_tol
        )
    symbol = lm.symbol if lm is not None else sm.symbol  # type: ignore[union-attr]
    pair = DecisionPair(
        key=f"q{q}:{symbol}",
        sim=sim_leg,
        live=live_leg,
        live_halt_active=halt_active,
        inputs_diverged=inputs_diverged,
    )
    return symbol, pair


def _grouped(records, key: str) -> dict:
    """Index ``records`` by the chosen join attribute (``question_idx`` or
    ``symbol``). Last record wins on a collision (callers pass one per market)."""
    return {getattr(m, key): m for m in records}


def build_pairs(
    live: list[LiveMarket],
    sim: list[SimMarket],
    *,
    key: str = "question_idx",
    sigma_rel_tol: float = DEFAULT_SIGMA_REL_TOL,
    ref_rel_tol: float = DEFAULT_REF_REL_TOL,
) -> list[DecisionPair]:
    """Match live and sim per-market records into one :class:`DecisionPair` per
    market, deriving the halt / input-divergence flags. Markets present on only
    one side become one-sided pairs.

    ``key`` selects the join attribute: ``"question_idx"`` (default) or
    ``"symbol"``. Symbol keying is required when sim and live disagree on
    question_idx (buckets) or venue fills carry the unattributed -1 placeholder
    — the HL symbol is the stable shared identifier."""
    live_by = _grouped(live, key)
    sim_by = _grouped(sim, key)
    pairs: list[DecisionPair] = []
    for k in sorted(set(live_by) | set(sim_by), key=str):
        _, pair = _pair_for_market(
            k,
            live_by.get(k),
            sim_by.get(k),
            sigma_rel_tol=sigma_rel_tol,
            ref_rel_tol=ref_rel_tol,
        )
        pairs.append(pair)
    return pairs


def reconcile_markets(
    live: list[LiveMarket],
    sim: list[SimMarket],
    *,
    key: str = "question_idx",
    sigma_rel_tol: float = DEFAULT_SIGMA_REL_TOL,
    ref_rel_tol: float = DEFAULT_REF_REL_TOL,
) -> list[MarketParity]:
    """Convenience: match ``live``/``sim`` and reconcile each market (one
    :class:`DecisionPair` per market, so each :class:`MarketParity` wraps one).
    ``key`` selects the join attribute (see :func:`build_pairs`)."""
    live_by = _grouped(live, key)
    sim_by = _grouped(sim, key)
    out: list[MarketParity] = []
    for k in sorted(set(live_by) | set(sim_by), key=str):
        lm, sm = live_by.get(k), sim_by.get(k)
        symbol, pair = _pair_for_market(
            k,
            lm,
            sm,
            sigma_rel_tol=sigma_rel_tol,
            ref_rel_tol=ref_rel_tol,
        )
        # question_idx for reporting comes from a present record (the group key
        # may itself be a symbol string when key="symbol").
        qidx = lm.question_idx if lm is not None else sm.question_idx  # type: ignore[union-attr]
        out.append(reconcile_market(question_idx=qidx, symbol=symbol, pairs=[pair]))
    return out


def sim_market_from_run(
    *,
    question_idx: int,
    symbol: str,
    result: RunResult,
    sigma: float | None = None,
    reference_price: float | None = None,
) -> SimMarket:
    """Adapt one per-question :class:`RunResult` into a :class:`SimMarket`.

    ``traded`` is ``True`` iff the run produced ≥1 fill; ``realized_pnl`` falls
    back to ``0.0`` when the run booked nothing (``realized_pnl_usd is None``).
    ``sigma`` / ``reference_price`` are the sim's evaluate() inputs (optional —
    when absent, input-divergence is simply not detectable for this market)."""
    n_fills = len(result.fills)
    return SimMarket(
        question_idx=question_idx,
        symbol=symbol,
        realized_pnl=result.realized_pnl_usd if result.realized_pnl_usd is not None else 0.0,
        traded=n_fills > 0,
        sigma=sigma,
        reference_price=reference_price,
        n_fills=n_fills,
    )


def _halt_active_from_json(halt_json: str | None) -> bool:
    if not halt_json:
        return False
    try:
        d = json.loads(halt_json)
    except (ValueError, TypeError):
        return False
    return any(bool(d.get(k)) for k in _HALT_GATE_KEYS)


def load_live_markets_from_db(
    db_path: Path | str,
    *,
    key: str = "question_idx",
) -> list[LiveMarket]:
    """Build per-market :class:`LiveMarket` records from an engine state DB,
    read-only. Joins the trade journal (decisions + halt + σ/reference inputs),
    the venue Fill mirror (Σ ``closed_pnl``, fill count, traded flag), and the
    settlement table (payout PnL).

    ``key`` selects the rollup grain: ``"question_idx"`` (default, legacy) or
    ``"symbol"``. Use ``"symbol"`` to reconcile against the backtest, because
    venue fills carry the unattributed ``question_idx=-1`` placeholder (which
    would collapse every such market into one bogus bucket) while the fill's
    ``symbol`` is always the real HL market id."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        journal = con.execute(
            "SELECT cloid, question_idx, symbol, decision_ts_ns, sigma, "
            "reference_price, halt_json, reject_reason, fill_ts_ns "
            "FROM trade_journal ORDER BY question_idx, decision_ts_ns"
        ).fetchall()
        fills = con.execute("SELECT question_idx, symbol, closed_pnl FROM fill").fetchall()
        settlements = con.execute("SELECT question_idx, symbol, realized_pnl FROM settlement").fetchall()
    finally:
        con.close()

    def gk(question_idx, symbol):
        """The rollup key for a row, per ``key`` mode."""
        return symbol if key == "symbol" else question_idx

    closed_by: dict = defaultdict(float)
    nfills_by: dict = defaultdict(int)
    qidx_by: dict = {}
    symbol_by: dict = {}
    for f in fills:
        k = gk(f["question_idx"], f["symbol"])
        closed_by[k] += f["closed_pnl"] or 0.0
        nfills_by[k] += 1
        qidx_by.setdefault(k, f["question_idx"])
        symbol_by.setdefault(k, f["symbol"])

    settle_by: dict = defaultdict(float)
    for s in settlements:
        settle_by[gk(s["question_idx"], s["symbol"])] += s["realized_pnl"] or 0.0

    # Per-key journal rollup: representative inputs (first decision), halt active
    # on ANY decision, traded iff any decision reached a fill.
    halt_by: dict = defaultdict(bool)
    sigma_by: dict = {}
    ref_by: dict = {}
    traded_by: dict = defaultdict(bool)
    ks_in_journal: list = []
    for r in journal:
        k = gk(r["question_idx"], r["symbol"])
        if k not in sigma_by:
            ks_in_journal.append(k)
            sigma_by[k] = r["sigma"]
            ref_by[k] = r["reference_price"]
            qidx_by.setdefault(k, r["question_idx"])
            symbol_by.setdefault(k, r["symbol"] or "")
        halt_by[k] = halt_by[k] or _halt_active_from_json(r["halt_json"])
        traded_by[k] = traded_by[k] or (r["fill_ts_ns"] is not None)

    all_keys = set(ks_in_journal) | set(closed_by) | set(settle_by)
    out: list[LiveMarket] = []
    for k in sorted(all_keys, key=str):
        realized = closed_by.get(k, 0.0) + settle_by.get(k, 0.0)
        n_fills = nfills_by.get(k, 0)
        traded = traded_by.get(k, False) or n_fills > 0
        out.append(
            LiveMarket(
                question_idx=qidx_by.get(k, -1),
                symbol=symbol_by.get(k, ""),
                realized_pnl=realized,
                traded=traded,
                halt_active=halt_by.get(k, False),
                sigma=sigma_by.get(k),
                reference_price=ref_by.get(k),
                n_fills=n_fills,
            )
        )
    return out


__all__ = [
    "DEFAULT_SIGMA_REL_TOL",
    "DEFAULT_REF_REL_TOL",
    "LiveMarket",
    "SimMarket",
    "build_pairs",
    "reconcile_markets",
    "sim_market_from_run",
    "load_live_markets_from_db",
]
