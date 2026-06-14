"""Slot construction helpers extracted from ``EngineRuntime``.

``build_slot`` and ``register_reference_cadences`` are the two slot-level
initialisation routines that previously lived as methods on ``EngineRuntime``.
They are pure construction/registration logic with no async I/O and no
dependency on the runtime control-flow — they only need the subset of engine
state listed in their explicit parameters.

``EngineRuntime._build_slot`` and ``EngineRuntime._register_reference_cadences``
remain as thin delegating methods so callers (tests, replay.py) that already
call ``rt._build_slot(...)`` / ``rt._register_reference_cadences(...)`` continue
to work without modification.

Do NOT import ``runtime.py`` from this module — that would create a cycle.
Use ``TYPE_CHECKING`` for annotation-only imports.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger  # noqa: F401  (available for future logging needs)

from ..marketdata.position_math import DUST_QTY_ABS_TOL
from .config import (
    AccountConfig,
    DeployConfig,
    StrategyConfig,
)
from .config_builders import _build_strategy_for_slot, build_exec_client
from .event_bus import EventBus
from .exec_client import ExecutionClient
from .market_state import MarketState
from .risk import RiskGate
from .router import Router
from .scanner import Scanner
from .state import CachedStateDAL
from .trade_journal import TradeJournal

if TYPE_CHECKING:
    from collections.abc import Callable


def build_slot(
    s_cfg: StrategyConfig,
    *,
    deploy_cfg: DeployConfig,
    exec_client_factory: Callable[[str, AccountConfig, bool], ExecutionClient] | None,
    bus: EventBus,
    market_state: MarketState,
) -> object:
    """Construct one ``AccountSlot`` for the given strategy config.

    Extracted from ``EngineRuntime._build_slot``.  The return type is declared
    ``object`` to avoid importing ``AccountSlot`` / ``PmSlotState`` from
    ``runtime.py`` (which would create a cycle); callers that need the
    concrete type already have it via their own ``runtime`` import.

    Parameters
    ----------
    s_cfg:
        The strategy config for this slot.
    deploy_cfg:
        The engine-wide deploy config (account registry, path templates, …).
    exec_client_factory:
        Optional injected factory ``(alias, account_cfg, paper_mode)`` →
        ``ExecutionClient``.  If ``None``, ``build_exec_client`` is used.
    bus:
        The shared engine event bus (passed to Router).
    market_state:
        The shared engine MarketState (passed to Scanner).
    """
    # Import here to avoid a module-level cycle (AccountSlot / PmSlotState
    # live in runtime.py, which imports this module).
    from .runtime import AccountSlot, PmSlotState

    alias = s_cfg.account_alias
    if alias not in deploy_cfg.accounts:
        raise ValueError(
            f"strategy '{s_cfg.name}' references account_alias={alias!r} but "
            f"deploy.accounts has only {list(deploy_cfg.accounts)}",
        )
    acct = deploy_cfg.accounts[alias]
    state_db_path = Path(deploy_cfg.state_db_path_for(alias))
    kill_switch_path = Path(deploy_cfg.kill_switch_path_for(alias))
    cloid_prefix = f"hla-{alias}-"

    # Cached DAL: positions/orders are read every loop wake (esp. under the
    # event-driven scan, P1); serve those from memory, write through to the
    # DB. run_migrations FIRST so the lazy cache load sees existing tables.
    dal = CachedStateDAL(state_db_path)
    dal.run_migrations()

    if exec_client_factory is not None:
        exec_client = exec_client_factory(alias, acct, s_cfg.paper_mode)
    else:
        exec_client = build_exec_client(alias, acct, s_cfg.paper_mode)

    risk = RiskGate(s_cfg)
    # Durable trade journal (SHR-83): persists into the slot's state.db, shared
    # by the Router (decision/send/reject/fill) and Reconciler (late fills).
    journal = TradeJournal(
        dal,
        suppress_veto_reasons=frozenset(deploy_cfg.journal_suppress_veto_reasons),
    )
    # PM market sells floor the share amount to 2dp, stranding sub-0.01 dust
    # that wedges the position open (2026-06-06 v31_pm). PM slots treat a
    # reduce landing within that dust of flat as a full close (and suppress
    # un-sellable dust sells). HL fills the exact size, so it stays ~exact.
    reduce_close_atol = DUST_QTY_ABS_TOL if acct.venue == "polymarket" else 1e-9
    router = Router(
        dal=dal,
        gate=risk,
        bus=bus,
        exec_client=exec_client,
        strategy_cfg=s_cfg,
        strategy_id=s_cfg.name,
        cloid_prefix=cloid_prefix,
        reduce_close_atol=reduce_close_atol,
        journal=journal,
    )
    strategy = _build_strategy_for_slot(s_cfg)
    # Gate-decision log sibling of state.db. Operators tail this during
    # forward-testing to see which gates are firing without combing
    # through journal heartbeats. State-change-debounced, so file size
    # stays small (one line per question per transition).
    gate_log_path = state_db_path.parent / "gate_decisions.jsonl"
    scanner = Scanner(
        strategy=strategy,
        cfg=s_cfg,
        market_state=market_state,
        dal=dal,
        kill_switch_path=kill_switch_path,
        reference_symbol=s_cfg.reference_symbol,
        last_reconcile_ns=0,
        # Daily-loss cap reads from HL (venue truth) rather than the local
        # DB. The DB's realized_pnl is structurally near-zero — fills aren't
        # persisted on the happy path and closed positions are deleted —
        # so without this the cap would never fire. outcome_only=True so the
        # operator's NON-strategy manual perp/spot trades on the same account
        # can't false-halt (or mask losses in) the strategy (no-op for PM).
        pnl_provider=lambda ts: exec_client.realized_pnl_since(
            ts,
            outcome_only=True,
        ),
        gate_log_path=gate_log_path,
    )
    return AccountSlot(
        cfg=s_cfg,
        account_cfg=acct,
        venue=acct.venue,
        state_db_path=state_db_path,
        kill_switch_path=kill_switch_path,
        cloid_prefix=cloid_prefix,
        dal=dal,
        exec_client=exec_client,
        risk=risk,
        router=router,
        strategy=strategy,
        scanner=scanner,
        journal=journal,
        pm=PmSlotState() if acct.venue == "polymarket" else None,
    )


def register_reference_cadences(
    slots: list,
    *,
    market_state: MarketState,
) -> None:
    """Register each slot's default reference cadence AND any per-class
    theta-override cadences on the shared MarketState.

    Extracted from ``EngineRuntime._register_reference_cadences``.  See that
    method's docstring for the full explanation; this function is behaviour-
    identical.

    Parameters
    ----------
    slots:
        The list of ``AccountSlot`` instances already constructed by
        ``build_slot``.
    market_state:
        The shared ``MarketState`` instance to register cadences on.
    """
    from ..marketdata.decision_input import from_engine

    for slot in slots:
        sym = slot.cfg.reference_symbol
        dic = from_engine(slot.cfg)
        market_state.set_reference_cadence(
            sym,
            sampling_dt_seconds=dic.sampling_dt_seconds,
            lookback_seconds=dic.vol_lookback_seconds,
        )
        for dt_s, _n in Scanner.cadence_by_class(slot.cfg).values():
            market_state.set_reference_cadence(
                sym,
                sampling_dt_seconds=dt_s,
                lookback_seconds=dic.vol_lookback_seconds,
            )
        # Couple the σ/OHLC source (mark | bbo) per reference symbol. Unlike
        # the cadence (which now accepts multiple per symbol), the source is
        # fail-fast: slots sharing a symbol must agree on one σ source.
        # Default "mark" preserves HL behaviour bit-identically.
        market_state.set_reference_source(
            sym,
            dic.reference_source,
        )
