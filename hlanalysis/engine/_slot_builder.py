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

import os
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
from .scoped_dal import StrategyScopedDAL
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
    shared_dal: CachedStateDAL | None = None,
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
    shared_dal:
        Optional pre-built ``CachedStateDAL`` that all slots share.  When
        provided (the normal engine path, where ``EngineRuntime.run()``
        constructs one DAL before the slot-build loop), this DAL is wrapped
        per-slot via ``StrategyScopedDAL`` so every slot sees only its own
        rows.  When ``None`` (legacy/test path), a fresh ``CachedStateDAL``
        is built from ``deploy_cfg.state_db_path_shared()``.
    """
    # Import here to avoid a module-level cycle (AccountSlot / PmSlotState
    # live in runtime.py, which imports this module).
    from .runtime import AccountSlot, PmSlotState

    # ``account_alias`` is the unique per-slot identifier enforced by
    # ``load_strategies_config``.  The effective ``strategy_id`` used for DB
    # row scoping and per-slot flag files is:
    #   * ``s_cfg.strategy_id`` when the operator sets it explicitly — enabling
    #     a future "Stage 1" where two strategy entries share the same account
    #     but use different DB-scoping keys (config-only change, no code change).
    #   * ``account_alias`` otherwise — the current default, preserving today's
    #     1-strategy-per-account invariant where ``account_alias`` is a stable
    #     proxy for "this slot".
    alias = s_cfg.account_alias
    strategy_id = s_cfg.strategy_id if s_cfg.strategy_id is not None else alias
    if alias not in deploy_cfg.accounts:
        raise ValueError(
            f"strategy '{s_cfg.name}' references account_alias={alias!r} but "
            f"deploy.accounts has only {list(deploy_cfg.accounts)}",
        )
    acct = deploy_cfg.accounts[alias]

    # Shared DB: one file for the whole engine, scoped per slot by strategy_id.
    # Per-strategy flag/sibling files live in slot_dir_for(strategy_id).
    shared_db_path = deploy_cfg.state_db_path_shared()
    slot_dir = deploy_cfg.slot_dir_for(strategy_id)
    slot_dir.mkdir(parents=True, exist_ok=True)
    kill_switch_path = slot_dir / Path(deploy_cfg.kill_switch_path).name

    cloid_prefix = f"hla-{alias}-"

    # DAL: build or reuse the shared CachedStateDAL; wrap with a strategy-
    # scoped view so this slot only reads/writes its own rows.
    # run_migrations is idempotent; calling it here (when shared_dal is None)
    # ensures single-slot / test callers don't need an extra setup step.
    if shared_dal is None:
        _base_dal = CachedStateDAL(shared_db_path)
        _base_dal.run_migrations()
    else:
        _base_dal = shared_dal
    # strategy_id = s_cfg.strategy_id ?? account_alias (explicit config value
    #   wins; falls back to account_alias, the current default).
    # account     = account_alias (venue-agnostic; account_address is HL-only).
    # Using account_alias for account keeps the scoping consistent and avoids
    # attribute differences between HyperliquidAccount and PolymarketAccount.
    dal = StrategyScopedDAL(_base_dal, strategy_id=strategy_id, account=alias)

    if exec_client_factory is not None:
        exec_client = exec_client_factory(alias, acct, s_cfg.paper_mode)
    else:
        exec_client = build_exec_client(alias, acct, s_cfg.paper_mode)

    risk = RiskGate(s_cfg)
    # Durable trade journal (SHR-83): persists into the shared state.db under
    # this slot's strategy_id, shared by the Router (decision/send/reject/fill)
    # and Reconciler (late fills).
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
        strategy_id=strategy_id,
        cloid_prefix=cloid_prefix,
        reduce_close_atol=reduce_close_atol,
        journal=journal,
    )
    strategy = _build_strategy_for_slot(s_cfg)
    # Gate-decision log lives in the per-strategy slot directory so each slot's
    # decisions remain in separate files. Operators tail this during
    # forward-testing to see which gates are firing without combing
    # through journal heartbeats. State-change-debounced, so file size
    # stays small (one line per question per transition).
    gate_log_path = slot_dir / "gate_decisions.jsonl"

    # Per-scan decision trace — gated off by default. Set HLBT_TRACE_QUESTIONS
    # (comma-separated) to enable. Each token is either a numeric question_idx
    # (e.g. "1000433") OR a "COIN:KLASS" coin/class filter (e.g.
    # "BTC:priceBinary") that traces any question with that underlying+class —
    # use this to capture tomorrow's question from open without knowing its
    # (irregularly-assigned) idx. HLBT_TRACE_PATH overrides the output path.
    _trace_env = os.environ.get("HLBT_TRACE_QUESTIONS", "").strip()
    _trace_idxs: frozenset[int] = frozenset()
    _trace_filters: frozenset[tuple[str, str]] = frozenset()
    if _trace_env:
        _idxs: set[int] = set()
        _filters: set[tuple[str, str]] = set()
        for tok in _trace_env.split(","):
            tok = tok.strip()
            if not tok:
                continue
            if tok.isdigit():
                _idxs.add(int(tok))
            elif ":" in tok:
                coin, _, klass = tok.partition(":")
                _filters.add((coin.strip(), klass.strip()))
        _trace_idxs = frozenset(_idxs)
        _trace_filters = frozenset(_filters)
    _trace_on = bool(_trace_idxs) or bool(_trace_filters)
    _trace_path_env = os.environ.get("HLBT_TRACE_PATH", "").strip()
    if _trace_on:
        _decision_trace_path: Path | None = (
            Path(_trace_path_env) if _trace_path_env else slot_dir / "decision_trace.jsonl"
        )
    else:
        _decision_trace_path = None

    # Compute config fingerprint once for the trace rows (stable 16-hex SHA-256).
    from .config import strategy_config_sig

    _config_hash: str | None = strategy_config_sig(s_cfg) if _trace_on else None

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
        decision_trace_path=_decision_trace_path,
        trace_question_idxs=_trace_idxs,
        trace_filters=_trace_filters,
        strategy_id=strategy_id,
        config_hash=_config_hash,
    )
    return AccountSlot(
        cfg=s_cfg,
        account_cfg=acct,
        venue=acct.venue,
        state_db_path=shared_db_path,
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
