"""Event persistence sink: consume every BusEvent and write it to one or more
slot DBs' ``events`` table for engine observability.

Extracted from runtime.py. The two near-identical persist loops that used to
live there (a @staticmethod single-DAL variant for tests + an instance method
fanning out to every slot's DAL) are unified here into ONE free function taking
a list of DALs — the instance method passes ``[s.dal for s in self.slots]``,
tests pass a single-element list. One loop = no drift hazard between the column
extraction / prune cadence of the two copies.
"""
from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

from .risk_events import BusEvent
from .state import StateDAL


def event_columns(ev: BusEvent) -> dict[str, Any]:
    """Extract the stable, queryable columns from a bus event.

    Single source of truth for how an event maps to the `events` table.
    ``reason`` is normalised across event types: ``.reason`` (RiskVeto/RiskHalt/
    StaleDataHalt/Exit/ReconcileDrift) or ``.error`` (OrderRejected), else None.
    The full event is kept as ``payload_json`` for fidelity.
    """
    return {
        "ts_ns": ev.ts_ns,
        "alias": getattr(ev, "account_alias", None) or None,
        "kind": ev.kind,
        "question_idx": getattr(ev, "question_idx", None),
        "reason": getattr(ev, "reason", None) or getattr(ev, "error", None) or None,
        "payload_json": ev.model_dump_json(),
    }


async def events_persist_loop(
    sub: "asyncio.Queue[BusEvent]",
    dals: list[StateDAL],
    *,
    max_age_ns: int,
    max_rows: int,
    prune_every_n: int = 500,
) -> None:
    """Consume every BusEvent and write it to each DAL's events table.

    Symmetric to AlertRules.run(alerts_sub): loops on sub.get(), extracts the
    stable fields (alias, kind, question_idx, reason) into named columns for SQL
    queries, and writes the full event as payload_json for fidelity. The same
    event is written to EVERY dal in ``dals`` — all slots' DBs are independent
    (one state.db per alias), so each slot keeps a full engine-wide event log
    (cheap; event volume is low) and no cross-alias query is needed.

    Terminates via task cancellation (CancelledError from sub.get()), matching
    the alerts loop pattern.

    Prune runs every prune_every_n inserts (not per-insert) to avoid a steady
    per-row overhead while still bounding growth between long idle periods. Both
    age and row-count bounds are applied on each prune call.

    Exposed as a free function so tests can call it directly (with a single-DAL
    list) without constructing a full EngineRuntime.
    """
    inserted = 0
    while True:
        ev = await sub.get()
        try:
            cols = event_columns(ev)
            for dal in dals:
                dal.append_event(**cols)
            inserted += 1
            if inserted % prune_every_n == 0:
                try:
                    for dal in dals:
                        dal.prune_events(max_age_ns=max_age_ns, max_rows=max_rows)
                except Exception:
                    logger.exception("events_persist_loop: prune failed")
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("events_persist_loop: failed to persist {}", ev.kind)
