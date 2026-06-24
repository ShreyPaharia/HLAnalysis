"""Component 3 — engine-diag one-shot JSON snapshot.

Run as: python -m hlanalysis.engine.diag [options]

Reads state.db (including the events table from Component 2) and flag files
on disk, and prints ONE JSON object to stdout. No engine introspection / no
IPC — safe to run against a live engine over SSM.

Unified-DB awareness: if `<root>/state.db` is the shared multi-slot DB
(detected by the presence of a `strategy_id` column on the `events` table),
it is used for all slots and all queries are scoped by `strategy_id=alias`.
Otherwise the legacy per-slot `<root>/<alias>/state.db` layout is used.

Per-slot snapshot sections:
  status           — running / paper / halted / blocked (flags + Session)
  positions        — open positions + true_pnl breakdown
  open_orders      — live orders + age_seconds
  feed             — last heartbeat ts, events-ingested count, stale/down state
  flags            — restart_blocked, halt — presence + mtime_ns
  rejects          — rolling counts by kind/reason (last N hours), filtered to alias
  last_decision    — most recent entry/exit/risk_veto event for this alias
  config_fingerprint — stable hash of effective strategy params + key params inline

Top-level:
  generated_at_ns — snapshot creation time (monotonic wall-clock ns)
  data_dir        — engine data root derived from deploy config
  slots           — dict[alias -> per-slot snapshot]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Unified-DB detection
# ---------------------------------------------------------------------------


def _is_unified_db(db_path: Path) -> bool:
    """Return True if db_path is a unified multi-slot DB.

    Detection: the ``events`` table exists AND has a ``strategy_id`` column
    (added in migration 0006_unified_slot_db).  A per-slot legacy DB only has
    ``alias``; a unified DB has both.  Missing file or missing table → False.
    """
    if not db_path.exists():
        return False
    try:
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute("PRAGMA table_info(events)").fetchall()
    except sqlite3.Error:
        return False
    # PRAGMA table_info columns: (cid, name, type, notnull, dflt_value, pk)
    return any(row[1] == "strategy_id" for row in rows)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _flag_info(path: Path) -> dict[str, Any]:
    """Return {present, mtime_ns} for a flag file path."""
    try:
        stat = path.stat()
        return {"present": True, "mtime_ns": int(stat.st_mtime_ns)}
    except FileNotFoundError:
        return {"present": False, "mtime_ns": None}


def _open_dal(db_path: Path):
    """Return a StateDAL for db_path, or None if the DB does not exist."""
    if not db_path.exists():
        return None
    # Import here to avoid top-level import cost (module is imported at test collection)
    from hlanalysis.engine.state import StateDAL

    return StateDAL(db_path)


def _reject_counts_since_alias(
    db_path: Path,
    *,
    alias: str,
    since_ts_ns: int,
    strategy_id: str | None = None,
) -> list[dict[str, Any]]:
    """Return reject-like counts grouped by (kind, reason) for a specific alias.

    Filters by alias (legacy DB) or strategy_id (unified DB) so cross-slot
    events in the same DB don't double-count.
    Returns [] if the DB doesn't exist or the events table is absent.

    Pass ``strategy_id`` when querying a unified DB; it takes precedence over
    the ``alias`` column filter so the correct scoping column is used.
    """
    if not db_path.exists():
        return []
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            if strategy_id is not None:
                rows = conn.execute(
                    """
                    SELECT kind, reason, COUNT(*) AS count,
                           MAX(payload_json) AS sample_payload
                    FROM events
                    WHERE ts_ns >= ? AND strategy_id = ?
                    GROUP BY kind, reason
                    ORDER BY count DESC
                    """,
                    (since_ts_ns, strategy_id),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT kind, reason, COUNT(*) AS count,
                           MAX(payload_json) AS sample_payload
                    FROM events
                    WHERE ts_ns >= ? AND alias = ?
                    GROUP BY kind, reason
                    ORDER BY count DESC
                    """,
                    (since_ts_ns, alias),
                ).fetchall()
        return [dict(r) for r in rows]
    except sqlite3.OperationalError:
        # events table not yet created (migration not run)
        return []


def _last_event_by_kinds(
    db_path: Path,
    *,
    kinds: list[str],
    alias: str | None,
    strategy_id: str | None = None,
) -> dict[str, Any] | None:
    """Return the most-recent event matching any of the given kinds.

    If alias is provided, filter to that alias (legacy DB) OR to strategy_id
    (unified DB — pass ``strategy_id`` to use the correct scoping column).
    Returns None on any error.
    """
    if not db_path.exists():
        return None
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            placeholders = ",".join("?" for _ in kinds)
            if strategy_id is not None:
                row = conn.execute(
                    f"SELECT id, ts_ns, alias, strategy_id, kind, question_idx, reason, payload_json "
                    f"FROM events WHERE kind IN ({placeholders}) AND strategy_id = ? "
                    f"ORDER BY ts_ns DESC LIMIT 1",
                    [*kinds, strategy_id],
                ).fetchone()
            elif alias is not None:
                row = conn.execute(
                    f"SELECT id, ts_ns, alias, kind, question_idx, reason, payload_json "
                    f"FROM events WHERE kind IN ({placeholders}) AND alias = ? "
                    f"ORDER BY ts_ns DESC LIMIT 1",
                    [*kinds, alias],
                ).fetchone()
            else:
                row = conn.execute(
                    f"SELECT id, ts_ns, alias, kind, question_idx, reason, payload_json "
                    f"FROM events WHERE kind IN ({placeholders}) "
                    f"ORDER BY ts_ns DESC LIMIT 1",
                    kinds,
                ).fetchone()
        return dict(row) if row is not None else None
    except sqlite3.OperationalError:
        return None


def _last_event_by_kind(db_path: Path, *, kind: str, alias: str | None) -> dict[str, Any] | None:
    """Return the most-recent event matching kind (and alias if given)."""
    return _last_event_by_kinds(db_path, kinds=[kind], alias=alias)


def _get_last_session(db_path: Path) -> dict[str, Any] | None:
    """Return the most recent session row (or None if absent/error)."""
    if not db_path.exists():
        return None
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT session_id, started_ts_ns, ended_ts_ns, halt_reason "
                "FROM session ORDER BY started_ts_ns DESC LIMIT 1"
            ).fetchone()
        return dict(row) if row is not None else None
    except sqlite3.OperationalError:
        return None


# ---------------------------------------------------------------------------
# Per-slot snapshot builder
# ---------------------------------------------------------------------------


def _build_slot_snapshot(
    alias: str,
    strategy_cfg: Any,  # StrategyConfig
    db_path: Path,
    halt_flag_path: Path,
    restart_blocked_path: Path,
    *,
    reject_window_hours: float = 24.0,
    now_ns: int,
    strategy_id: str | None = None,
) -> dict[str, Any]:
    """Build one slot's snapshot dict.

    ``strategy_id``: when querying a unified DB, pass this to scope all queries
    to the correct slot. When None (legacy per-slot DB), alias-based filtering
    is used instead.
    """

    # ---- flag files ----
    halt_info = _flag_info(halt_flag_path)
    blocked_info = _flag_info(restart_blocked_path)
    # Venue-PnL fail-safe halt (auto-clearing, written by _venue_io). Distinct
    # from the persistent operator/daily-loss `halt` flag — surfaced here so a
    # slot stuck on the venue-fail latch is visible, not reported as "running"
    # (incident 2026-06-24).
    venue_pnl_halt_info = _flag_info(halt_flag_path.parent / "venue_pnl_halt")
    flags: dict[str, Any] = {
        "restart_blocked": blocked_info,
        "halt": halt_info,
        "venue_pnl_halt": venue_pnl_halt_info,
    }

    # ---- status ----
    # Priority: blocked > halted > paper > running
    if blocked_info["present"]:
        status = "blocked"
    elif halt_info["present"] or venue_pnl_halt_info["present"]:
        status = "halted"
    elif strategy_cfg.paper_mode:
        status = "paper"
    else:
        # Check last session: if the most recent session has ended_ts_ns set
        # and there's no active heartbeat, the engine is likely stopped.
        # We keep it simple: no flag + not paper = running (the engine runs as
        # a systemd service; absence of restart_blocked/halt means it's active).
        status = "running"

    # ---- open DAL (may be None if DB missing) ----
    dal = _open_dal(db_path)

    # ---- positions ----
    open_positions: list[dict[str, Any]] = []
    fills_realized = 0.0
    settlement = 0.0
    if dal is not None:
        try:
            positions = dal.all_positions(strategy_id=strategy_id or "")
            # fills_realized = realized_pnl_since(0) - settlement_pnl_since(0)
            # because realized_pnl_since already includes settlement
            total_realized = dal.realized_pnl_since(0, strategy_id=strategy_id or "")
            settlement = dal.settlement_pnl_since(0, strategy_id=strategy_id or "")
            fills_realized = total_realized - settlement

            for p in positions:
                open_positions.append(
                    {
                        "question_idx": p.question_idx,
                        "symbol": p.symbol,
                        "qty": p.qty,
                        "avg_entry": p.avg_entry,
                        "stop_loss_price": p.stop_loss_price,
                    }
                )
        except Exception:
            pass

    positions_summary = {
        "open_count": len(open_positions),
        "positions": open_positions,
        "fills_realized_pnl": fills_realized,
        "settlement_pnl": settlement,
        "true_pnl": fills_realized + settlement,
    }

    # ---- open orders ----
    live_orders: list[dict[str, Any]] = []
    if dal is not None:
        try:
            orders = dal.live_orders(strategy_id=strategy_id or "")
            for o in orders:
                age_seconds = (now_ns - o.placed_ts_ns) / 1e9
                live_orders.append(
                    {
                        "cloid": o.cloid,
                        "venue_oid": o.venue_oid,
                        "question_idx": o.question_idx,
                        "symbol": o.symbol,
                        "side": o.side,
                        "price": o.price,
                        "size": o.size,
                        "status": o.status,
                        "placed_ts_ns": o.placed_ts_ns,
                        "age_seconds": age_seconds,
                    }
                )
        except Exception:
            pass

    # ---- feed info ----
    # Heartbeat events are cross-slot (alias=None/empty); query without slot filter.
    # In the unified DB they have strategy_id=None so we also pass no strategy_id.
    feed: dict[str, Any] = {
        "last_heartbeat_ts_ns": None,
        "events_ingested": None,
        "last_feed_status": None,
        "last_feed_status_ts_ns": None,
    }
    hb_event = _last_event_by_kind(db_path, kind="engine_heartbeat", alias=None)
    if hb_event is not None:
        feed["last_heartbeat_ts_ns"] = hb_event["ts_ns"]
        if hb_event.get("payload_json"):
            try:
                payload = json.loads(hb_event["payload_json"])
                feed["events_ingested"] = payload.get("events_ingested")
                feed["d_events"] = payload.get("d_events")
                feed["n_questions"] = payload.get("n_questions")
            except (json.JSONDecodeError, TypeError):
                pass

    feed_status_event = _last_event_by_kinds(
        db_path,
        kinds=["feed_stale", "feed_down", "feed_recovered"],
        alias=None,
    )
    if feed_status_event is not None:
        feed["last_feed_status"] = feed_status_event["kind"]
        feed["last_feed_status_ts_ns"] = feed_status_event["ts_ns"]

    # ---- rejects (scoped to this slot) ----
    reject_window_ns = int(reject_window_hours * 3600 * 1e9)
    since_ts_ns = now_ns - reject_window_ns
    rejects = _reject_counts_since_alias(
        db_path,
        alias=alias,
        since_ts_ns=since_ts_ns,
        strategy_id=strategy_id,
    )

    # ---- last_decision: most recent entry/exit/risk_veto/risk_halt for this slot ----
    decision_kinds = [
        "entry",
        "exit",
        "risk_veto",
        "risk_halt",
        "stop_loss_triggered",
        "daily_loss_halt",
        "stale_data_halt",
        "order_rejected",
    ]
    last_decision = _last_event_by_kinds(
        db_path,
        kinds=decision_kinds,
        alias=alias,
        strategy_id=strategy_id,
    )

    # ---- config fingerprint ----
    config_fingerprint = _build_config_fingerprint(alias, strategy_cfg)

    return {
        "status": status,
        "positions": positions_summary,
        "open_orders": live_orders,
        "feed": feed,
        "flags": flags,
        "rejects": rejects,
        "last_decision": last_decision,
        "config_fingerprint": config_fingerprint,
    }


def _build_config_fingerprint(alias: str, strategy_cfg: Any) -> dict[str, Any]:
    """Build a stable config fingerprint for a strategy slot.

    Hash comes from the shared ``strategy_config_sig`` function so that the
    engine-diag hash equals the backtest ``slot config_sig`` for the same slot
    config — the whole point of R5.  Key params are inlined alongside the hash
    for quick config-vs-runtime skew detection (e.g. exit_safety_d=0.0 when
    YAML says 1.0 — SHR-65).
    """
    # Import here to avoid a top-level circular import (diag.py is imported
    # early; config.py imports nothing from diag.py so the lazy import is safe).
    from hlanalysis.engine.config import strategy_config_sig

    defaults = strategy_cfg.defaults
    digest = strategy_config_sig(strategy_cfg)

    # Inline the key params most likely to cause config-vs-runtime skew
    inline: dict[str, Any] = {
        "hash": digest,
        "strategy_type": strategy_cfg.strategy_type,
        "paper_mode": strategy_cfg.paper_mode,
        "exit_safety_d": getattr(defaults, "exit_safety_d", None),
        "min_safety_d": getattr(defaults, "min_safety_d", None),
        "tte_max_seconds": getattr(defaults, "tte_max_seconds", None),
        "vol_estimator": getattr(defaults, "vol_estimator", None),
        "vol_sampling_dt_seconds": getattr(defaults, "vol_sampling_dt_seconds", None),
        "reference_symbol": strategy_cfg.reference_symbol,
        "reference_sigma_source": strategy_cfg.reference_sigma_source,
    }

    # Add theta-specific skew-prone params when present
    if strategy_cfg.theta is not None:
        inline["theta_exit_safety_d"] = strategy_cfg.theta.exit_safety_d
        inline["theta_vol_estimator"] = strategy_cfg.theta.vol_estimator

    return inline


# ---------------------------------------------------------------------------
# Main snapshot builder (public API, called by tests and by __main__)
# ---------------------------------------------------------------------------


def build_snapshot(
    deploy_cfg: Any,  # DeployConfig
    strategies_cfg: Any,  # StrategiesConfig
    *,
    reject_window_hours: float = 24.0,
    alias_filter: str | None = None,
    now_ns: int | None = None,
) -> dict[str, Any]:
    """Build and return the full snapshot dict.

    Args:
        deploy_cfg: DeployConfig (loaded from deploy.yaml)
        strategies_cfg: StrategiesConfig (loaded from strategy.yaml)
        reject_window_hours: how far back to look for rejects (default 24h)
        alias_filter: if set, only include this alias in the snapshot
        now_ns: override current time (for testing)
    """
    if now_ns is None:
        now_ns = time.time_ns()

    # data_dir: parent of the state_db_path
    data_dir = str(Path(deploy_cfg.state_db_path).parent)

    # Detect unified DB: single shared <root>/state.db with strategy_id column.
    # When present, all slots read from one file scoped by strategy_id=alias.
    # When absent, fall back to legacy per-slot <root>/<alias>/state.db layout.
    unified_db_path = deploy_cfg.state_db_path_shared()
    use_unified = _is_unified_db(unified_db_path)

    slots: dict[str, Any] = {}

    for s_cfg in strategies_cfg.strategies:
        alias = s_cfg.account_alias
        if alias_filter is not None and alias != alias_filter:
            continue

        if use_unified:
            # Unified layout: one DB, per-slot data scoped by strategy_id.
            db_path = unified_db_path
            slot_strategy_id: str | None = alias
            # Flag files live in <root>/<alias>/ (slot_dir_for layout).
            slot_dir = deploy_cfg.slot_dir_for(alias)
            kill_switch_name = Path(deploy_cfg.kill_switch_path).name
            halt_flag_path = slot_dir / kill_switch_name
            restart_blocked_path = slot_dir / "restart_blocked"
        else:
            # Legacy layout: per-slot DB at <root>/<alias>/state.db.
            db_path = Path(deploy_cfg.state_db_path_for(alias))
            slot_strategy_id = None
            kill_switch_path_str = deploy_cfg.kill_switch_path_for(alias)
            halt_flag_path = Path(kill_switch_path_str)
            restart_blocked_path = halt_flag_path.parent / "restart_blocked"

        slot_snapshot = _build_slot_snapshot(
            alias=alias,
            strategy_cfg=s_cfg,
            db_path=db_path,
            halt_flag_path=halt_flag_path,
            restart_blocked_path=restart_blocked_path,
            reject_window_hours=reject_window_hours,
            now_ns=now_ns,
            strategy_id=slot_strategy_id,
        )
        slots[alias] = slot_snapshot

    return {
        "generated_at_ns": now_ns,
        "data_dir": data_dir,
        "slots": slots,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "engine-diag: one-shot JSON snapshot of engine state. "
            "Reads state.db + flag files on disk; safe against a live engine."
        )
    )
    p.add_argument(
        "--strategy-config",
        type=Path,
        default=Path("config/strategy.yaml"),
        help="Path to strategy.yaml (default: config/strategy.yaml)",
    )
    p.add_argument(
        "--deploy-config",
        type=Path,
        default=Path("config/deploy.yaml"),
        help="Path to deploy.yaml (default: config/deploy.yaml)",
    )
    p.add_argument(
        "--reject-window-hours",
        type=float,
        default=24.0,
        help="Rolling window for reject counts in hours (default: 24)",
    )
    p.add_argument(
        "--alias",
        type=str,
        default=None,
        help="Only include this alias in the snapshot (default: all slots)",
    )
    p.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output (default: compact)",
    )
    args = p.parse_args()

    # Import here to avoid pulling in all engine deps at module import time
    from hlanalysis.engine.config import load_deploy_config, load_strategies_config

    deploy_cfg = load_deploy_config(args.deploy_config)
    strategies_cfg = load_strategies_config(args.strategy_config)

    snapshot = build_snapshot(
        deploy_cfg,
        strategies_cfg,
        reject_window_hours=args.reject_window_hours,
        alias_filter=args.alias,
    )

    indent = 2 if args.pretty else None
    print(json.dumps(snapshot, indent=indent))


if __name__ == "__main__":
    main()
