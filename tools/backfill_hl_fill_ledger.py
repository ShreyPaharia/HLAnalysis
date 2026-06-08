"""One-time backfill: re-mirror HL venue user_fills into a slot's local Fill table.

SHR-74 context
--------------
The local Fill table historically held only 'router' rows (cloid-keyed, fee=0,
locally-computed closed_pnl) written by Router._book_fill.  HIP-4 settlement
payouts arrive as HL user_fills with dir="Settlement" and never go through
_book_fill, so the local realized ledger was blind to the dominant PnL component.

The fix (state.py: Fill.source, StateDAL.mirror_venue_fills, realized_pnl_since
source-preference) is forward-correct from the deploy date, but pre-existing
state.db files on the prod box still hold only stale router rows.  This tool
wipes them and rebuilds from venue truth.

What it does (per HL slot)
--------------------------
1. Print the BEFORE realized_pnl_since(0) (stale router-row figure).
2. Wipe ALL Fill rows and ALL Settlement rows from the DB.
   - Settlement rows: booked under the old "settlement as full-notional loss"
     shape (SHR-74); they're wrong and must not mix with the new venue-mirror.
   - Fill rows: the old router rows (cloid-keyed, fee=0) must be gone before
     mirror_venue_fills runs, because append_fill dedups by fill_id — a stale
     router row keyed to the same tid would silently block the venue insert.
3. Re-mirror all HL venue user_fills (since_ts_ns=0) via
   StateDAL.mirror_venue_fills so realized_pnl_since(0) == venue truth.
4. Print the AFTER figure and the count of rows inserted.  The two realized
   figures must match exactly (within float precision).

PM slots are SKIPPED.  Their 'router' ledger IS authoritative (PM has no venue
realized counter) and must not be wiped.

Safety
------
- ``--dry-run``: computes and prints the before/after without touching the DB.
- ``--alias``: limit to a single slot (useful for targeted repair or testing).

Run on the prod box via SSM (env-sourced for credentials):

    # dry run — see what would change:
    set -a; . /etc/hl-engine/env
    .venv/bin/python tools/backfill_hl_fill_ledger.py --dry-run

    # execute all HL slots:
    .venv/bin/python tools/backfill_hl_fill_ledger.py

    # execute a single slot (e.g. v31):
    .venv/bin/python tools/backfill_hl_fill_ledger.py --alias v31
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hlanalysis.engine.state import StateDAL


# ---------------------------------------------------------------------------
# Pure / testable core
# ---------------------------------------------------------------------------

def backfill_dal(
    dal: "StateDAL",
    venue_fills,
    *,
    symbol_to_question: dict[str, int] | None = None,
    dry_run: bool = False,
) -> dict:
    """Wipe stale rows and re-mirror venue fills for one HL slot's StateDAL.

    Parameters
    ----------
    dal:
        An already-migrated StateDAL instance for the slot.
    venue_fills:
        Iterable of objects with the UserFillRow shape (fill_id, cloid, symbol,
        side, price, size, fee, ts_ns, closed_pnl).  Typically the return value
        of ``exec_client.user_fills(since_ts_ns=0)``.
    symbol_to_question:
        Optional coin("#N") → question_idx mapping forwarded to mirror_venue_fills.
    dry_run:
        When True, compute and return the summary but write nothing to the DB.

    Returns
    -------
    dict with keys:
        before_realized  – realized_pnl_since(0) before any changes
        venue_realized   – Σ (closed_pnl - fee) of '#' fills from venue
        after_realized   – realized_pnl_since(0) after the wipe + mirror
                           (== before_realized when dry_run=True)
        rows_mirrored    – number of venue rows inserted (0 when dry_run=True)
        fills_wiped      – number of Fill rows deleted
        settlements_wiped – number of Settlement rows deleted
    """
    from sqlalchemy import delete as _delete
    from sqlmodel import Session as _Session, select as _select, func as _func

    from hlanalysis.engine.state import Fill, Settlement

    before_realized = dal.realized_pnl_since(0)

    # Compute venue realized from the '#'-symbol fills only (same filter as
    # mirror_venue_fills / gather_slot in reconcile_report.py).
    outcome_fills = [f for f in venue_fills if f.symbol.startswith("#")]
    venue_realized = sum(f.closed_pnl - f.fee for f in outcome_fills)

    if dry_run:
        # Count what *would* be wiped without touching the DB.
        with _Session(dal._engine) as s:
            fills_wiped = s.exec(_select(_func.count()).select_from(Fill)).one()
            settlements_wiped = s.exec(
                _select(_func.count()).select_from(Settlement)
            ).one()
        return {
            "before_realized": before_realized,
            "venue_realized": venue_realized,
            "after_realized": before_realized,   # unchanged
            "rows_mirrored": 0,
            "fills_wiped": int(fills_wiped),
            "settlements_wiped": int(settlements_wiped),
        }

    # 1. Wipe existing Fill and Settlement rows.
    with _Session(dal._engine) as s:
        fills_wiped = s.exec(_select(_func.count()).select_from(Fill)).one()
        settlements_wiped = s.exec(
            _select(_func.count()).select_from(Settlement)
        ).one()
        s.exec(_delete(Fill))
        s.exec(_delete(Settlement))
        s.commit()

    # 2. Re-mirror venue fills (idempotent; but we just wiped so every '#' fill
    #    will be inserted fresh).
    rows_mirrored = dal.mirror_venue_fills(
        outcome_fills, symbol_to_question=symbol_to_question,
    )

    after_realized = dal.realized_pnl_since(0)

    return {
        "before_realized": before_realized,
        "venue_realized": venue_realized,
        "after_realized": after_realized,
        "rows_mirrored": rows_mirrored,
        "fills_wiped": int(fills_wiped),
        "settlements_wiped": int(settlements_wiped),
    }


# ---------------------------------------------------------------------------
# IO shell
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--strategy-config",
        type=Path,
        default=Path("config/strategy.yaml"),
        help="path to strategy.yaml (default: config/strategy.yaml)",
    )
    ap.add_argument(
        "--deploy-config",
        type=Path,
        default=Path("config/deploy.yaml"),
        help="path to deploy.yaml (default: config/deploy.yaml)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print the before/after summary but do NOT write/wipe the DB",
    )
    ap.add_argument(
        "--alias",
        default=None,
        help="limit to a single slot alias (default: process all HL slots)",
    )
    args = ap.parse_args(argv)

    from hlanalysis.engine.config import (
        HyperliquidAccount,
        load_deploy_config,
        load_strategies_config,
    )
    from hlanalysis.engine.config_builders import build_exec_client
    from hlanalysis.engine.state import StateDAL

    deploy_cfg = load_deploy_config(args.deploy_config)
    strategies_cfg = load_strategies_config(args.strategy_config)

    if args.dry_run:
        print("(dry run — DB will NOT be modified)")

    any_failed = False
    processed = 0

    for s_cfg in strategies_cfg.strategies:
        alias = s_cfg.account_alias

        if args.alias is not None and alias != args.alias:
            continue

        acct = deploy_cfg.accounts.get(alias)
        if acct is None:
            print(f"[{alias}] ERROR: alias not found in deploy config — skipping",
                  file=sys.stderr)
            any_failed = True
            continue

        if not isinstance(acct, HyperliquidAccount):
            print(f"[{alias}] SKIP: not an HL slot (venue={getattr(acct, 'venue', '?')})")
            continue

        try:
            exec_client = build_exec_client(alias, acct, paper_mode=False)
            db_path = Path(deploy_cfg.state_db_path_for(alias))
            dal = StateDAL(db_path)
            dal.run_migrations()

            venue_fills = exec_client.user_fills(since_ts_ns=0)
            result = backfill_dal(dal, venue_fills, dry_run=args.dry_run)

            mode = "DRY " if args.dry_run else "    "
            print(
                f"[{alias}]{mode}"
                f"  before={result['before_realized']:+.4f}"
                f"  venue={result['venue_realized']:+.4f}"
                f"  after={result['after_realized']:+.4f}"
                f"  rows_mirrored={result['rows_mirrored']}"
                f"  fills_wiped={result['fills_wiped']}"
                f"  settlements_wiped={result['settlements_wiped']}"
            )

            if not args.dry_run:
                delta = abs(result["after_realized"] - result["venue_realized"])
                if delta > 0.01:
                    print(
                        f"[{alias}] WARNING: after_realized vs venue_realized "
                        f"diverges by {delta:.4f} — check fill coverage",
                        file=sys.stderr,
                    )
                    any_failed = True

            processed += 1

        except Exception as exc:  # noqa: BLE001
            print(f"[{alias}] ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
            any_failed = True

    if processed == 0 and args.alias is not None:
        print(
            f"ERROR: --alias '{args.alias}' did not match any HL slot",
            file=sys.stderr,
        )
        return 1

    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
