"""Out-of-band venue reconciliation report (Phase 0, W0.3/W0.4).

Run on the box via SSM (env-sourced for credentials), same pattern as
engine-diag. Per slot: compare engine-DB realized PnL + open positions against
the venue's clearinghouse_state(), report realized + open-MTM = total true PnL,
flag position drift, and alert on Telegram when any drift is detected.

Split into a PURE core (SlotRecon / compare_slot / format_report — no IO, fully
unit-tested) and a thin IO shell (gather_slot / build_report / main).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from .exec_types import ClearinghouseState

if TYPE_CHECKING:
    from .exec_client import ExecutionClient
    from .state import StateDAL


@dataclass(frozen=True, slots=True)
class Drift:
    kind: str       # "qty_mismatch" | "vanished" | "orphan"
    symbol: str
    db_qty: float
    venue_qty: float


@dataclass(frozen=True, slots=True)
class SlotRecon:
    alias: str
    realized_pnl: float
    open_mtm: float
    account_value_usd: float
    positions_known: bool
    drift: list[Drift] = field(default_factory=list)

    @property
    def total_true_pnl(self) -> float:
        return self.realized_pnl + self.open_mtm

    @property
    def has_drift(self) -> bool:
        return len(self.drift) > 0


def compare_slot(
    *,
    alias: str,
    db_positions: list[tuple[str, float]],   # (symbol, qty)
    db_realized_pnl: float,
    venue: ClearinghouseState,
    qty_tolerance: float,
) -> SlotRecon:
    """Pure three-way-ish compare: DB positions vs venue positions, plus PnL.

    Open MTM = Σ venue unrealized_pnl. Position drift is skipped entirely when
    venue.positions_known is False (PM data-api flap) so an empty venue set is
    never mistaken for 'everything vanished'.
    """
    open_mtm = sum(vp.unrealized_pnl for vp in venue.positions)
    drift: list[Drift] = []

    if venue.positions_known:
        venue_by_sym = {vp.symbol: vp.qty for vp in venue.positions}
        db_by_sym = {sym: qty for sym, qty in db_positions}

        for sym, db_qty in db_by_sym.items():
            v_qty = venue_by_sym.get(sym)
            if v_qty is None:
                drift.append(Drift("vanished", sym, db_qty, 0.0))
            elif abs(v_qty - db_qty) > qty_tolerance:
                drift.append(Drift("qty_mismatch", sym, db_qty, v_qty))
        for sym, v_qty in venue_by_sym.items():
            if sym not in db_by_sym:
                drift.append(Drift("orphan", sym, 0.0, v_qty))

    return SlotRecon(
        alias=alias,
        realized_pnl=db_realized_pnl,
        open_mtm=open_mtm,
        account_value_usd=venue.account_value_usd,
        positions_known=venue.positions_known,
        drift=drift,
    )


def format_report(recon: list[SlotRecon]) -> str:
    """Plain-text/HTML-safe report. One block per slot, drift called out.

    Kept dependency-free so it can be unit-tested and reused by the Telegram
    alert (TelegramClient.send escapes <, >, & itself).
    """
    lines: list[str] = ["Engine reconciliation report", ""]
    for r in recon:
        status = "DRIFT" if r.has_drift else "OK"
        lines.append(f"[{r.alias}] {status}")
        lines.append(
            f"  realized={r.realized_pnl:+.2f}  open_mtm={r.open_mtm:+.2f}  "
            f"true_pnl={r.total_true_pnl:+.2f}  acct_value={r.account_value_usd:.2f}"
        )
        if not r.positions_known:
            lines.append("  (positions unknown — recon skipped this cycle)")
        for d in r.drift:
            lines.append(
                f"  ! {d.kind} {d.symbol}: db_qty={d.db_qty} venue_qty={d.venue_qty}"
            )
        lines.append("")
    return "\n".join(lines).rstrip()


def gather_slot(*, alias: str, dal: StateDAL, exec_client: ExecutionClient, qty_tolerance: float) -> SlotRecon:
    """IO: pull realized PnL (incl settlement) + DB positions + venue state for
    one slot and run the pure compare. clearinghouse_state() is a blocking SDK
    call; the caller offloads it via asyncio.to_thread when needed."""
    db_realized = dal.realized_pnl_since(0)
    db_positions = [(p.symbol, p.qty) for p in dal.all_positions()]
    venue = exec_client.clearinghouse_state()
    return compare_slot(
        alias=alias, db_positions=db_positions, db_realized_pnl=db_realized,
        venue=venue, qty_tolerance=qty_tolerance,
    )


def build_report(deploy_cfg, strategies_cfg, *, qty_tolerance: float) -> list[SlotRecon]:
    """IO: build a read-only client + open the DAL per slot, gather each.

    A slot whose client/DAL errors yields a positions_known=False SlotRecon so
    one bad slot never aborts the whole report."""
    from .config_builders import build_exec_client
    from .state import StateDAL

    out: list[SlotRecon] = []
    for s_cfg in strategies_cfg.strategies:
        alias = s_cfg.account_alias
        try:
            acct = deploy_cfg.accounts[alias]
            client = build_exec_client(alias, acct, paper_mode=True)  # read-only path
            dal = StateDAL(Path(deploy_cfg.state_db_path_for(alias)))
            out.append(gather_slot(alias=alias, dal=dal, exec_client=client,
                                   qty_tolerance=qty_tolerance))
        except Exception as e:  # noqa: BLE001 — a bad slot must not abort the report
            out.append(SlotRecon(alias=alias, realized_pnl=0.0, open_mtm=0.0,
                                 account_value_usd=0.0, positions_known=False,
                                 drift=[]))
            logger.warning("recon slot {} failed: {}", alias, e)
    return out


async def _maybe_alert(
    report_text: str,
    has_drift: bool,
    *,
    session_factory=None,
    tg_factory=None,
) -> None:
    """Send the report to Telegram when there is drift. Credentials come from
    the same env the engine uses; no-op if unset.

    session_factory and tg_factory are injectable for testing; both default to
    the real aiohttp.ClientSession and TelegramClient respectively."""
    import os
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat = os.environ.get("TELEGRAM_CHAT_ID")
    if not (has_drift and token and chat):
        return
    if session_factory is None:
        import aiohttp as _aiohttp
        session_factory = _aiohttp.ClientSession
    if tg_factory is None:
        from hlanalysis.alerts.telegram import TelegramClient
        tg_factory = TelegramClient
    async with session_factory() as session:
        tg = tg_factory(bot_token=token, chat_id=chat, session=session)
        await tg.send("⚠️ Reconciliation DRIFT\n\n" + report_text)


def main() -> None:
    p = argparse.ArgumentParser(description="Out-of-band venue reconciliation report.")
    p.add_argument("--strategy-config", type=Path, default=Path("config/strategy.yaml"))
    p.add_argument("--deploy-config", type=Path, default=Path("config/deploy.yaml"))
    p.add_argument("--qty-tolerance", type=float, default=1e-6)
    p.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = p.parse_args()

    from .config import load_deploy_config, load_strategies_config
    deploy_cfg = load_deploy_config(args.deploy_config)
    strategies_cfg = load_strategies_config(args.strategy_config)

    recon = build_report(deploy_cfg, strategies_cfg, qty_tolerance=args.qty_tolerance)
    has_drift = any(r.has_drift for r in recon)
    text = format_report(recon)

    if args.json:
        print(json.dumps({
            "generated_at_ns": time.time_ns(),
            "has_drift": has_drift,
            "slots": [
                {"alias": r.alias, "realized_pnl": r.realized_pnl,
                 "open_mtm": r.open_mtm, "total_true_pnl": r.total_true_pnl,
                 "account_value_usd": r.account_value_usd,
                 "positions_known": r.positions_known,
                 "drift": [vars(d) for d in r.drift]}
                for r in recon
            ],
        }))
    else:
        print(text)

    asyncio.run(_maybe_alert(text, has_drift))
    # Nonzero exit on drift so an SSM/cron caller can detect it.
    raise SystemExit(1 if has_drift else 0)


if __name__ == "__main__":
    main()
