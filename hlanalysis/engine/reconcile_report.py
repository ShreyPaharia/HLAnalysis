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
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from .exec_types import ClearinghouseState
# Reuse the live engine reconcile's qty-drift tolerances so the report never
# flags a position diff the engine itself tolerates. PM data-api settled sizes
# carry ~8e-3-share rounding noise vs our booked size; with a tighter tolerance
# the daily report false-flags PM slots as DRIFT every cycle (2026-06-08).
from .reconcile import _QTY_MISMATCH_ABS_TOL, _QTY_MISMATCH_REL_TOL

if TYPE_CHECKING:
    from .exec_client import ExecutionClient
    from .state import StateDAL


# SHR-77: render order + friendly labels for the per-class daily-report split.
# Any class not listed here (shouldn't happen for HL) is appended, sorted, under
# its raw key so nothing is silently dropped.
_KLASS_ORDER = ("priceBinary", "priceBucket", "unknown")
_KLASS_LABELS = {
    "priceBinary": "binary",
    "priceBucket": "bucket",
    "unknown": "unknown",
}

_24H_NS = 24 * 3600 * 1_000_000_000


@dataclass(frozen=True, slots=True)
class Drift:
    kind: str       # "qty_mismatch" | "vanished" | "orphan"
    symbol: str
    db_qty: float
    venue_qty: float


@dataclass(frozen=True, slots=True)
class KlassStat:
    """Per-market-class slice of a slot's outcome PnL (SHR-77). realized_pnl is
    the venue closedPnl-fee of fills in this class; open_mtm is the unrealized
    PnL of still-open positions in this class. total_pnl mirrors the slot's
    total_true_pnl decomposition (realized + open MTM)."""
    realized_pnl: float
    open_mtm: float
    fills: int

    @property
    def total_pnl(self) -> float:
        return self.realized_pnl + self.open_mtm


@dataclass(frozen=True, slots=True)
class SlotRecon:
    alias: str
    realized_pnl: float                       # local DB realized (diagnostic)
    open_mtm: float
    account_value_usd: float
    positions_known: bool
    venue_realized_pnl: float | None = None   # venue realized closedPnl (Σ closedPnl-fee)
    account_pnl_all_time: float | None = None # HL portfolio equity-based PnL (matches UI)
    fills_count: int | None = None            # # of strategy (outcome-market) fills
    pnl_mismatch: bool = False                # local vs venue closedPnl diverge > tolerance
    # SHR-77: per-market-class (priceBinary/priceBucket/unknown) split of the
    # outcome PnL + fills. Populated for HL slots only (PM is binary by
    # construction); None for PM and for slots whose venue read failed.
    klass_breakdown: dict[str, "KlassStat"] | None = None
    drift: list[Drift] = field(default_factory=list)

    # --- Trailing-24h windowed fields ---
    # venue_realized_pnl_window: sum of closedPnl-fee for HL outcome fills
    # within the trailing window (None when venue fills not fetched).
    venue_realized_pnl_window: float | None = None
    # realized_pnl_window: local DB realized within the trailing window (PM uses
    # this as the primary source; HL uses it for diagnostics only).
    realized_pnl_window: float = 0.0
    # fills_count_window: number of outcome fills within the trailing window.
    fills_count_window: int | None = None
    # klass_breakdown_window: per-class split for the trailing window (HL only).
    klass_breakdown_window: dict[str, "KlassStat"] | None = None

    @property
    def total_true_pnl(self) -> float:
        """The STRATEGY's PnL: outcome-markets only. HL venue_realized_pnl is the
        outcome-only realized closedPnl (excludes the operator's perp/spot trades
        on the same account); PM realized lives in the local ledger and is already
        outcome-only. NOT the HL portfolio `allTime` PnL — that nets in unrelated
        HYPE/perp/spot activity (see account_pnl_all_time, shown for context)."""
        base = self.venue_realized_pnl if self.venue_realized_pnl is not None else self.realized_pnl
        return base + self.open_mtm

    @property
    def window_total_pnl(self) -> float:
        """Trailing-window strategy PnL: windowed realized + current open MTM.
        Uses venue_realized_pnl_window (HL) when available; falls back to
        realized_pnl_window (local, PM). Open MTM is always today's figure."""
        base = (
            self.venue_realized_pnl_window
            if self.venue_realized_pnl_window is not None
            else self.realized_pnl_window
        )
        return base + self.open_mtm

    @property
    def has_drift(self) -> bool:
        return len(self.drift) > 0 or self.pnl_mismatch


def compare_slot(
    *,
    alias: str,
    db_positions: list[tuple[str, float]],   # (symbol, qty)
    db_realized_pnl: float,
    venue: ClearinghouseState,
    qty_tolerance: float,
    venue_realized_pnl: float | None = None,
    pnl_tolerance: float = 1.0,
    account_pnl_all_time: float | None = None,
    fills_count: int | None = None,
    klass_breakdown: dict[str, KlassStat] | None = None,
    # Trailing-window fields (optional; defaults leave window fields unset)
    venue_realized_pnl_window: float | None = None,
    realized_pnl_window: float = 0.0,
    fills_count_window: int | None = None,
    klass_breakdown_window: dict[str, KlassStat] | None = None,
) -> SlotRecon:
    """Pure three-way-ish compare: DB positions vs venue positions, plus PnL.

    Open MTM = Σ venue unrealized_pnl. Position drift is skipped entirely when
    venue.positions_known is False (PM data-api flap) so an empty venue set is
    never mistaken for 'everything vanished'. When the authoritative venue
    realized PnL is supplied, a divergence from the local DB realized beyond
    ``pnl_tolerance`` is flagged as drift — this is the core Phase-0 check that
    the local ledger reconciles to venue truth.
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
            elif not math.isclose(
                v_qty, db_qty,
                rel_tol=_QTY_MISMATCH_REL_TOL, abs_tol=qty_tolerance,
            ):
                drift.append(Drift("qty_mismatch", sym, db_qty, v_qty))
        for sym, v_qty in venue_by_sym.items():
            if sym not in db_by_sym:
                drift.append(Drift("orphan", sym, 0.0, v_qty))

    pnl_mismatch = (
        venue_realized_pnl is not None
        and abs(db_realized_pnl - venue_realized_pnl) > pnl_tolerance
    )

    return SlotRecon(
        alias=alias,
        realized_pnl=db_realized_pnl,
        open_mtm=open_mtm,
        account_value_usd=venue.account_value_usd,
        positions_known=venue.positions_known,
        venue_realized_pnl=venue_realized_pnl,
        account_pnl_all_time=account_pnl_all_time,
        fills_count=fills_count,
        klass_breakdown=klass_breakdown,
        pnl_mismatch=pnl_mismatch,
        drift=drift,
        venue_realized_pnl_window=venue_realized_pnl_window,
        realized_pnl_window=realized_pnl_window,
        fills_count_window=fills_count_window,
        klass_breakdown_window=klass_breakdown_window,
    )


def _split_by_klass(
    outcome_fills, venue_positions, coin_klass: dict[str, str],
) -> dict[str, KlassStat]:
    """Group outcome fills + open positions by market class via the persisted
    coin("#N")→klass map (SHR-77). Pure: callers pass the already-filtered
    outcome fills ("#N"), the venue positions, and the map.

    realized_pnl is Σ(closedPnl-fee) of fills in the class (matching the slot's
    venue_realized_pnl); open_mtm is Σ unrealized of still-open "#N" positions in
    the class. Unmapped coins → an explicit "unknown" class."""
    realized: dict[str, float] = {}
    fills: dict[str, int] = {}
    mtm: dict[str, float] = {}
    for f in outcome_fills:
        k = coin_klass.get(f.symbol, "unknown")
        realized[k] = realized.get(k, 0.0) + (f.closed_pnl - f.fee)
        fills[k] = fills.get(k, 0) + 1
    for vp in venue_positions:
        if not vp.symbol.startswith("#"):
            continue  # non-outcome (perp/spot) leg — not a strategy market
        k = coin_klass.get(vp.symbol, "unknown")
        mtm[k] = mtm.get(k, 0.0) + vp.unrealized_pnl
    classes = set(realized) | set(mtm)
    return {
        k: KlassStat(
            realized_pnl=realized.get(k, 0.0),
            open_mtm=mtm.get(k, 0.0),
            fills=fills.get(k, 0),
        )
        for k in classes
    }


def format_report(recon: list[SlotRecon]) -> str:
    """Plain-text/HTML-safe report. One block per slot, drift called out.

    Kept dependency-free so it can be unit-tested and reused by the Telegram
    alert (TelegramClient.send escapes <, >, & itself).
    """
    lines: list[str] = ["Engine reconciliation report", ""]
    for r in recon:
        status = "DRIFT" if r.has_drift else "OK"
        lines.append(f"[{r.alias}] {status}")
        venue_str = (
            f"{r.venue_realized_pnl:+.2f}" if r.venue_realized_pnl is not None else "n/a"
        )
        lines.append(
            f"  strategy_pnl(outcome-only)={r.total_true_pnl:+.2f}  "
            f"acct_value={r.account_value_usd:.2f}"
        )
        lines.append(
            f"    venue_outcome_realized={venue_str}  local={r.realized_pnl:+.2f}  "
            f"open_mtm={r.open_mtm:+.2f}"
        )
        if r.account_pnl_all_time is not None:
            lines.append(
                f"    full_account_pnl(all-time, incl non-strategy perp/spot)="
                f"{r.account_pnl_all_time:+.2f}"
            )
        if not r.positions_known:
            lines.append("  (positions unknown — recon skipped this cycle)")
        if r.pnl_mismatch:
            lines.append(
                f"  ! pnl_mismatch: local={r.realized_pnl:+.2f} vs venue={venue_str} "
                f"(local ledger diverges from venue truth)"
            )
        for d in r.drift:
            lines.append(
                f"  ! {d.kind} {d.symbol}: db_qty={d.db_qty} venue_qty={d.venue_qty}"
            )
        lines.append("")
    return "\n".join(lines).rstrip()


def format_daily_summary(recon: list[SlotRecon], *, date_str: str | None = None) -> str:
    """Compact ONE-message daily summary for Telegram: per-strategy outcome PnL
    (both trailing-24h window and all-time total), fill count, and reconcile
    status, plus a desk total."""
    header = "📊 Desk daily report" + (f" — {date_str}" if date_str else "")
    lines = [header, ""]
    total_window = 0.0
    total_alltime = 0.0
    drifting: list[str] = []
    for r in recon:
        status = "⚠️ DRIFT" if r.has_drift else "✅"
        # Fill counts: prefer windowed/total pair; fall back to "?" when unknown.
        fills_win: int | str = r.fills_count_window if r.fills_count_window is not None else "?"
        fills_tot: int | str = r.fills_count if r.fills_count is not None else "?"
        lines.append(
            f"{r.alias}: 24h {r.window_total_pnl:+.2f} | total {r.total_true_pnl:+.2f} | "
            f"fills {fills_win}/{fills_tot} | "
            f"acct ${r.account_value_usd:.0f} | {status}"
        )
        # SHR-77: HL slots trade both binary + bucket markets on one account, so
        # split their outcome PnL + fills by class. PM slots (klass_breakdown
        # None) are binary-only and stay a single line. The split is purely
        # informational — the slot PnL above and the desk total are unchanged.
        if r.klass_breakdown or r.klass_breakdown_window:
            all_klasses = set()
            if r.klass_breakdown:
                all_klasses |= set(r.klass_breakdown)
            if r.klass_breakdown_window:
                all_klasses |= set(r.klass_breakdown_window)
            for klass in _KLASS_ORDER + tuple(
                k for k in sorted(all_klasses) if k not in _KLASS_ORDER
            ):
                if klass not in all_klasses:
                    continue
                st_tot = r.klass_breakdown.get(klass) if r.klass_breakdown else None
                st_win = r.klass_breakdown_window.get(klass) if r.klass_breakdown_window else None
                label = _KLASS_LABELS.get(klass, klass)
                win_pnl_str = f"{st_win.total_pnl:+.2f}" if st_win is not None else "n/a"
                tot_pnl_str = f"{st_tot.total_pnl:+.2f}" if st_tot is not None else "n/a"
                win_fills_str = str(st_win.fills) if st_win is not None else "?"
                tot_fills_str = str(st_tot.fills) if st_tot is not None else "?"
                lines.append(
                    f"    {label}: 24h {win_pnl_str} | total {tot_pnl_str} | "
                    f"fills {win_fills_str}/{tot_fills_str}"
                )
        total_window += r.window_total_pnl
        total_alltime += r.total_true_pnl
        if r.has_drift:
            drifting.append(r.alias)
    lines.append("")
    recon_line = "all reconciled ✅" if not drifting else f"⚠️ DRIFT: {', '.join(drifting)}"
    lines.append(
        f"Total strategy PnL: 24h {total_window:+.2f} | total {total_alltime:+.2f}  |  {recon_line}"
    )
    return "\n".join(lines)


def gather_slot(
    *, alias: str, dal: StateDAL, exec_client: ExecutionClient,
    qty_tolerance: float, pnl_tolerance: float = 1.0,
    fetch_venue_realized: bool = True,
    now_ns: int | None = None,
    window_hours: float = 24.0,
) -> SlotRecon:
    """IO: pull local + venue realized PnL + DB positions + venue state for one
    slot and run the pure compare. clearinghouse_state()/realized_pnl_since() are
    blocking read-only SDK calls; the caller offloads via asyncio.to_thread when
    needed.

    Venue realized is authoritative ONLY where the venue tracks it: HL exposes
    per-fill closedPnl via user_fills (the daily-loss gate's source). PM has no
    venue realized counter — its realized lives in our settlement/fill ledger —
    so callers pass fetch_venue_realized=False for PM and the local figure stands
    (venue_realized=None), avoiding a false local-vs-venue mismatch.

    now_ns is injected by the caller (from main's clock) for testability; if
    None it defaults to time.time_ns(). window_hours controls the look-back
    window (default 24h) for the trailing-window PnL figures.
    """
    if now_ns is None:
        now_ns = time.time_ns()
    window_start_ns = now_ns - int(window_hours * 3600 * 1_000_000_000)

    db_realized = dal.realized_pnl_since(0)
    db_positions = [(p.symbol, p.qty) for p in dal.all_positions()]
    venue = exec_client.clearinghouse_state()
    venue_realized: float | None = None
    account_pnl: float | None = None
    fills_count: int | None = None
    klass_breakdown: dict[str, KlassStat] | None = None
    venue_realized_window: float | None = None
    fills_count_window: int | None = None
    klass_breakdown_window: dict[str, KlassStat] | None = None
    if fetch_venue_realized:
        # Fetch fills ONCE and derive both realized PnL and the fill count from
        # it — calling realized_pnl_since AND user_fills separately would paginate
        # HL's (capped) fill history twice and risk a per-IP 429. outcome_only:
        # strategy = HIP-4 outcome markets ("#N"), excluding non-strategy perp/spot.
        try:
            outcome = [
                f for f in exec_client.user_fills(since_ts_ns=0)
                if f.symbol.startswith("#")
            ]
            venue_realized = sum(f.closed_pnl - f.fee for f in outcome)
            fills_count = len(outcome)
            # SHR-77: split realized PnL + fills (and open-MTM) by market class.
            # The coin("#N")→klass map is persisted at QuestionMetaEvent ingest,
            # so this is a durable join — not a description heuristic. Coins with
            # no mapped class fall into an explicit "unknown" bucket (never
            # silently folded into binary/bucket on a money report).
            klass_breakdown = _split_by_klass(
                outcome, venue.positions, dal.coin_klass_map(),
            )
            # Trailing-window: filter the already-fetched fills — no second fetch.
            # Fills carry a ts_ns attribute; filter to those within the window.
            outcome_window = [
                f for f in outcome
                if getattr(f, "ts_ns", None) is not None and f.ts_ns >= window_start_ns
            ]
            venue_realized_window = sum(f.closed_pnl - f.fee for f in outcome_window)
            fills_count_window = len(outcome_window)
            # Window klass split: open MTM is passed through (it IS today's
            # unrealized); realized/fills come from the windowed fill list.
            klass_breakdown_window = _split_by_klass(
                outcome_window, venue.positions, dal.coin_klass_map(),
            )
        except Exception:  # noqa: BLE001 — venue read is best-effort; report still useful
            venue_realized = None
        # Equity-based account PnL (HL portfolio = the UI number). Optional: only
        # HL implements it; PM clients won't have the method.
        pnl_fn = getattr(exec_client, "account_pnl_all_time", None)
        if pnl_fn is not None:
            try:
                account_pnl = pnl_fn()
            except Exception:  # noqa: BLE001
                account_pnl = None
    else:
        # PM: the local ledger is the source of truth and is already outcome-only.
        try:
            fills_count = dal.fills_count()
        except Exception:  # noqa: BLE001
            fills_count = None

    # PM windowed local realized (dal.realized_pnl_since supports a cutoff).
    realized_pnl_window: float = 0.0
    if not fetch_venue_realized:
        try:
            realized_pnl_window = dal.realized_pnl_since(window_start_ns)
        except Exception:  # noqa: BLE001
            realized_pnl_window = 0.0

    return compare_slot(
        alias=alias, db_positions=db_positions, db_realized_pnl=db_realized,
        venue=venue, qty_tolerance=qty_tolerance,
        venue_realized_pnl=venue_realized, pnl_tolerance=pnl_tolerance,
        account_pnl_all_time=account_pnl, fills_count=fills_count,
        klass_breakdown=klass_breakdown,
        venue_realized_pnl_window=venue_realized_window,
        realized_pnl_window=realized_pnl_window,
        fills_count_window=fills_count_window,
        klass_breakdown_window=klass_breakdown_window,
    )


def build_report(
    deploy_cfg, strategies_cfg, *,
    qty_tolerance: float, pnl_tolerance: float = 1.0,
    now_ns: int | None = None,
    window_hours: float = 24.0,
) -> list[SlotRecon]:
    """IO: build a read-only client + open the DAL per slot, gather each.

    A slot whose client/DAL errors yields a positions_known=False SlotRecon so
    one bad slot never aborts the whole report.

    now_ns is injected for testability; defaults to time.time_ns() in gather_slot.
    window_hours controls the trailing PnL window (default 24h)."""
    from .config import HyperliquidAccount
    from .config_builders import build_exec_client
    from .state import StateDAL

    if now_ns is None:
        now_ns = time.time_ns()

    out: list[SlotRecon] = []
    for s_cfg in strategies_cfg.strategies:
        alias = s_cfg.account_alias
        try:
            acct = deploy_cfg.accounts[alias]
            # paper_mode=False: clearinghouse_state() is a read-only venue call
            # (HL /info user_state, PM data-api positions) — NO orders. We MUST
            # hit the real venue, because in paper mode the clients return an
            # empty stub (account_value_usd=0, no positions) which would make the
            # report show false "vanished" drift and a zero account value.
            client = build_exec_client(alias, acct, paper_mode=False)
            dal = StateDAL(Path(deploy_cfg.state_db_path_for(alias)))
            # Only HL exposes an authoritative venue realized (user_fills
            # closedPnl); PM realized lives in our local settlement/fill ledger.
            fetch_venue_realized = isinstance(acct, HyperliquidAccount)
            out.append(gather_slot(alias=alias, dal=dal, exec_client=client,
                                   qty_tolerance=qty_tolerance, pnl_tolerance=pnl_tolerance,
                                   fetch_venue_realized=fetch_venue_realized,
                                   now_ns=now_ns, window_hours=window_hours))
        except Exception as e:  # noqa: BLE001 — a bad slot must not abort the report
            out.append(SlotRecon(alias=alias, realized_pnl=0.0, open_mtm=0.0,
                                 account_value_usd=0.0, positions_known=False,
                                 drift=[]))
            logger.warning("recon slot {} failed: {}", alias, e)
    return out


async def _post_tg(
    text: str, *, bot_token: str | None, chat_id: str | None,
    session_factory=None, tg_factory=None,
) -> bool:
    """Send one Telegram message. Credentials are the engine's own
    deploy.alerts.telegram (env TG_BOT_TOKEN/TG_CHAT_ID) — NOT TELEGRAM_*.
    No-op (returns False) if creds are missing. session_factory/tg_factory are
    injectable for tests."""
    if not (bot_token and chat_id):
        return False
    if session_factory is None:
        import aiohttp as _aiohttp
        session_factory = _aiohttp.ClientSession
    if tg_factory is None:
        from hlanalysis.alerts.telegram import TelegramClient
        tg_factory = TelegramClient
    async with session_factory() as session:
        return await tg_factory(bot_token=bot_token, chat_id=chat_id, session=session).send(text)


async def _maybe_alert(
    report_text: str, has_drift: bool, *,
    bot_token: str | None = None, chat_id: str | None = None,
    session_factory=None, tg_factory=None,
) -> None:
    """Send the full reconciliation report to Telegram ONLY when there is drift."""
    if not has_drift:
        return
    await _post_tg("⚠️ Reconciliation DRIFT\n\n" + report_text, bot_token=bot_token,
                   chat_id=chat_id, session_factory=session_factory, tg_factory=tg_factory)


def main() -> None:
    p = argparse.ArgumentParser(description="Out-of-band venue reconciliation report.")
    p.add_argument("--strategy-config", type=Path, default=Path("config/strategy.yaml"))
    p.add_argument("--deploy-config", type=Path, default=Path("config/deploy.yaml"))
    p.add_argument("--qty-tolerance", type=float, default=_QTY_MISMATCH_ABS_TOL,
                   help="abs share tolerance for position qty drift; defaults to the "
                        "engine reconcile's (2e-2) so benign PM data-api rounding "
                        "doesn't false-flag DRIFT")
    p.add_argument("--pnl-tolerance", type=float, default=1.0,
                   help="USD divergence between local and venue realized that counts as drift")
    p.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    p.add_argument("--summary", action="store_true",
                   help="send the compact one-message daily summary to Telegram (always)")
    p.add_argument("--date", type=str, default=None, help="date label for the summary header")
    p.add_argument("--window-hours", type=float, default=24.0,
                   help="look-back window in hours for the trailing PnL figures (default 24)")
    args = p.parse_args()

    from .config import load_deploy_config, load_strategies_config
    deploy_cfg = load_deploy_config(args.deploy_config)
    strategies_cfg = load_strategies_config(args.strategy_config)

    # Telegram creds — the engine's own (deploy.alerts.telegram = TG_BOT_TOKEN/TG_CHAT_ID).
    tg = getattr(getattr(deploy_cfg, "alerts", None), "telegram", None)
    bot_token = getattr(tg, "bot_token", None)
    chat_id = getattr(tg, "chat_id", None)

    now_ns = time.time_ns()
    recon = build_report(deploy_cfg, strategies_cfg, qty_tolerance=args.qty_tolerance,
                         pnl_tolerance=args.pnl_tolerance, now_ns=now_ns,
                         window_hours=args.window_hours)
    has_drift = any(r.has_drift for r in recon)
    text = format_report(recon)

    if args.summary:
        summary = format_daily_summary(recon, date_str=args.date)
        print(summary)
        sent = asyncio.run(_post_tg(summary, bot_token=bot_token, chat_id=chat_id))
        if not sent:
            logger.warning("daily summary NOT sent to Telegram (creds missing?)")
        raise SystemExit(1 if has_drift else 0)

    if args.json:
        print(json.dumps({
            "generated_at_ns": now_ns,
            "has_drift": has_drift,
            "slots": [
                {"alias": r.alias, "realized_pnl": r.realized_pnl,
                 "venue_realized_pnl": r.venue_realized_pnl,
                 "account_pnl_all_time": r.account_pnl_all_time,
                 "pnl_mismatch": r.pnl_mismatch,
                 "open_mtm": r.open_mtm, "total_true_pnl": r.total_true_pnl,
                 "account_value_usd": r.account_value_usd,
                 "positions_known": r.positions_known,
                 "drift": [vars(d) for d in r.drift]}
                for r in recon
            ],
        }))
    else:
        print(text)

    asyncio.run(_maybe_alert(text, has_drift, bot_token=bot_token, chat_id=chat_id))
    # Nonzero exit on drift so an SSM/cron caller can detect it.
    raise SystemExit(1 if has_drift else 0)


if __name__ == "__main__":
    main()
