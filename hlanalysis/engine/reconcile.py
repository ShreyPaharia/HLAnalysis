from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from loguru import logger

from ..marketdata.position_math import DUST_QTY_ABS_TOL, STOP_DISABLED_SENTINEL
from .exec_types import ClearinghouseState, OpenOrderRow, UserFillRow
from .hl_client import _extract_cloid_hex32
from .risk_events import ReconcileDrift
from .state import FILL_SOURCE_ROUTER, Fill, OpenOrder, Position, StateDAL

CLOID_PREFIX = "hla-"

# Position quantities are compared across two independently-sourced floats: the
# venue's reported size (HL szi / spot balance, or the PM data-api `/positions`
# `size` truncated to 4dp) versus our fill ledger's summed fill sizes. They
# express the same holding but disagree in the low decimals — a 56.1685-share
# PM position routinely differs from our accumulated 56.16850001 by sub-share
# rounding. An exact `>1e-9` check treats that noise as a position_mismatch and
# re-fires a DRIFT alert every reconcile cycle. A real fill discrepancy (a
# missed buy/sell) is on the order of whole shares, so we only flag a mismatch
# when the quantities differ by more than rounding noise. The PM BUY size we book
# (notional/limit → 55.5444) vs the data-api `/positions` settled size (55.5523)
# routinely diverges by ~8e-3 shares — an order of magnitude above 4dp truncation
# — so abs_tol is set to 2e-2 to absorb it (still ~50x below a 1-share real
# discrepancy); rel_tol keeps it scale-safe for large lots.
_QTY_MISMATCH_REL_TOL = 1e-4
_QTY_MISMATCH_ABS_TOL = 2e-2

# Dust floor for clearing stranded PM positions — shared with the router's
# reduce_close_atol (see marketdata.position_math.DUST_QTY_ABS_TOL). PM market
# sells round the share amount to 2dp, so closing a non-round-2dp buy strands
# sub-precision dust (a 58.1279-share exit sells 58.12, leaving 0.0079). That
# residual is BOTH un-sellable (PM floors it to 0.00 → `invalid maker amount`)
# and ~$0.01 of notional. Whether the laggy data-api reports it absent OR still
# shows the stale pre-sell qty, a local position at or below this size is
# genuinely stranded-closed: clear the row so the alert-only DRIFT stops re-firing
# every reconcile cycle (incidents 2026-06-05 q729375628 venue-absent; 2026-06-06
# v31_pm venue-stale). Above the max 2dp sell residual (≤5e-3) with margin and
# ~100x below a 1-share min order, so it can never swallow a real lagging
# position (which is whole shares).
_DUST_QTY_ABS_TOL = DUST_QTY_ABS_TOL


@dataclass(frozen=True, slots=True)
class ReconcileResult:
    drift_events: list[ReconcileDrift]
    # (cloid, symbol) pairs the caller should defensively cancel. Symbol must be
    # non-empty — HL's cancelByCloid rejects empty coin and the SDK silently
    # reports outer status['ok' even when the per-cancel statuses[] failed.
    orphans_to_cancel: list[tuple[str, str]]
    # Local positions that no longer exist on the venue — almost always means
    # the HIP-4 market settled and HL auto-closed the position. The caller is
    # expected to (a) mark the question settled in MarketState so subsequent
    # stale-data checks skip the now-silent leg, and (b) publish a settlement
    # Exit event so operators see a 🏁 alert instead of a generic DRIFT.
    # The list carries the pre-delete Position snapshot so the caller can
    # populate the Exit's qty/realized_pnl without re-reading the DB.
    vanished_positions: list[tuple[int, str, Position]]
    # True when at least one position's qty drift exceeded the material threshold
    # configured on the Reconciler (material_drift_qty). The runtime uses this to
    # escalate from an alert to a halt — small noise-level drifts remain
    # alert-only, while a material discrepancy (whole-share mismatch) indicates
    # a real accounting problem that should stop new entries.
    material_qty_drift: bool = False


class Reconciler:
    """Three-way merge between local DB, HL openOrders, HL clearinghouseState.

    Spec §5.2 + §5.5. Pure synchronous logic — no IO. The caller wires in:
      - `fills_lookup(cloid)` → list[UserFillRow] for resolving local-ghost cases
      - `symbol_to_question` → optional mapping from venue symbol to question_idx
        for position-row attribution.
      - `cloid_prefix` → retained for backward compatibility but no longer used
        for matching. HL normalizes the engine's `hla-vN-<hex32>` cloid to
        `0x<hex32>`, so the join is now done on the 32-hex tail
        (via `_extract_cloid_hex32`). Per-account isolation comes from
        `info.open_orders(account_address)` being address-scoped on the venue
        side and from the per-account `state.db` on the local side.
    """

    # Conservative default: a 1-share absolute qty difference is material.
    # Positions under _QTY_MISMATCH_ABS_TOL (2e-2) are already filtered as
    # noise by the mismatch check, so the effective material window is
    # (2e-2, 1.0) = rounding noise vs. real discrepancy.
    _DEFAULT_MATERIAL_DRIFT_QTY: float = 1.0

    def __init__(
        self,
        dal: StateDAL,
        *,
        fills_lookup: Callable[[str], list[UserFillRow]],
        symbol_to_question: dict[str, int] | None = None,
        cloid_prefix: str = CLOID_PREFIX,
        account_alias: str = "",
        apply_position_changes: bool = True,
        venue_fill_source: str = FILL_SOURCE_ROUTER,
        settled_qidxs: frozenset[int] | set[int] | None = None,
        journal=None,
        material_drift_qty: float | None = None,
    ) -> None:
        self.dal = dal
        # Optional trade journal (SHR-83). When a synchronous ACK carried no
        # fill but user_fills later reveals the order filled, stamp the journal
        # row's fill_ts/px/sz here so the late-fill latency is captured too.
        # Best-effort (TradeJournal swallows its own errors); None elsewhere.
        self.journal = journal
        self.fills_lookup = fills_lookup
        self.symbol_to_question = symbol_to_question or {}
        self.cloid_prefix = cloid_prefix
        # Provenance stamped on fills replayed from venue user_fills here
        # (lost-ACK recovery). For HL pass FILL_SOURCE_VENUE so these tid-keyed
        # rows agree with the reconcile loop's full venue mirror — otherwise a
        # 'router'-stamped tid would block the mirror's 'venue' write (append_fill
        # dedups by fill_id) and realized_pnl_since's venue-preference would drop
        # it. PM keeps the default 'router' (its local ledger is authoritative).
        self.venue_fill_source = venue_fill_source
        # When False, the positions pass DETECTS venue/local mismatches and
        # emits drift alerts but does NOT mutate the DB (no vanish, no
        # qty-overwrite, no adopt). Used for the PM live loop: PM's only
        # position-truth source is the data-api indexer, which lags our own
        # fills by seconds and flaps — so live we trust our fill ledger and only
        # alert on differences. Venue truth is APPLIED at restart (the
        # RestartDriftGate, to recover fills missed while down) and continuously
        # for HL (clearinghouseState is instant + authoritative). PM positions
        # close via our sell fills + the endDate-gated gamma SettlementEvent.
        self.apply_position_changes = apply_position_changes
        # Stamped onto every ReconcileDrift this Reconciler emits so alerts
        # carry the account that detected the drift.
        self.account_alias = account_alias
        # Question idxs the engine knows have settled. During the
        # settlement→redemption gap a PM market keeps reporting the winning
        # shares on the venue (~15 min until Polymarket auto-redeems them
        # on-chain) while the engine's local row is already gone — that's
        # expected, not position drift. We suppress the venue_orphan /
        # venue_absent alert-only DRIFT for these qidxs (and never adopt them:
        # adopting a position about to be redeemed resurrects an un-sellable
        # phantom). A genuinely-stuck redemption is caught by the 6h
        # RedemptionTimeout watchdog, not this per-cycle alert (incident
        # 2026-06-12 q700064348 flooded ~240/h through the gap).
        self._settled_qidxs: frozenset[int] = frozenset(settled_qidxs or ())
        # Absolute qty diff above which a position mismatch is escalated to
        # material (ReconcileResult.material_qty_drift=True) so the runtime can
        # halt rather than just alert. Below this level the discrepancy is
        # treated as noise (rounding, sub-precision PM dust). Defaults to 1
        # share — a whole-share discrepancy is unambiguously a real problem.
        self.material_drift_qty: float = (
            material_drift_qty if material_drift_qty is not None else self._DEFAULT_MATERIAL_DRIFT_QTY
        )

    def run(
        self,
        *,
        venue_open: list[OpenOrderRow],
        venue_state: ClearinghouseState,
        now_ns: int,
    ) -> ReconcileResult:
        drift: list[ReconcileDrift] = []
        orphans: list[tuple[str, str]] = []
        vanished: list[tuple[int, str, Position]] = []
        material_qty_drift: bool = False

        # --- orders ---
        # Match local↔venue by the 32-hex tail of the cloid. HL normalizes
        # the engine's `hla-v1-<hex32>` form to `0x<hex32>` on its side, so a
        # literal prefix filter would exclude every venue order. Normalizing
        # both sides to the bare hex makes the join work regardless of which
        # form HL hands back. The per-account isolation that the legacy
        # cloid_prefix filter was guarding is already provided by
        # info.open_orders(account_address) being address-scoped (and by the
        # per-account state.db on the local side).
        def _hex(c: str) -> str:
            return _extract_cloid_hex32(c)

        venue_by_hex = {_hex(r.cloid): r for r in venue_open}
        local_live = list(self.dal.live_orders())
        local_by_hex = {_hex(o.cloid): o for o in local_live}

        # local-ghost: in DB live, not on venue
        for cloid_hex, db_o in local_by_hex.items():
            if cloid_hex in venue_by_hex:
                continue
            # We pass the LOCAL cloid (the internal form) to fills_lookup so
            # downstream filters that key off the engine's cloid work; HL
            # returns its normalized form on the rows themselves and the
            # caller compares hex tails when matching.
            cloid = db_o.cloid
            fills = self.fills_lookup(cloid)
            if fills:
                # Mark filled, replay fills into DB
                for f in fills:
                    self.dal.append_fill(
                        Fill(
                            fill_id=f.fill_id,
                            cloid=cloid,
                            question_idx=db_o.question_idx,
                            symbol=db_o.symbol,
                            side=f.side,
                            price=f.price,
                            size=f.size,
                            fee=f.fee,
                            ts_ns=f.ts_ns,
                            closed_pnl=f.closed_pnl,
                            source=self.venue_fill_source,
                        )
                    )
                self.dal.update_order_status(cloid, status="filled", now_ns=now_ns)
                # SHR-83: record the late-discovered fill on the journal row the
                # router opened at decision time (latest fill = the resolving one).
                if self.journal is not None:
                    last = fills[-1]
                    self.journal.record_fill(
                        cloid=cloid,
                        fill_ts_ns=last.ts_ns,
                        fill_px=last.price,
                        fill_sz=last.size,
                    )
                # Diagnostic (incident 2026-06-04, #1 root-cause suspect): this
                # path replays the Fill rows + marks the order filled but does
                # NOT apply the fills to the position table. If the router never
                # booked this fill (PM ack returned no fill info), the position
                # stays open by `net_delta` forever → endless re-exit loop. Log
                # the unapplied delta so the next occurrence is unambiguous.
                net_delta = sum((f.size if f.side == "buy" else -f.size) for f in fills)
                logger.info(
                    "reconcile_fill_discovered cloid={} qidx={} symbol={} n_fills={} net_delta={:g}",
                    cloid,
                    db_o.question_idx,
                    db_o.symbol,
                    len(fills),
                    net_delta,
                )
                # SHR-46: apply the venue-confirmed position so the re-exit loop
                # can't re-fire. Use the venue's reported qty/avg_entry
                # (authoritative), keyed off db_o.question_idx which is always
                # available — unlike symbol_to_question which may be absent.
                # Only applies when apply_position_changes=True (HL). PM stays
                # alert-only: position truth there comes from the fill ledger +
                # the endDate gamma settlement, not from reconcile injection.
                if self.apply_position_changes:
                    vp = next(
                        (p for p in venue_state.positions if p.symbol == db_o.symbol),
                        None,
                    )
                    if vp is not None and abs(vp.qty) > 1e-9:
                        # Adopt the venue's authoritative qty/avg, but PRESERVE
                        # any prior local realized_pnl (accumulated by earlier
                        # partial reduces) and stop_loss_price — a fresh upsert
                        # would erase both, under-reporting PnL on the eventual
                        # close and stripping stop protection from a still-open
                        # position. Mirrors the adopt-on-mismatch path below.
                        existing = self.dal.get_position(db_o.question_idx)
                        self.dal.upsert_position(
                            Position(
                                question_idx=db_o.question_idx,
                                symbol=db_o.symbol,
                                qty=vp.qty,
                                avg_entry=vp.avg_entry,
                                realized_pnl=existing.realized_pnl if existing else 0.0,
                                last_update_ts_ns=now_ns,
                                stop_loss_price=(existing.stop_loss_price if existing else STOP_DISABLED_SENTINEL),
                            )
                        )
                drift.append(
                    ReconcileDrift(
                        ts_ns=now_ns,
                        account_alias=self.account_alias,
                        case="state_mismatch",
                        cloid=cloid,
                        question_idx=db_o.question_idx,
                        detail={"resolution": "filled_via_user_fills"},
                    )
                )
            else:
                self.dal.update_order_status(cloid, status="cancelled", now_ns=now_ns)
                drift.append(
                    ReconcileDrift(
                        ts_ns=now_ns,
                        account_alias=self.account_alias,
                        case="local_ghost",
                        cloid=cloid,
                    )
                )

        # venue-orphan: on venue, not in DB live. Orphan cloid is reported in
        # the venue's form (HL hex) — that's what cancel() needs to address it.
        for cloid_hex, vo in venue_by_hex.items():
            if cloid_hex in local_by_hex:
                continue
            orphans.append((vo.cloid, vo.symbol))
            drift.append(
                ReconcileDrift(
                    ts_ns=now_ns,
                    account_alias=self.account_alias,
                    case="venue_orphan",
                    cloid=vo.cloid,
                    detail={"symbol": vo.symbol},
                )
            )

        # state-mismatch: both, fields differ — HL wins. Keyed by hex tail; we
        # carry the local cloid through to upsert_order so the DB row keeps its
        # internal identity.
        for cloid_hex, vo in venue_by_hex.items():
            db_o = local_by_hex.get(cloid_hex)
            if db_o is None:
                continue
            if (
                abs(db_o.price - vo.price) > 1e-9
                or abs(db_o.size - vo.size) > 1e-9
                or db_o.symbol != vo.symbol
                or db_o.venue_oid != vo.venue_oid
            ):
                self.dal.upsert_order(
                    OpenOrder(
                        cloid=db_o.cloid,
                        venue_oid=vo.venue_oid,
                        question_idx=db_o.question_idx,
                        symbol=vo.symbol,
                        side=db_o.side,
                        price=vo.price,
                        size=vo.size,
                        status="open",
                        placed_ts_ns=db_o.placed_ts_ns,
                        last_update_ts_ns=now_ns,
                        strategy_id=db_o.strategy_id,
                    )
                )
                drift.append(
                    ReconcileDrift(
                        ts_ns=now_ns,
                        account_alias=self.account_alias,
                        case="state_mismatch",
                        cloid=db_o.cloid,
                        detail={"hl_price": f"{vo.price}", "db_price": f"{db_o.price}"},
                    )
                )

        # --- positions ---
        # When the venue position set couldn't be fetched (PM data-api error),
        # we have no truth to reconcile against. Skipping is critical: treating
        # an unknown set as empty would vanish-delete every live position.
        if not venue_state.positions_known:
            return ReconcileResult(
                drift_events=drift,
                orphans_to_cancel=orphans,
                vanished_positions=vanished,
            )

        apply = self.apply_position_changes
        venue_by_symbol = {p.symbol: p for p in venue_state.positions}
        local_by_qidx = {p.question_idx: p for p in self.dal.all_positions()}

        for qidx, lp in local_by_qidx.items():
            vp = venue_by_symbol.get(lp.symbol)
            # Stranded sub-precision dust (alert-only / PM). The PM 2dp sell-floor
            # leaves an un-sellable residual (≤0.0079 sh, ≤$0.01). Whether the
            # laggy data-api reports it absent OR still shows the stale pre-sell
            # qty, our own ledger says we're ~flat and the dust can't be traded —
            # so clear the row to stop the alert-only DRIFT re-firing every cycle
            # (2026-06-05 venue-absent q729375628; 2026-06-06 v31_pm venue-stale).
            # No vanished_positions entry / settlement Exit: the round-trip PnL is
            # already booked on the sell fill's closed_pnl, and we must NOT adopt
            # the stale venue qty (that resurrects a closed position → 2026-06-02
            # double-exit). Apply mode (HL) is venue-authoritative, so dust there
            # falls through to the vanish/adopt paths below.
            if not apply and abs(lp.qty) <= _DUST_QTY_ABS_TOL:
                self.dal.delete_position(qidx)
                drift.append(
                    ReconcileDrift(
                        ts_ns=now_ns,
                        account_alias=self.account_alias,
                        case="position_mismatch",
                        question_idx=qidx,
                        detail={"resolution": "dust_cleared", "symbol": lp.symbol, "db_qty": f"{lp.qty}"},
                    )
                )
                continue
            if vp is None or abs(vp.qty) < 1e-9:
                # Position absent on venue (or present at qty=0). On HL this is
                # a HIP-4 settlement auto-close: surface it as a
                # vanished_positions entry so the runtime publishes a settlement
                # Exit + marks the question settled (suppressing stale_data_halt
                # on the now-silent book). In alert-only mode (PM live) venue
                # absence is NOT trusted as a close — the data-api drops
                # positions transiently / lags our own sell fills — so we only
                # emit an informational drift and leave local state to the fill
                # ledger + the endDate gamma settlement. (Dust is handled above.)
                if apply:
                    vanished.append((qidx, lp.symbol, lp))
                    self.dal.delete_position(qidx)
                elif qidx not in self._settled_qidxs:
                    drift.append(
                        ReconcileDrift(
                            ts_ns=now_ns,
                            account_alias=self.account_alias,
                            case="position_mismatch",
                            question_idx=qidx,
                            detail={
                                "resolution": "venue_absent_alert_only",
                                "symbol": lp.symbol,
                                "db_qty": f"{lp.qty}",
                            },
                        )
                    )
                continue
            qty_diff = not math.isclose(
                vp.qty,
                lp.qty,
                rel_tol=_QTY_MISMATCH_REL_TOL,
                abs_tol=_QTY_MISMATCH_ABS_TOL,
            )
            avg_diff = abs(vp.avg_entry - lp.avg_entry) > 1e-9
            if qty_diff or avg_diff:
                # In apply mode the venue wins (HL is authoritative): adopt its
                # qty/avg. In alert-only mode we KEEP local (our fill ledger is
                # truth) and only alert — overwriting with a stale data-api qty
                # reverts our own fills (2026-06-02 double-exit).
                if apply:
                    # avg_entry is display-only (the daily-loss gate reads PnL
                    # from HL directly); qty is load-bearing (risk caps +
                    # have_position), so qty drift surfaces as an alert.
                    self.dal.upsert_position(
                        Position(
                            question_idx=qidx,
                            symbol=lp.symbol,
                            qty=vp.qty,
                            avg_entry=vp.avg_entry,
                            realized_pnl=lp.realized_pnl,
                            last_update_ts_ns=now_ns,
                            stop_loss_price=lp.stop_loss_price,
                        )
                    )
                if qty_diff:
                    abs_diff = abs(vp.qty - lp.qty)
                    if abs_diff >= self.material_drift_qty:
                        # Whole-share discrepancy: set the material flag so the
                        # runtime can escalate from alert → halt (finding #26).
                        material_qty_drift = True
                    detail = {"hl_qty": f"{vp.qty}", "db_qty": f"{lp.qty}"}
                    if not apply:
                        detail["resolution"] = "qty_mismatch_alert_only"
                    drift.append(
                        ReconcileDrift(
                            ts_ns=now_ns,
                            account_alias=self.account_alias,
                            case="position_mismatch",
                            question_idx=qidx,
                            detail=detail,
                        )
                    )

        # Position on venue we don't track locally. In apply mode (restart / HL)
        # adopt it into the DB so risk caps + the strategy's have_position HOLD
        # branch see it (without adoption the strategy re-fires entries). In
        # alert-only mode (PM live) we don't adopt from the laggy data-api —
        # re-adopting a just-closed position would resurrect it — so we only
        # alert. Stop-loss is disabled on adopted rows (no entry context).
        for sym, vp in venue_by_symbol.items():
            qidx = self.symbol_to_question.get(sym)
            if qidx is None:
                # No question mapping yet (e.g. meta event not ingested). We
                # cannot invent a question_idx — it's the position table's
                # primary key, and collisions would corrupt accounting. Emit
                # an unattributed drift event so the orphan is visible.
                drift.append(
                    ReconcileDrift(
                        ts_ns=now_ns,
                        account_alias=self.account_alias,
                        case="position_mismatch",
                        question_idx=0,
                        detail={"resolution": "venue_orphan_unattributed", "symbol": sym, "qty": f"{vp.qty}"},
                    )
                )
                continue
            if qidx in local_by_qidx:
                continue
            if qidx in self._settled_qidxs:
                # Settled, pending Polymarket auto-redemption — the venue keeps
                # showing the winning shares until the on-chain redeem (~15 min
                # post-settle) clears them. Not drift, and must NOT be adopted
                # (resurrecting a position about to be redeemed strands an
                # un-sellable phantom). See _settled_qidxs note.
                continue
            if apply:
                self.dal.upsert_position(
                    Position(
                        question_idx=qidx,
                        symbol=sym,
                        qty=vp.qty,
                        avg_entry=vp.avg_entry,
                        realized_pnl=0.0,
                        last_update_ts_ns=now_ns,
                        stop_loss_price=STOP_DISABLED_SENTINEL,
                    )
                )
            drift.append(
                ReconcileDrift(
                    ts_ns=now_ns,
                    account_alias=self.account_alias,
                    case="position_mismatch",
                    question_idx=qidx,
                    detail={
                        "resolution": "adopted_venue_orphan" if apply else "venue_orphan_alert_only",
                        "symbol": sym,
                        "qty": f"{vp.qty}",
                        "avg_entry": f"{vp.avg_entry}",
                    },
                )
            )

        return ReconcileResult(
            drift_events=drift,
            orphans_to_cancel=orphans,
            vanished_positions=vanished,
            material_qty_drift=material_qty_drift,
        )
