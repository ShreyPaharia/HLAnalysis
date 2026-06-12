"""Mutable run state dataclass and order-routing helpers for the hftbacktest runner.

Extracted verbatim from ``hftbt_runner.py`` — no logic changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from hftbacktest import order as hb_order
from loguru import logger

from hlanalysis.marketdata.position_math import (
    PositionState,
    apply_fill,
)
from hlanalysis.marketdata.position_math import (
    stop_price as _stop_price_fn,
)
from hlanalysis.strategy.types import BookState, Position

from ..core.data_source import DataSource, QuestionDescriptor
from ..halt_replay import (
    EntryGateInputs,
    HaltWindow,
    SimRiskCaps,
    daily_window_start_ns,
    entry_veto,
)
from ._fees import _binary_fee
from ._fills import (
    _classify_reject,
    _hedge_avg_entry,
    _hedge_mtm_fill,
    _inter_order_blocked,
    _is_fleeting_ask,
    _is_fleeting_bid,
    _settle_px_for_outcome,
    _slipped_buy_price,
    _slipped_sell_price,
)
from .result import DiagnosticRow, Fill, RunResult

# ---------------------------------------------------------------------------
# Mutable run state + order-routing helpers
# ---------------------------------------------------------------------------
#
# The scan loop's enter / exit / hedge / stop-loss / settlement blocks were
# extracted into the helpers below to keep ``run_one_question`` legible. They
# all mutate one ``_RunState``; the behaviour is identical to the prior inline
# code — same submit order, same fill recording, same position math.


@dataclass(slots=True)
class _RunState:
    """Mutable state threaded through one question's scan loop + settlement.

    Carries the immutable per-run context (``hbt``, ``cfg``, ``q`` …), the
    accumulated outputs (``result``, ``fill_ts`` …), the evolving binary/hedge
    positions, and the per-tick view (``books``, ``now_ns``, ``current_diag``)
    that the routing helpers read.
    """

    hbt: Any
    cfg: Any
    q: QuestionDescriptor
    data_source: DataSource
    leg_to_asset: dict[str, int]
    hedge_asset_no: int | None
    stop_pct: float | None
    fills_dir_active: bool
    result: RunResult
    pos: Position | None = None
    hedge_positions: dict[str, Position] = field(default_factory=dict)
    last_hedge_mid: float | None = None
    fill_ts: dict[str, int] = field(default_factory=dict)
    fill_question_idx: dict[str, int] = field(default_factory=dict)
    fill_meta: dict[str, dict[str, Any]] = field(default_factory=dict)
    # SHR-79/SHR-89: per-leg timestamp (ns) of the last order DISPATCHED to the
    # venue (filled, rejected, or fleeting-vetoed — any round-trip consumed).
    # Read by the inter-order re-fire floor to throttle re-submission on a leg.
    last_order_ns: dict[str, int] = field(default_factory=dict)
    _next_oid_counter: int = 1
    # Per-tick view, refreshed each scan before routing.
    books: dict[str, BookState] = field(default_factory=dict)
    now_ns: int = 0
    current_diag: DiagnosticRow | None = None
    # SHR-85 — sim state/halt replay + daily-loss-cap + inventory caps. The gate
    # suppresses ENTRIES (exits stay exempt) the way the live RiskGate does. All
    # default to "no caps / no windows" so existing callers are unaffected.
    sim_risk_caps: SimRiskCaps | None = None
    halt_windows: tuple[HaltWindow, ...] = ()
    # Realized PnL accumulated per daily-window start (running) and its low-water
    # floor (so a window that crossed the daily-loss cap stays latched-halted even
    # if a later win recovers it — matching the live kill-switch latch).
    realized_running_by_window: dict[int, float] = field(default_factory=dict)
    realized_floor_by_window: dict[int, float] = field(default_factory=dict)
    # SHR-94: per-snapshot top-of-book arrays for IOC marketability re-check.
    # Populated from LegArrays.snap_best_ask/bid (parallel to book_ts_per_leg).
    # Empty dict (default) disables the check — synthetic / PM sources that
    # don't supply snap_best arrays fall back to the pre-SHR-94 behaviour.
    snap_best_ask_per_leg: dict[str, np.ndarray] = field(default_factory=dict)
    snap_best_bid_per_leg: dict[str, np.ndarray] = field(default_factory=dict)
    # book_ts_per_leg: per-leg snapshot timestamps, parallel to snap_best arrays.
    # Injected from run_one_question; used by _is_fleeting_ask/bid.
    book_ts_per_leg: dict[str, np.ndarray] = field(default_factory=dict)
    # Per-leg snapshot cursor (shared with stale-book gate in the scan loop).
    # Injected via the runner's book_idx dict — the _RunState doesn't own it
    # directly; it's passed to _is_fleeting_level via the routing helpers.
    _book_idx: dict[str, int] | None = None
    # SHR-91: cross-question inventory visible to the entry gate.
    # Populated by run_questions_parallel from the SharedInventoryLedger
    # (positions held in OTHER questions that were live at now_ns). Defaults to
    # zero so all existing callers are unaffected.
    extra_held_notional: float = 0.0
    extra_n_held: int = 0
    # SHR-91: timestamp when the current position opened (None = no position).
    # Used to record position_windows when a position closes.
    _pos_open_ts_ns: int | None = None

    def next_oid(self) -> int:
        oid = self._next_oid_counter
        self._next_oid_counter += 1
        return oid

    def record_realized(self, now_ns: int, amount: float) -> None:
        """Accumulate realized PnL into the daily window containing ``now_ns``.

        No-op unless a daily-loss cap is configured. Updates both the running
        total and the per-window floor (the minimum running PnL seen this window)
        which the daily-loss gate reads so the halt latches for the rest of the
        window."""
        caps = self.sim_risk_caps
        if caps is None or caps.daily_loss_cap_usd is None:
            return
        ws = daily_window_start_ns(now_ns, hour=caps.daily_window_start_hour_utc)
        running = self.realized_running_by_window.get(ws, 0.0) + amount
        self.realized_running_by_window[ws] = running
        if running < self.realized_floor_by_window.get(ws, 0.0):
            self.realized_floor_by_window[ws] = running

    def entry_blocked(
        self, *, intent_notional: float, is_topup: bool,
        held_notional: float, n_held: int,
    ) -> str | None:
        """Return a veto reason if this entry would be suppressed, else ``None``.

        Mirrors the live RiskGate entry-only caps via the shared
        ``halt_replay.entry_veto``. Exits never call this.

        SHR-91: ``extra_held_notional`` and ``extra_n_held`` (cross-question
        positions from the SharedInventoryLedger) are added to the totals so
        the concurrent / inventory caps are evaluated against the full set of
        positions that were live at ``now_ns`` across ALL questions, matching
        the live engine's single per-slot ledger."""
        caps = self.sim_risk_caps
        if caps is None and not self.halt_windows:
            return None
        caps = caps or SimRiskCaps()
        ws = daily_window_start_ns(
            self.now_ns, hour=caps.daily_window_start_hour_utc
        )
        inp = EntryGateInputs(
            now_ns=self.now_ns,
            intent_notional=intent_notional,
            held_inventory_usd=held_notional + self.extra_held_notional,
            n_held_positions=n_held + self.extra_n_held,
            is_topup=is_topup,
            realized_pnl_window=self.realized_floor_by_window.get(ws, 0.0),
        )
        return entry_veto(caps, self.halt_windows, inp)

    def record_fill_from_order(
        self, oid: int, asset_no: int, symbol: str, side: str, cloid: str,
        intent_size: float,
    ) -> Fill | None:
        order = self.hbt.orders(asset_no).get(oid)
        if order is None:
            return None
        # Accept FILLED(3), PARTIALLY_FILLED(5), and EXPIRED(2).
        # With partial_fill_exchange + IOC, a partially-filled order whose
        # remainder was cancelled gets status=EXPIRED (the IOC cancellation).
        # exec_qty > 0 confirms something actually filled; exec_qty == 0 means
        # the order was rejected/expired with no fill (e.g. no book at all).
        exec_qty = float(order.exec_qty)
        if order.status not in (2, 3, 5) or exec_qty <= 0.0:
            return None
        exec_px = float(order.exec_price)
        # Bound to [0, 1] for binary tokens.
        exec_px = max(0.0, min(1.0, exec_px))
        # Cap fill size by the strategy's intended size; apply the optional
        # book_depth_assumption cap when set (None = unlimited).
        if self.cfg.book_depth_assumption is not None:
            exec_qty = min(exec_qty, intent_size, self.cfg.book_depth_assumption)
        else:
            exec_qty = min(exec_qty, intent_size)
        fee = _binary_fee(exec_px, exec_qty, self.cfg)
        return Fill(
            cloid=cloid, symbol=symbol, side=side, price=exec_px, size=exec_qty,
            fee=fee, partial=exec_qty < intent_size,
        )

    def record_hedge_fill_from_order(
        self, oid: int, asset_no: int, symbol: str, side: str, cloid: str,
        intent_size: float,
    ) -> Fill | None:
        """Like ``record_fill_from_order`` but for the hedge leg: no [0, 1] price
        clamp (BTC perp is ~100k), uses ``cfg.hedge_fee_bps``, sets is_hedge."""
        order = self.hbt.orders(asset_no).get(oid)
        if order is None:
            return None
        if order.status not in (3, 5):
            return None
        exec_qty = float(order.exec_qty)
        if exec_qty <= 0.0:
            return None
        exec_px = float(order.exec_price)
        exec_qty = min(exec_qty, intent_size)
        fee_rate = self.cfg.hedge_fee_bps / 1e4
        fee = exec_px * exec_qty * fee_rate
        return Fill(
            cloid=cloid, symbol=symbol, side=side, price=exec_px, size=exec_qty,
            fee=fee, partial=exec_qty < intent_size, is_hedge=True,
        )


def _route_stop_loss(st: _RunState) -> None:
    """Force an exit when the held leg's bid has fallen through the stop price,
    before evaluating the strategy's intent."""
    pos = st.pos
    if pos is None or st.stop_pct is None or pos.symbol not in st.books:
        return
    held_bid = st.books[pos.symbol].bid_px
    if held_bid is None or held_bid > pos.stop_loss_price:
        return
    asset_no = st.leg_to_asset[pos.symbol]
    # SHR-79/SHR-89: a stop-loss is a safety dispatch — it is never throttled by
    # the inter-order floor (the stop must fire), but it does mark the leg's last
    # dispatch so a follow-on enter/exit respects the one-order-in-flight floor.
    st.last_order_ns[pos.symbol] = st.now_ns
    oid = st.next_oid()
    st.hbt.submit_sell_order(
        asset_no, oid, held_bid, abs(pos.qty), hb_order.IOC, hb_order.LIMIT, True,
    )
    cloid = f"stop-{oid}"
    stop_book = st.books[pos.symbol]
    px = _slipped_sell_price(stop_book, held_bid, st.cfg.slippage_bps)
    # Read the actual exec from hftbacktest; fall back to the slipped price if
    # no fill recorded (e.g. zero-bid).
    fill = st.record_fill_from_order(oid, asset_no, pos.symbol, "sell", cloid, abs(pos.qty))
    if fill is None:
        fill = Fill(
            cloid=cloid, symbol=pos.symbol, side="sell", price=px,
            size=abs(pos.qty), fee=_binary_fee(px, abs(pos.qty), st.cfg), partial=False,
        )
    st.result.fills.append(fill)
    st.fill_ts[cloid] = st.now_ns
    st.fill_question_idx[cloid] = st.q.question_idx
    # SHR-85: a stop-loss is a closing reduce — realize PnL into the daily-loss
    # window accumulator. Long-leg close realizes (sell_px - avg_entry)*size − fee.
    st.record_realized(
        st.now_ns, (fill.price - pos.avg_entry) * fill.size - fill.fee
    )
    # SHR-91: record the completed position window for the shared-inventory ledger.
    if st._pos_open_ts_ns is not None:
        notional = abs(pos.qty) * pos.avg_entry
        st.result.position_windows.append((st._pos_open_ts_ns, st.now_ns, notional))
    st._pos_open_ts_ns = None
    st.pos = None


def _route_hedge(st: _RunState, decision: Any) -> None:
    """Route hedge intents (independent of binary action / pos state)."""
    cfg = st.cfg
    if st.hedge_asset_no is None or not cfg.hedge_symbol or not decision.intents:
        return
    for h_intent in decision.intents:
        if h_intent.symbol != cfg.hedge_symbol:
            continue
        if cfg.hedge_symbol not in st.books:
            continue
        h_book = st.books[cfg.hedge_symbol]
        if h_intent.side == "buy":
            h_slipped = (
                h_book.ask_px * (1.0 + cfg.hedge_slippage_bps / 1e4)
                if h_book.ask_px is not None else 0.0
            )
            h_oid = st.next_oid()
            st.hbt.submit_buy_order(
                st.hedge_asset_no, h_oid, h_slipped, h_intent.size,
                hb_order.IOC, hb_order.LIMIT, True,
            )
            h_side = "buy"
        else:
            h_slipped = (
                h_book.bid_px * (1.0 - cfg.hedge_slippage_bps / 1e4)
                if h_book.bid_px is not None else 0.0
            )
            h_oid = st.next_oid()
            st.hbt.submit_sell_order(
                st.hedge_asset_no, h_oid, h_slipped, h_intent.size,
                hb_order.IOC, hb_order.LIMIT, True,
            )
            h_side = "sell"
        h_fill = st.record_hedge_fill_from_order(
            h_oid, st.hedge_asset_no, cfg.hedge_symbol, h_side, h_intent.cloid, h_intent.size,
        )
        if h_fill is None:
            # SHR-89: marketable hedge IOC with no fill → reject (re-fires next scan).
            if _classify_reject(h_book, h_side, h_slipped):
                st.result.n_rejects += 1
        if h_fill is not None:
            st.result.fills.append(h_fill)
            st.fill_ts[h_fill.cloid] = st.now_ns
            st.fill_question_idx[h_fill.cloid] = st.q.question_idx
            # Track hedge position independently. Qty-weighted cost basis across
            # fills (SHR-55) — avg_entry was previously clobbered with the latest
            # fill price on every top-up.
            h_pos = st.hedge_positions.get(cfg.hedge_symbol)
            prev_qty = h_pos.qty if h_pos is not None else 0.0
            prev_avg = h_pos.avg_entry if h_pos is not None else 0.0
            new_qty, new_avg = _hedge_avg_entry(
                prev_qty, prev_avg, h_fill.side, h_fill.size, h_fill.price,
            )
            st.hedge_positions[cfg.hedge_symbol] = Position(
                question_idx=st.q.question_idx, symbol=cfg.hedge_symbol, qty=new_qty,
                avg_entry=new_avg, stop_loss_price=0.0, last_update_ts_ns=st.now_ns,
            )


def _route_enter(st: _RunState, decision: Any) -> None:
    """Route an ENTER decision: open or top-up the binary position."""
    cfg = st.cfg
    q = st.q
    intent = decision.intents[0]
    # Skip hedge intents for binary position management.
    if cfg.hedge_enabled and intent.symbol == cfg.hedge_symbol:
        intent = next(
            (i for i in decision.intents if i.symbol != cfg.hedge_symbol), None
        )
    # Topup intents target the held position's symbol; reject any other.
    if intent is not None and st.pos is not None and intent.symbol != st.pos.symbol:
        intent = None
    asset_no = None
    if intent is not None:
        asset_no = st.leg_to_asset.get(intent.symbol)
    if intent is None or asset_no is None or intent.symbol not in st.books:
        return
    book = st.books[intent.symbol]
    # SHR-85 entry gate: suppress this ENTER if live would have been halted /
    # capped at this instant (halt window, daily-loss cap, inventory caps).
    # Notional uses the intent's limit price to match the live RiskGate's
    # ``intent.size * intent.limit_price``.
    is_topup = st.pos is not None and intent.symbol == st.pos.symbol
    held_notional = abs(st.pos.qty) * st.pos.avg_entry if st.pos is not None else 0.0
    n_held = 1 if st.pos is not None else 0
    veto = st.entry_blocked(
        intent_notional=intent.size * intent.limit_price,
        is_topup=is_topup, held_notional=held_notional, n_held=n_held,
    )
    if veto is not None:
        st.result.n_entries_suppressed += 1
        return
    # SHR-79/SHR-89: minimum inter-order re-fire floor. Suppress re-submission on
    # this leg until the floor has elapsed since the last dispatch (live
    # serializes one order per leg in flight). Throttled re-fires re-evaluate on
    # a later scan; the floor stamp below marks this dispatch.
    min_io_ns = int(cfg.min_inter_order_seconds * 1_000_000_000)
    if _inter_order_blocked(st.now_ns, st.last_order_ns.get(intent.symbol), min_io_ns):
        st.result.n_refire_throttled += 1
        return
    st.last_order_ns[intent.symbol] = st.now_ns
    slipped = _slipped_buy_price(book, intent.limit_price, cfg.slippage_bps)
    # SHR-94: pre-flight IOC marketability re-check at send+latency.
    # If the next recorded book snapshot falls within the latency window and
    # shows the ask above the fill price, the level is fleeting — veto the
    # submit and count a reject (strategy re-fires next scan as with live).
    if cfg.ioc_marketability_recheck and st._book_idx is not None:
        latency_ns = int(cfg.order_latency_ms * 1_000_000)
        persistence_ns = int(cfg.ioc_fleeting_persistence_seconds * 1_000_000_000)
        if _is_fleeting_ask(
            intent.symbol, slipped, st.now_ns, latency_ns,
            st.snap_best_ask_per_leg, st.book_ts_per_leg, st._book_idx,
            persistence_ns=persistence_ns,
        ):
            st.result.n_rejects += 1
            return
    # Submit IOC limit at the slipped/limit price.
    oid = st.next_oid()
    st.hbt.submit_buy_order(
        asset_no, oid, slipped, intent.size, hb_order.IOC, hb_order.LIMIT, True,
    )
    fill = st.record_fill_from_order(oid, asset_no, intent.symbol, "buy", intent.cloid, intent.size)
    if fill is None:
        # SHR-89: a marketable buy that returned no fill is a reject (book moved
        # away during latency δ). Count it; the strategy re-fires next scan.
        if _classify_reject(book, "buy", slipped):
            st.result.n_rejects += 1
        return
    st.result.fills.append(fill)
    st.fill_ts[fill.cloid] = st.now_ns
    st.fill_question_idx[fill.cloid] = q.question_idx
    if st.fills_dir_active and st.current_diag is not None and st.pos is None:
        edge_chosen = (
            st.current_diag.edge_yes
            if intent.symbol == (q.leg_symbols[0] if q.leg_symbols else "")
            else st.current_diag.edge_no
        )
        st.fill_meta[fill.cloid] = {
            "entry_p_model": st.current_diag.p_model,
            "entry_edge_chosen_side": edge_chosen,
            "entry_sigma": st.current_diag.sigma,
            "entry_tau_yr": st.current_diag.tau_yr,
        }
    # Open or topup (add-on) via the shared position math — the SAME
    # ``apply_fill`` the live router calls, so add-on weighted-average entry is
    # bit-identical to production. The stop is (re)stamped at the resulting
    # basis: the fill price on open, the new weighted average on a topup.
    prev = PositionState(st.pos.qty, st.pos.avg_entry) if st.pos is not None else None
    new_pos, _ = apply_fill(prev, "buy", fill.size, fill.price)
    assert new_pos is not None
    # SHR-91: stamp the position open time on the first ENTER (prev=None means
    # no prior position; top-ups keep the existing open timestamp).
    if prev is None:
        st._pos_open_ts_ns = st.now_ns
    st.pos = Position(
        question_idx=st.pos.question_idx if st.pos is not None else q.question_idx,
        symbol=intent.symbol, qty=new_pos.qty, avg_entry=new_pos.avg_entry,
        stop_loss_price=_stop_price(new_pos.avg_entry, st.stop_pct),
        last_update_ts_ns=st.now_ns,
    )


def _route_exit(st: _RunState, decision: Any) -> None:
    """Route an EXIT decision: reduce / close the held binary position.

    Caller guarantees ``st.pos is not None``.
    """
    cfg = st.cfg
    pos = st.pos
    intent = decision.intents[0]
    # Skip hedge intents for binary position management.
    if cfg.hedge_enabled and intent.symbol == cfg.hedge_symbol:
        intent = next(
            (i for i in decision.intents if i.symbol != cfg.hedge_symbol), None
        )
    if intent is None or st.leg_to_asset.get(intent.symbol) is None or intent.symbol not in st.books:
        return
    exit_asset_no = st.leg_to_asset[intent.symbol]
    book = st.books[intent.symbol]
    size = min(intent.size, abs(pos.qty))
    # SHR-79/SHR-89: minimum inter-order re-fire floor (see _route_enter). An
    # exit re-fire within the floor is suppressed; the position stays open and
    # the strategy re-fires on a later scan. Throttles the wide-bucket exit churn
    # (the #1670 doom-loop) to live's one-order-in-flight cadence.
    min_io_ns = int(cfg.min_inter_order_seconds * 1_000_000_000)
    if _inter_order_blocked(st.now_ns, st.last_order_ns.get(intent.symbol), min_io_ns):
        st.result.n_refire_throttled += 1
        return
    st.last_order_ns[intent.symbol] = st.now_ns
    slipped = _slipped_sell_price(book, intent.limit_price, cfg.slippage_bps)
    # SHR-94: pre-flight IOC marketability re-check at send+latency.
    # If the next recorded book snapshot shows the bid has retreated below the
    # fill price within the latency window, the level is fleeting — veto.
    if cfg.ioc_marketability_recheck and st._book_idx is not None:
        latency_ns = int(cfg.order_latency_ms * 1_000_000)
        persistence_ns = int(cfg.ioc_fleeting_persistence_seconds * 1_000_000_000)
        if _is_fleeting_bid(
            intent.symbol, slipped, st.now_ns, latency_ns,
            st.snap_best_bid_per_leg, st.book_ts_per_leg, st._book_idx,
            persistence_ns=persistence_ns,
        ):
            st.result.n_rejects += 1
            return
    oid = st.next_oid()
    st.hbt.submit_sell_order(
        exit_asset_no, oid, slipped, size, hb_order.IOC, hb_order.LIMIT, True,
    )
    fill = st.record_fill_from_order(oid, exit_asset_no, intent.symbol, "sell", intent.cloid, size)
    if fill is None:
        # SHR-89: a marketable sell that returned no fill is a reject (bid moved
        # away during latency δ). Count it; the strategy re-fires next scan with
        # the position still open.
        if _classify_reject(book, "sell", slipped):
            st.result.n_rejects += 1
        return
    st.result.fills.append(fill)
    st.fill_ts[fill.cloid] = st.now_ns
    st.fill_question_idx[fill.cloid] = st.q.question_idx
    # Reduce via the shared position math. ``close_atol=cfg.lot_size`` reproduces
    # the runner's close rule: when fill.size is capped below abs(pos.qty) by
    # book_depth_assumption, keep the residual open so settlement (or a later
    # exit) closes it — but anything below one lot is unfillable, so treat it as
    # closed to avoid an infinite-exit loop.
    new_pos, realized = apply_fill(
        PositionState(pos.qty, pos.avg_entry), fill.side, fill.size, fill.price,
        close_atol=cfg.lot_size,
    )
    # SHR-85: feed the realized PnL on this reduce into the daily-loss window
    # accumulator (net of fee, matching live closedPnl semantics).
    st.record_realized(st.now_ns, realized - fill.fee)
    if new_pos is None:
        # SHR-91: position fully closed — record window for shared-inventory ledger.
        if st._pos_open_ts_ns is not None:
            notional = abs(pos.qty) * pos.avg_entry
            st.result.position_windows.append((st._pos_open_ts_ns, st.now_ns, notional))
        st._pos_open_ts_ns = None
        st.pos = None
    else:
        st.pos = Position(
            question_idx=pos.question_idx, symbol=pos.symbol, qty=new_pos.qty,
            avg_entry=new_pos.avg_entry, stop_loss_price=pos.stop_loss_price,
            last_update_ts_ns=st.now_ns,
        )


def _settle(st: _RunState) -> None:
    """End-of-data settlement: close any open binary position at its leg payoff
    and mark any open hedge residual to the last observed hedge mid (SHR-55)."""
    q = st.q
    if st.pos is not None:
        pos = st.pos
        outcome = st.data_source.resolved_outcome(q)
        # Prefer the data source's per-leg payoff when it provides one (HL HIP-4
        # handles bucket markets here). Fall back to the binary-only
        # leg_symbols[0]/[1] lookup when the source doesn't expose it.
        leg_payoff = getattr(st.data_source, "leg_payoff", None)
        if leg_payoff is not None:
            settle_px = float(leg_payoff(q, pos.symbol))
        else:
            settle_px = _settle_px_for_outcome(pos, q, outcome)
        settle_fill = Fill(
            cloid="settle", symbol=pos.symbol,
            side="sell" if pos.qty > 0 else "buy",
            price=settle_px, size=abs(pos.qty), fee=0.0, partial=False,
        )
        st.result.fills.append(settle_fill)
        st.fill_ts["settle"] = q.end_ts_ns
        st.fill_question_idx["settle"] = pos.question_idx
        if st.fills_dir_active:
            st.fill_meta["settle"] = {"resolved_outcome": outcome}
        # SHR-91: record the settled position window for the shared-inventory ledger.
        if st._pos_open_ts_ns is not None:
            notional = abs(pos.qty) * pos.avg_entry
            st.result.position_windows.append((st._pos_open_ts_ns, q.end_ts_ns, notional))
        st._pos_open_ts_ns = None
        st.pos = None

    # Mark any open hedge residual to the last observed hedge mid and book a
    # closing fill. Without this only the OPENING hedge leg lands in
    # result.fills, so realized PnL omits the entire hedge value.
    for h_sym, h_pos in st.hedge_positions.items():
        if h_pos.qty == 0.0:
            continue
        if st.last_hedge_mid is None:
            logger.warning(
                f"hedge {h_sym} qty={h_pos.qty} held to expiry but no hedge mid "
                f"was observed; cannot mark-to-market (PnL excludes this leg)"
            )
            continue
        mtm_fill = _hedge_mtm_fill(h_sym, h_pos.qty, st.last_hedge_mid)
        st.result.fills.append(mtm_fill)
        st.fill_ts[mtm_fill.cloid] = q.end_ts_ns
        st.fill_question_idx[mtm_fill.cloid] = h_pos.question_idx
    st.hedge_positions.clear()


def _stop_price(fill_price: float, stop_pct: float | None) -> float:
    """Thin alias over the shared ``position_math.stop_price`` (single source of
    truth shared with the live router)."""
    return _stop_price_fn(fill_price, stop_pct)
