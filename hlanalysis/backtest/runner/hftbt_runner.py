"""Per-question runner driven by hftbacktest's HashMapMarketDepthBacktest.

The runner is strategy-agnostic and source-agnostic. Inputs:

- ``Strategy``  — pure policy; see ``hlanalysis.strategy.base.Strategy``
- ``DataSource`` — yields ``MarketEvent``s for a single ``QuestionDescriptor``
- ``QuestionDescriptor`` — identifies the question + leg symbols
- ``RunConfig``  — tick/lot size, scan cadence, fee/slippage knobs

Algorithm:

1. Collect every ``MarketEvent`` for the question. Partition:
   - per-leg book + trade events  → numpy event arrays (one per leg) fed to
     hftbacktest as a ``BacktestAsset``;
   - reference events             → kept in a list and drained into the
     runner's ``MarketState`` at each scan tick (these don't drive
     hftbacktest's depth);
   - settlement events            → not strictly required; ``resolved_outcome``
     is queried from the data source at end-of-data.
2. Build one ``HashMapMarketDepthBacktest([asset0, asset1, ...])``.
3. Loop: ``hbt.elapse(scan_interval_ns)``. After each step:
   - drain reference events up to ``hbt.current_timestamp`` into ``MarketState``;
   - build a ``QuestionView``, per-leg ``BookState``, ``recent_returns``;
   - call ``strategy.evaluate(...)``;
   - submit IOC ENTER / EXIT orders against the asset's depth;
   - record fill from ``orders(asset_no).get(oid)``.
4. End-of-data: settle any open position at 1.0 if its leg won, 0.0 otherwise.

API choice rationale: hftbacktest 2.4.4 exposes both ``HashMapMarketDepthBacktest``
(per-tick hashmap depth) and ``ROIVectorMarketDepthBacktest`` (region-of-interest
vector depth). For sparse prediction-market books and a per-second scan cadence,
the hashmap variant is the right fit — no ROI bounds to configure.

Execution model (SHR-79/56/57): HL HIP-4 assets use ``partial_fill_exchange``
(fills walk real recorded book depth; IOC remainder cancels) and
``constant_order_latency`` wired to ``RunConfig.order_latency_ms`` (default 50 ms,
empirical HL median). ``slippage_bps`` is an additive haircut applied to the
recorded fill price (buys fill higher, sells lower). HL fee = 0 empirically;
``fee_taker`` defaults to 0.0 and is logged at run start. PM / synthetic paths
are unaffected — they still build with identical API calls (only the exchange
model and latency changed, which was already the right trade-off for both).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

import hftbacktest as hb
from hftbacktest import order as hb_order
from hftbacktest.types import (
    BUY_EVENT,
    DEPTH_CLEAR_EVENT,
    EXCH_EVENT,
    LOCAL_EVENT,
    SELL_EVENT,
    event_dtype,
)

from hlanalysis.marketdata.position_math import (
    STOP_DISABLED_SENTINEL,
    PositionState,
    apply_fill,
    stop_price,
)
from hlanalysis.strategy.base import Strategy
from hlanalysis.strategy.types import Action, BookState, Position

from ..core.data_source import DataSource, QuestionDescriptor
from ..halt_replay import (
    EntryGateInputs,
    HaltWindow,
    SimRiskCaps,
    daily_window_start_ns,
    entry_veto,
)
from ..core.events import BookSnapshot, ReferenceEvent, SettlementEvent, TradeEvent
from ..core.question import build_question_view
from ..data._fastpath_core import build_leg_event_array_from_snapshots
from .market_state import MarketState
from .result import (
    DiagnosticRow,
    Fill,
    FillRow,
    RunResult,
    build_diagnostic_row,
    write_diagnostics,
    write_fills,
)


@dataclass(frozen=True, slots=True)
class RunConfig:
    scanner_interval_seconds: int = 60
    tick_size: float = 0.001
    lot_size: float = 1.0
    slippage_bps: float = 5.0
    fee_taker: float = 0.0
    # book_depth_assumption: optional explicit fill-size cap (SHR-79).
    # None (default) = unlimited — the real recorded book governs fill size via
    # partial_fill_exchange. A finite value caps exec_qty at that level as a
    # worst-case guard (e.g. --depth 500 for a $500 position limit). The
    # legacy default of 10_000 is intentionally removed so thin HIP-4 books
    # produce realistic partial fills instead of uniform full-cap fills.
    book_depth_assumption: float | None = None
    vol_lookback_seconds: int = 86_400
    last_trades_capacity: int = 256
    # Binary-leg fee model. "flat" → fee = px * qty * fee_taker (legacy; HL,
    # synthetic, anywhere a constant per-notional rate is wanted). "pm_binary"
    # → fee = qty * fee_rate * px * (1-px) — Polymarket's published curve
    # (docs.polymarket.com/trading/fees: fee = C · feeRate · p · (1-p)). The
    # PM curve peaks at $1.75/100 shares at p=0.5 (feeRate=0.07, crypto) and
    # collapses in the tails where the near-resolution strategies actually
    # trade. fee_taker is ignored when model != "flat".
    fee_model: str = "flat"
    fee_rate: float = 0.07
    # Order latency (SHR-79): constant round-trip to/from the exchange.
    # Empirical HL HIP-4 median ≈ 46 ms; default 50 ms. Set to 0 for the
    # legacy zero-latency behaviour (e.g. tests that don't care about timing).
    order_latency_ms: float = 50.0
    # Hedge leg config (used by v5_delta_hedged; ignored by all other strategies)
    hedge_enabled: bool = False
    hedge_symbol: str = ""
    hedge_tick_size: float = 0.1
    hedge_lot_size: float = 0.001
    hedge_slippage_bps: float = 10.0
    hedge_fee_bps: float = 1.0


def _binary_fee(px: float, qty: float, cfg: RunConfig) -> float:
    """Per-fill fee for binary tokens. Branches on cfg.fee_model.

    - "flat":      px * qty * fee_taker          (legacy; HL & synthetic)
    - "pm_binary": qty * fee_rate * p * (1-p)    (Polymarket docs formula)
    """
    if cfg.fee_model == "pm_binary":
        p = max(0.0, min(1.0, px))
        return qty * cfg.fee_rate * p * (1.0 - p)
    return px * qty * cfg.fee_taker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Re-exported from the shared position-math module so the runner and the live
# router/reconcile share ONE sentinel definition (no copy to drift).
_STOP_DISABLED_SENTINEL = STOP_DISABLED_SENTINEL


def _strategy_stop_loss_pct(strategy: Strategy) -> float | None:
    """Pull the legacy ``cfg.stop_loss_pct`` sim convention if present.

    A value of ``None`` (or a sentinel float ≥1e8 used by v1) disables the stop.
    """
    raw = getattr(getattr(strategy, "cfg", None), "stop_loss_pct", None)
    if raw is None:
        return None
    if float(raw) >= 1e8:
        return None
    return float(raw)


def _stop_price(fill_price: float, stop_pct: float | None) -> float:
    """Thin alias over the shared ``position_math.stop_price`` (single source of
    truth shared with the live router)."""
    return stop_price(fill_price, stop_pct)


def _initial_clear_array(start_ts_ns: int) -> np.ndarray:
    """Two depth-clear events at the start so the engine begins with an empty book.

    Without this an asset's first elapse can leave book sides in an undefined
    initial state. Cheap (2 events) and clearer than relying on the first
    snapshot to overwrite ghost levels.
    """
    arr = np.zeros(2, dtype=event_dtype)
    flag = EXCH_EVENT | LOCAL_EVENT
    arr[0]["ev"] = DEPTH_CLEAR_EVENT | flag | BUY_EVENT
    arr[0]["exch_ts"] = start_ts_ns
    arr[0]["local_ts"] = start_ts_ns
    arr[1]["ev"] = DEPTH_CLEAR_EVENT | flag | SELL_EVENT
    arr[1]["exch_ts"] = start_ts_ns
    arr[1]["local_ts"] = start_ts_ns
    return arr


def _build_asset(arr: np.ndarray, cfg: RunConfig):
    # SHR-79: partial_fill_exchange replaces no_partial_fill_exchange so IOC
    # orders fill only real recorded book depth and walk levels naturally.
    # SHR-79: order_latency_ms wires the empirical ~50 ms HL RTT into the sim.
    latency_ns = int(cfg.order_latency_ms * 1_000_000)
    return (
        hb.BacktestAsset()
        .data(arr)
        .linear_asset(1.0)
        .constant_order_latency(latency_ns, latency_ns)
        .risk_adverse_queue_model()
        .partial_fill_exchange()
        .trading_value_fee_model(0.0, 0.0)
        .tick_size(cfg.tick_size)
        .lot_size(cfg.lot_size)
        .last_trades_capacity(cfg.last_trades_capacity)
    )


def _build_hedge_asset(arr: np.ndarray, cfg: RunConfig):
    """Like _build_asset but uses hedge-specific tick/lot sizes."""
    latency_ns = int(cfg.order_latency_ms * 1_000_000)
    return (
        hb.BacktestAsset()
        .data(arr)
        .linear_asset(1.0)
        .constant_order_latency(latency_ns, latency_ns)
        .risk_adverse_queue_model()
        .partial_fill_exchange()
        .trading_value_fee_model(0.0, 0.0)
        .tick_size(cfg.hedge_tick_size)
        .lot_size(cfg.hedge_lot_size)
        .last_trades_capacity(cfg.last_trades_capacity)
    )


def _book_state(
    hbt: Any, asset_no: int, symbol: str, *, last_l2_ts_ns: int = 0
) -> BookState | None:
    depth = hbt.depth(asset_no)
    bid = float(depth.best_bid)
    ask = float(depth.best_ask)
    bid_sz = float(depth.best_bid_qty)
    ask_sz = float(depth.best_ask_qty)
    has_bid = bid > 0.0
    has_ask = ask > 0.0
    if not (has_bid or has_ask):
        return None
    return BookState(
        symbol=symbol,
        bid_px=bid if has_bid else None,
        bid_sz=bid_sz if has_bid else None,
        ask_px=ask if has_ask else None,
        ask_sz=ask_sz if has_ask else None,
        last_trade_ts_ns=0,
        last_l2_ts_ns=last_l2_ts_ns,
    )


def _slipped_buy_price(book: BookState, limit_price: float, slippage_bps: float) -> float:
    if book.ask_px is None:
        return 0.0
    slipped = book.ask_px * (1.0 + slippage_bps / 1e4)
    px = min(slipped, limit_price) if limit_price > 0 else slipped
    return max(0.0, min(1.0, px))


def _slipped_sell_price(book: BookState, limit_price: float, slippage_bps: float) -> float:
    if book.bid_px is None:
        return 0.0
    slipped = book.bid_px * (1.0 - slippage_bps / 1e4)
    px = max(slipped, limit_price) if limit_price > 0 else slipped
    return max(0.0, min(1.0, px))


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
    cfg: RunConfig
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
    _next_oid_counter: int = 1
    # Per-tick view, refreshed each scan before routing.
    books: dict[str, BookState] = field(default_factory=dict)
    now_ns: int = 0
    current_diag: "DiagnosticRow | None" = None
    # SHR-85 — sim state/halt replay + daily-loss-cap + inventory caps. The gate
    # suppresses ENTRIES (exits stay exempt) the way the live RiskGate does. All
    # default to "no caps / no windows" so existing callers are unaffected.
    sim_risk_caps: "SimRiskCaps | None" = None
    halt_windows: tuple[HaltWindow, ...] = ()
    # Realized PnL accumulated per daily-window start (running) and its low-water
    # floor (so a window that crossed the daily-loss cap stays latched-halted even
    # if a later win recovers it — matching the live kill-switch latch).
    realized_running_by_window: dict[int, float] = field(default_factory=dict)
    realized_floor_by_window: dict[int, float] = field(default_factory=dict)

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
        ``halt_replay.entry_veto``. Exits never call this."""
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
            held_inventory_usd=held_notional,
            n_held_positions=n_held,
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
    slipped = _slipped_buy_price(book, intent.limit_price, cfg.slippage_bps)
    # Submit IOC limit at the slipped/limit price.
    oid = st.next_oid()
    st.hbt.submit_buy_order(
        asset_no, oid, slipped, intent.size, hb_order.IOC, hb_order.LIMIT, True,
    )
    fill = st.record_fill_from_order(oid, asset_no, intent.symbol, "buy", intent.cloid, intent.size)
    if fill is None:
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
    slipped = _slipped_sell_price(book, intent.limit_price, cfg.slippage_bps)
    oid = st.next_oid()
    st.hbt.submit_sell_order(
        exit_asset_no, oid, slipped, size, hb_order.IOC, hb_order.LIMIT, True,
    )
    fill = st.record_fill_from_order(oid, exit_asset_no, intent.symbol, "sell", intent.cloid, size)
    if fill is None:
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


# ---------------------------------------------------------------------------
# Per-question runner
# ---------------------------------------------------------------------------


def run_one_question(
    strategy: Strategy,
    data_source: DataSource,
    q: QuestionDescriptor,
    cfg: RunConfig,
    *,
    diagnostics_dir: Path | None = None,
    fills_dir: Path | None = None,
    strike: float = 0.0,
    hedge_events: list[BookSnapshot] | None = None,
    halt_windows: list[HaltWindow] | None = None,
    sim_risk_caps: SimRiskCaps | None = None,
) -> RunResult:
    """Run one question end-to-end through hftbacktest.

    SHR-85: ``halt_windows`` (live halt periods to replay) and ``sim_risk_caps``
    (daily-loss / inventory caps) make the sim stop ENTERING when live would have
    stopped. Both default to off, so callers that don't pass them are unaffected;
    exits are never suppressed (matching live)."""

    # SHR-57: log effective fee model/rate once at run start so it is never a
    # silent default. HL HIP-4 empirically has fee=0 across 767 live fills;
    # the 'flat' model with fee_taker=0.0 is the correct default. PM uses
    # 'pm_binary' with fee_rate=0.07; that path is unchanged.
    if cfg.fee_model == "pm_binary":
        logger.debug(
            f"fee_model=pm_binary  fee_rate={cfg.fee_rate:.4f}  "
            f"(Polymarket curve: fee = qty * {cfg.fee_rate} * p * (1-p))"
        )
    else:
        logger.debug(
            f"fee_model={cfg.fee_model}  fee_taker={cfg.fee_taker:.6f}  "
            f"(HL HIP-4 empirical fee = $0 across 767 live fills)"
        )

    # --- Collect events ----------------------------------------------------
    # Two paths:
    #   1. Fast path: the data source provides ``events_arrays(q)`` returning
    #      pre-built numpy ``event_dtype`` arrays per leg + reference/settlement
    #      event lists. This skips the ``BookSnapshot``/``TradeEvent`` dataclass
    #      round-trip and lets the data source build the hftbacktest event
    #      array fully vectorised. HL HIP-4 implements this.
    #   2. Legacy path: iterate ``events(q)`` and build per-leg arrays here via
    #      ``build_leg_event_array_from_snapshots`` (the same shared assembler,
    #      fed by an in-memory snapshot→column adapter). Synthetic / pm_nba /
    #      binance_perp sources stay on this path.
    leg_event_arrays: dict[str, np.ndarray]
    book_ts_per_leg: dict[str, np.ndarray]
    ref_events: list[ReferenceEvent]
    settle_events: list[SettlementEvent]

    fast_path_fn = (
        None if os.environ.get("HLBT_DISABLE_FASTPATH")
        else getattr(data_source, "events_arrays", None)
    )
    bundle = None
    if fast_path_fn is not None:
        try:
            bundle = fast_path_fn(q)
        except NotImplementedError:
            # Source exposes events_arrays() but not for this config (e.g. PM
            # synthetic book mode) — fall back to the legacy events() path.
            bundle = None
    # Flat sorted list of all leg trade events for MarketState volume accounting
    # (SHR-78). Drained incrementally in the scan loop alongside ref_events.
    all_trade_events: list[TradeEvent]

    if bundle is not None:
        leg_event_arrays = {sym: legarr.events for sym, legarr in bundle.leg_arrays.items()}
        book_ts_per_leg = {sym: legarr.book_ts for sym, legarr in bundle.leg_arrays.items()}
        ref_events = bundle.reference_events
        settle_events = bundle.settlement_events
        # Fast path: trade events are pre-read in the bundle (SHR-78).
        raw: list[TradeEvent] = []
        for evs in bundle.trade_events_per_leg.values():
            raw.extend(evs)
        raw.sort(key=lambda t: t.ts_ns)
        all_trade_events = raw
    else:
        book_events: dict[str, list[BookSnapshot]] = {sym: [] for sym in q.leg_symbols}
        trade_events: dict[str, list[TradeEvent]] = {sym: [] for sym in q.leg_symbols}
        ref_events = []
        settle_events = []
        for ev in data_source.events(q):
            if isinstance(ev, BookSnapshot):
                if ev.symbol in book_events:
                    book_events[ev.symbol].append(ev)
            elif isinstance(ev, TradeEvent):
                if ev.symbol in trade_events:
                    trade_events[ev.symbol].append(ev)
            elif isinstance(ev, ReferenceEvent):
                ref_events.append(ev)
            elif isinstance(ev, SettlementEvent):
                settle_events.append(ev)
        leg_event_arrays = {}
        book_ts_per_leg = {}
        for sym in q.leg_symbols:
            leg_event_arrays[sym] = build_leg_event_array_from_snapshots(
                book_events[sym], trade_events[sym]
            )
            book_ts_per_leg[sym] = np.asarray(
                [b.ts_ns for b in book_events[sym]], dtype=np.int64
            )
        # Legacy path: merge and sort all per-leg trade events (SHR-78).
        raw_legacy: list[TradeEvent] = []
        for evs in trade_events.values():
            raw_legacy.extend(evs)
        raw_legacy.sort(key=lambda t: t.ts_ns)
        all_trade_events = raw_legacy

    # Binary HIP-4 / PM markets settle relative to the reference price at
    # question start ("BTC > day_open"). When the caller hasn't supplied a
    # strike, derive it from the first ReferenceEvent — preferring `.open`
    # (the bar's open price, matching the legacy `day_open_btc` convention)
    # and falling back to `.close` for tick-style reference streams that
    # don't carry an opening price.
    if strike == 0.0 and ref_events:
        first_ref = ref_events[0]
        strike = float(first_ref.open) if first_ref.open > 0 else float(first_ref.close)

    # --- Build hftbacktest assets per leg ---------------------------------
    # ``BacktestAsset.data(ndarray)`` stashes the array's ctypes pointer in the
    # native engine. The Python array must stay alive for the entire backtest;
    # otherwise GC frees it and the engine reads dangling memory (visible as
    # NaN bids/asks and a corrupted ``current_timestamp``). We hold the per-leg
    # arrays in ``_data_keepalive`` for the lifetime of this call.
    assets = []
    leg_to_asset: dict[str, int] = {}
    _data_keepalive: list[np.ndarray] = []
    for i, sym in enumerate(q.leg_symbols):
        leg_arr = leg_event_arrays[sym]
        clear_arr = _initial_clear_array(q.start_ts_ns)
        full = (
            np.concatenate([clear_arr, leg_arr]) if len(leg_arr) > 0 else clear_arr
        )
        _data_keepalive.append(full)
        assets.append(_build_asset(full, cfg))
        leg_to_asset[sym] = i

    # Hedge leg: build a third BacktestAsset when hedge_enabled and events provided.
    hedge_asset_no: int | None = None
    if cfg.hedge_enabled and hedge_events is not None:
        hedge_arr = build_leg_event_array_from_snapshots(hedge_events, [])
        hedge_clear = _initial_clear_array(q.start_ts_ns)
        hedge_full = (
            np.concatenate([hedge_clear, hedge_arr]) if len(hedge_arr) > 0 else hedge_clear
        )
        _data_keepalive.append(hedge_full)
        hedge_asset_no = len(assets)
        assets.append(_build_hedge_asset(hedge_full, cfg))

    hbt = hb.HashMapMarketDepthBacktest(assets)

    state = MarketState()
    stop_pct = _strategy_stop_loss_pct(strategy)
    scan_interval_ns = cfg.scanner_interval_seconds * 1_000_000_000

    # Reference events are kept sorted and consumed incrementally.
    ref_events.sort(key=lambda r: r.ts_ns)
    ref_idx = 0
    settle_events.sort(key=lambda s: s.ts_ns)
    # Trade events: all_trade_events is already sorted (built above). Drained
    # into MarketState at each scan tick for recent_volume_usd (SHR-78).
    trade_idx = 0

    need_diag = (diagnostics_dir is not None) or (fills_dir is not None)
    diag_rows: list[DiagnosticRow] = []

    # All mutable position/fill bookkeeping lives in one struct so the routing
    # helpers (_route_enter/_route_exit/_route_hedge/_route_stop_loss/_settle)
    # can mutate it without a forest of nonlocal closures.
    st = _RunState(
        hbt=hbt,
        cfg=cfg,
        q=q,
        data_source=data_source,
        leg_to_asset=leg_to_asset,
        hedge_asset_no=hedge_asset_no,
        stop_pct=stop_pct,
        fills_dir_active=fills_dir is not None,
        result=RunResult(),
        sim_risk_caps=sim_risk_caps,
        halt_windows=tuple(halt_windows) if halt_windows else (),
    )

    # --- Scan loop ---------------------------------------------------------
    # Fixed 60s grid (mirrors `hl-sim`'s scan cadence in spirit; legacy
    # ran event-driven but PM trade density is high enough that the two
    # produce the same v2 P&L within 1e-5 USD). Event-driven was tried and
    # did NOT close the v1 trade-count gap, so the simpler form stays.

    # Per-leg cursor into the book event stream tracks the latest L2 snap ts
    # for the "stale data halt" gate. The cursor advances over book_ts_per_leg
    # (an int64 array of snapshot timestamps); the previous BookSnapshot list
    # is no longer needed by the scan loop.
    book_idx: dict[str, int] = {sym: 0 for sym in q.leg_symbols}

    while True:
        rc = hbt.elapse(scan_interval_ns)
        if rc != 0:
            break

        now_ns = int(hbt.current_timestamp)
        if now_ns >= q.end_ts_ns:
            break

        # Drain reference events up to now into MarketState.
        while ref_idx < len(ref_events) and ref_events[ref_idx].ts_ns <= now_ns:
            state.apply_reference(ref_events[ref_idx])
            ref_idx += 1

        # Drain trade events up to now into MarketState for volume accounting
        # (SHR-78). Mirrors how ref_events are consumed above.
        while trade_idx < len(all_trade_events) and all_trade_events[trade_idx].ts_ns <= now_ns:
            state.apply_trade(all_trade_events[trade_idx])
            trade_idx += 1

        # Advance per-leg book cursors to track the latest L2 snapshot's ts.
        for sym in q.leg_symbols:
            ts_arr = book_ts_per_leg.get(sym)
            if ts_arr is None or len(ts_arr) == 0:
                continue
            i = book_idx[sym]
            n = len(ts_arr)
            while i < n and ts_arr[i] <= now_ns:
                i += 1
            book_idx[sym] = i

        # Build per-leg books.
        books: dict[str, BookState] = {}
        for sym, asset_no in leg_to_asset.items():
            ts_arr = book_ts_per_leg.get(sym)
            i = book_idx[sym]
            if ts_arr is not None and i > 0:
                last_l2_ts = int(ts_arr[i - 1])
            else:
                last_l2_ts = 0
            bs = _book_state(hbt, asset_no, sym, last_l2_ts_ns=last_l2_ts)
            if bs is not None:
                books[sym] = bs
                state.apply_l2(
                    BookSnapshot(
                        ts_ns=now_ns,
                        symbol=sym,
                        bids=((bs.bid_px, bs.bid_sz or 0.0),) if bs.bid_px is not None else (),
                        asks=((bs.ask_px, bs.ask_sz or 0.0),) if bs.ask_px is not None else (),
                    )
                )

        # Inject hedge book into the books mapping so strategies can read it.
        if hedge_asset_no is not None and cfg.hedge_symbol:
            hbs = _book_state(hbt, hedge_asset_no, cfg.hedge_symbol)
            if hbs is not None:
                books[cfg.hedge_symbol] = hbs
                # Track the last observed hedge mid for end-of-data MTM (SHR-55).
                # `books` is rebuilt every tick, so a dedicated var is needed.
                if hbs.bid_px is not None and hbs.ask_px is not None:
                    st.last_hedge_mid = (hbs.bid_px + hbs.ask_px) / 2.0

        # Publish the per-tick view into the run state for the routing helpers.
        st.books = books
        st.now_ns = now_ns

        if not books:
            continue

        # Defer to the data source for the QuestionView (it knows the
        # market's resolution convention — e.g. PM Up/Down strike = Binance
        # close 24h pre-expiry, HL HIP-4 buckets carry priceThresholds in kv).
        # The runner's `strike` kwarg only acts as an override for binary
        # markets where the override is meaningful; for buckets we always
        # need the data source's view because the strategy reads kv for the
        # per-leg winning region.
        if strike > 0.0 and q.klass == "priceBinary":
            qv = build_question_view(q, now_ns=now_ns, strike=strike, settled=False)
        else:
            qv = data_source.question_view(q, now_ns=now_ns, settled=False)
        recent_returns, recent_hl = state.recent_returns_and_hl(
            now_ns=now_ns, lookback_seconds=cfg.vol_lookback_seconds
        )
        ref_close = state.latest_btc_close() or qv.strike

        decision = strategy.evaluate(
            question=qv,
            books=books,
            reference_price=float(ref_close),
            recent_returns=recent_returns,
            recent_volume_usd=state.recent_volume_usd(q.leg_symbols, now_ns=now_ns),
            position=st.pos,
            now_ns=now_ns,
            recent_hl_bars=recent_hl,
        )
        st.result.n_decisions += 1

        current_diag: DiagnosticRow | None = None
        if need_diag:
            # Best-effort yes/no extraction for binaries; bucket markets leave these None.
            yes_sym = q.leg_symbols[0] if len(q.leg_symbols) >= 1 else ""
            no_sym = q.leg_symbols[1] if len(q.leg_symbols) >= 2 else ""
            yes_book = books.get(yes_sym)
            no_book = books.get(no_sym)
            current_diag = build_diagnostic_row(
                ts_ns=now_ns,
                question_id=q.question_id,
                question_idx=q.question_idx,
                decision=decision,
                ref_price=float(ref_close),
                yes_bid=yes_book.bid_px if yes_book is not None else None,
                yes_ask=yes_book.ask_px if yes_book is not None else None,
                no_bid=no_book.bid_px if no_book is not None else None,
                no_ask=no_book.ask_px if no_book is not None else None,
            )
            if diagnostics_dir is not None:
                diag_rows.append(current_diag)
        st.current_diag = current_diag

        # Stop-loss first: if the held leg's bid has fallen through the stop,
        # force an exit before evaluating the strategy's intent. Then route the
        # hedge intents (independent of binary action / pos state), then the
        # binary ENTER / EXIT. Mutually-exclusive action branches preserved.
        _route_stop_loss(st)
        _route_hedge(st, decision)
        if decision.action == Action.ENTER and decision.intents:
            _route_enter(st, decision)
        elif decision.action == Action.EXIT and decision.intents and st.pos is not None:
            _route_exit(st, decision)

    # --- Settlement -------------------------------------------------------
    _settle(st)

    result = st.result
    fill_ts = st.fill_ts
    fill_question_idx = st.fill_question_idx
    fill_meta = st.fill_meta

    try:
        hbt.close()
    except Exception:
        # close() can raise after end-of-data depending on engine state; treat
        # as nonfatal since the run has already finished.
        logger.opt(exception=True).debug("hbt.close() raised; ignoring")

    # --- Realized P&L + persistence ---------------------------------------
    realized = 0.0
    for f in result.fills:
        notional = f.price * f.size
        if f.side == "buy":
            realized += -(notional + f.fee)
        else:
            realized += notional - f.fee
    result.realized_pnl_usd = realized

    if diagnostics_dir is not None:
        write_diagnostics(diag_rows, diagnostics_dir / f"{q.question_id}.parquet")

    if fills_dir is not None:
        fill_rows: list[FillRow] = []
        for f in result.fills:
            meta = fill_meta.get(f.cloid, {})
            fill_rows.append(
                FillRow(
                    cloid=f.cloid,
                    ts_ns=fill_ts.get(f.cloid, q.end_ts_ns),
                    side=f.side,
                    price=f.price,
                    size=f.size,
                    fee=f.fee,
                    question_id=q.question_id,
                    question_idx=fill_question_idx.get(f.cloid, q.question_idx),
                    symbol=f.symbol,
                    entry_p_model=meta.get("entry_p_model"),
                    entry_edge_chosen_side=meta.get("entry_edge_chosen_side"),
                    entry_sigma=meta.get("entry_sigma"),
                    entry_tau_yr=meta.get("entry_tau_yr"),
                    realized_pnl_at_settle=realized,
                    resolved_outcome=meta.get("resolved_outcome"),
                    is_hedge=f.is_hedge,
                )
            )
        write_fills(fill_rows, fills_dir / f"{q.question_id}.parquet")

    return result


def _hedge_avg_entry(
    prev_qty: float, prev_avg: float, fill_side: str, fill_size: float, fill_price: float,
) -> tuple[float, float]:
    """Return (new_qty, new_avg_entry) after applying a hedge fill (SHR-55).

    Qty-weighted average when adding in the same direction, basis preserved when
    reducing, and reset to the fill price when the position flips through zero.
    Replaces the bug where avg_entry was overwritten with the latest fill price
    on every top-up.
    """
    add = fill_size if fill_side == "buy" else -fill_size
    new_qty = prev_qty + add
    if prev_qty == 0.0:
        return new_qty, fill_price
    if (prev_qty > 0) == (add > 0):  # growing in the same direction
        new_avg = (prev_avg * abs(prev_qty) + fill_price * fill_size) / (
            abs(prev_qty) + fill_size
        )
        return new_qty, new_avg
    if abs(add) < abs(prev_qty):  # partial reduction — basis unchanged
        return new_qty, prev_avg
    if new_qty == 0.0:  # fully closed
        return 0.0, 0.0
    return new_qty, fill_price  # flipped past zero — new basis is the fill price


def _hedge_mtm_fill(symbol: str, qty: float, mark_px: float) -> Fill:
    """Closing Fill that marks an open hedge residual to ``mark_px`` (the last
    observed hedge mid) at end-of-data, mirroring binary settlement (SHR-55).

    No slippage/fee — this is a mark-to-market of the residual at fair value,
    not a modelled liquidation, so realized PnL reflects the hedge's true
    economic value rather than only its opening leg.
    """
    side = "sell" if qty > 0 else "buy"
    return Fill(
        cloid=f"hedge_settle:{symbol}",
        symbol=symbol,
        side=side,
        price=mark_px,
        size=abs(qty),
        fee=0.0,
        partial=False,
        is_hedge=True,
    )


def _settle_px_for_outcome(pos: Position, q: QuestionDescriptor, outcome: str) -> float:
    """Binary-leg payoff lookup.

    For ``priceBinary`` the held leg wins when it matches the resolved outcome
    (leg[0] = yes, leg[1] = no). Unknown / unrecognised outcomes settle at 0.
    """
    if outcome == "yes" and pos.symbol == (q.leg_symbols[0] if q.leg_symbols else ""):
        return 1.0
    if outcome == "no" and pos.symbol == (q.leg_symbols[1] if len(q.leg_symbols) > 1 else ""):
        return 1.0
    return 0.0


__all__ = ["RunConfig", "run_one_question"]
