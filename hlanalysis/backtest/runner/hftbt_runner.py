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
the hashmap variant is the right fit — no ROI bounds to configure, and the
constant-latency / risk-adverse-queue / no-partial-fill defaults match the
existing sim's IOC-against-synthetic-L2 semantics.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

import hftbacktest as hb
from hftbacktest import order as hb_order
from hftbacktest.types import (
    BUY_EVENT,
    DEPTH_CLEAR_EVENT,
    DEPTH_EVENT,
    EXCH_EVENT,
    LOCAL_EVENT,
    SELL_EVENT,
    TRADE_EVENT,
    event_dtype,
)

from hlanalysis.strategy.base import Strategy
from hlanalysis.strategy.types import Action, BookState, Position

from ..core.data_source import DataSource, QuestionDescriptor
from ..core.events import BookSnapshot, ReferenceEvent, SettlementEvent, TradeEvent
from ..core.question import build_question_view
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
    book_depth_assumption: float = 10_000.0
    vol_lookback_seconds: int = 86_400
    last_trades_capacity: int = 256


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STOP_DISABLED_SENTINEL = -1.0


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
    if stop_pct is None:
        return _STOP_DISABLED_SENTINEL
    return max(0.0, fill_price * (1.0 - stop_pct / 100.0))


def _build_leg_event_array(
    snapshots: list[BookSnapshot], trades: list[TradeEvent]
) -> np.ndarray:
    """Convert per-leg book + trade events into an hftbacktest event array.

    Depth events are emitted as incremental updates: ``qty=0`` clears any prior
    level whose price no longer appears in the snapshot, then ``qty=N`` events
    set the new levels. This keeps the engine's depth coherent even for
    multi-level snapshots while remaining cheap for synthetic top-of-book.
    """
    # First pass — count.
    n_events = 0
    prev_bids: set[float] = set()
    prev_asks: set[float] = set()
    for snap in snapshots:
        new_bids = {b[0] for b in snap.bids}
        new_asks = {a[0] for a in snap.asks}
        # Removals: levels in prev not in new.
        n_events += len(prev_bids - new_bids)
        n_events += len(prev_asks - new_asks)
        # Additions / size-updates: every level in new.
        n_events += len(snap.bids)
        n_events += len(snap.asks)
        prev_bids = new_bids
        prev_asks = new_asks
    n_events += len(trades)

    arr = np.zeros(n_events, dtype=event_dtype)
    idx = 0
    prev_bids = set()
    prev_asks = set()
    flag_local_exch = EXCH_EVENT | LOCAL_EVENT
    for snap in snapshots:
        new_bid_set = {b[0] for b in snap.bids}
        new_ask_set = {a[0] for a in snap.asks}
        # Remove stale bid levels (qty=0).
        for px in prev_bids - new_bid_set:
            arr[idx]["ev"] = DEPTH_EVENT | flag_local_exch | BUY_EVENT
            arr[idx]["exch_ts"] = snap.ts_ns
            arr[idx]["local_ts"] = snap.ts_ns
            arr[idx]["px"] = px
            arr[idx]["qty"] = 0.0
            idx += 1
        # Remove stale ask levels (qty=0).
        for px in prev_asks - new_ask_set:
            arr[idx]["ev"] = DEPTH_EVENT | flag_local_exch | SELL_EVENT
            arr[idx]["exch_ts"] = snap.ts_ns
            arr[idx]["local_ts"] = snap.ts_ns
            arr[idx]["px"] = px
            arr[idx]["qty"] = 0.0
            idx += 1
        # Set new bid levels.
        for px, qty in snap.bids:
            arr[idx]["ev"] = DEPTH_EVENT | flag_local_exch | BUY_EVENT
            arr[idx]["exch_ts"] = snap.ts_ns
            arr[idx]["local_ts"] = snap.ts_ns
            arr[idx]["px"] = px
            arr[idx]["qty"] = qty
            idx += 1
        # Set new ask levels.
        for px, qty in snap.asks:
            arr[idx]["ev"] = DEPTH_EVENT | flag_local_exch | SELL_EVENT
            arr[idx]["exch_ts"] = snap.ts_ns
            arr[idx]["local_ts"] = snap.ts_ns
            arr[idx]["px"] = px
            arr[idx]["qty"] = qty
            idx += 1
        prev_bids = new_bid_set
        prev_asks = new_ask_set

    for trade in trades:
        side_flag = BUY_EVENT if trade.side == "buy" else SELL_EVENT
        arr[idx]["ev"] = TRADE_EVENT | flag_local_exch | side_flag
        arr[idx]["exch_ts"] = trade.ts_ns
        arr[idx]["local_ts"] = trade.ts_ns
        arr[idx]["px"] = trade.price
        arr[idx]["qty"] = trade.size
        idx += 1

    # Sort ascending by exch_ts so hftbacktest's feed is monotone.
    if idx > 0:
        arr = arr[:idx]
        arr = arr[np.argsort(arr["exch_ts"], kind="stable")]
    return arr


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
    return (
        hb.BacktestAsset()
        .data(arr)
        .linear_asset(1.0)
        .constant_order_latency(0, 0)
        .risk_adverse_queue_model()
        .no_partial_fill_exchange()
        .trading_value_fee_model(0.0, 0.0)
        .tick_size(cfg.tick_size)
        .lot_size(cfg.lot_size)
        .last_trades_capacity(cfg.last_trades_capacity)
    )


def _book_state(hbt: Any, asset_no: int, symbol: str) -> BookState | None:
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
        last_l2_ts_ns=0,
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
) -> RunResult:
    """Run one question end-to-end through hftbacktest."""

    # --- Collect events ----------------------------------------------------
    book_events: dict[str, list[BookSnapshot]] = {sym: [] for sym in q.leg_symbols}
    trade_events: dict[str, list[TradeEvent]] = {sym: [] for sym in q.leg_symbols}
    ref_events: list[ReferenceEvent] = []
    settle_events: list[SettlementEvent] = []

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
        leg_arr = _build_leg_event_array(book_events[sym], trade_events[sym])
        clear_arr = _initial_clear_array(q.start_ts_ns)
        full = (
            np.concatenate([clear_arr, leg_arr]) if len(leg_arr) > 0 else clear_arr
        )
        _data_keepalive.append(full)
        assets.append(_build_asset(full, cfg))
        leg_to_asset[sym] = i

    hbt = hb.HashMapMarketDepthBacktest(assets)

    state = MarketState()
    result = RunResult()
    pos: Position | None = None
    diag_rows: list[DiagnosticRow] = []
    fill_meta: dict[str, dict[str, Any]] = {}
    fill_ts: dict[str, int] = {}
    fill_question_idx: dict[str, int] = {}
    stop_pct = _strategy_stop_loss_pct(strategy)
    scan_interval_ns = cfg.scanner_interval_seconds * 1_000_000_000

    # Reference events are kept sorted and consumed incrementally.
    ref_events.sort(key=lambda r: r.ts_ns)
    ref_idx = 0
    settle_events.sort(key=lambda s: s.ts_ns)

    need_diag = (diagnostics_dir is not None) or (fills_dir is not None)
    next_oid = 1

    def _next_oid() -> int:
        nonlocal next_oid
        oid = next_oid
        next_oid += 1
        return oid

    def _record_fill_from_order(
        oid: int,
        asset_no: int,
        symbol: str,
        side: str,
        cloid: str,
        intent_size: float,
    ) -> Fill | None:
        order = hbt.orders(asset_no).get(oid)
        if order is None:
            return None
        # FILLED=3, PARTIALLY_FILLED=5
        if order.status not in (3, 5):
            return None
        exec_qty = float(order.exec_qty)
        if exec_qty <= 0.0:
            return None
        exec_px = float(order.exec_price)
        # Bound to [0, 1] for binary tokens.
        exec_px = max(0.0, min(1.0, exec_px))
        # Cap fill size by the strategy's intended size and the configured
        # book-depth assumption (mirrors sim/fills.py semantics).
        exec_qty = min(exec_qty, intent_size, cfg.book_depth_assumption)
        fee = exec_px * exec_qty * cfg.fee_taker
        return Fill(
            cloid=cloid,
            symbol=symbol,
            side=side,
            price=exec_px,
            size=exec_qty,
            fee=fee,
            partial=exec_qty < intent_size,
        )

    # --- Scan loop ---------------------------------------------------------
    # The engine's `elapse(duration)` returns 1 when the *internal clock* runs
    # past the last data event, but pinning the loop on `q.end_ts_ns` is the
    # authoritative termination (some hftbacktest builds keep returning 0 when
    # the clock advances past the data without there being a *next* event to
    # report end-of-data on). Both are checked.
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

        # Build per-leg books.
        books: dict[str, BookState] = {}
        for sym, asset_no in leg_to_asset.items():
            bs = _book_state(hbt, asset_no, sym)
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

        if not books:
            continue

        qv = build_question_view(
            q, now_ns=now_ns, strike=strike, settled=False
        )
        recent_returns = state.recent_returns(now_ns=now_ns, lookback_seconds=cfg.vol_lookback_seconds)
        recent_hl = state.recent_hl_bars(now_ns=now_ns, lookback_seconds=cfg.vol_lookback_seconds)
        ref_close = state.latest_btc_close() or strike

        decision = strategy.evaluate(
            question=qv,
            books=books,
            reference_price=float(ref_close),
            recent_returns=recent_returns,
            recent_volume_usd=0.0,
            position=pos,
            now_ns=now_ns,
            recent_hl_bars=recent_hl,
        )
        result.n_decisions += 1

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

        # Stop-loss handling: if held position's leg bid has fallen through
        # the stop price, force an exit before evaluating the strategy's intent.
        if pos is not None and stop_pct is not None and pos.symbol in books:
            held_bid = books[pos.symbol].bid_px
            if held_bid is not None and held_bid <= pos.stop_loss_price:
                asset_no = leg_to_asset[pos.symbol]
                oid = _next_oid()
                hbt.submit_sell_order(
                    asset_no,
                    oid,
                    held_bid,
                    abs(pos.qty),
                    hb_order.IOC,
                    hb_order.LIMIT,
                    True,
                )
                cloid = f"stop-{oid}"
                stop_book = books[pos.symbol]
                px = _slipped_sell_price(stop_book, held_bid, cfg.slippage_bps)
                # Read the actual exec from hftbacktest; fall back to the
                # slipped price if no fill recorded (e.g. zero-bid).
                fill = _record_fill_from_order(
                    oid,
                    asset_no,
                    pos.symbol,
                    "sell",
                    cloid,
                    abs(pos.qty),
                )
                if fill is None:
                    fill = Fill(
                        cloid=cloid,
                        symbol=pos.symbol,
                        side="sell",
                        price=px,
                        size=abs(pos.qty),
                        fee=px * abs(pos.qty) * cfg.fee_taker,
                        partial=False,
                    )
                result.fills.append(fill)
                fill_ts[cloid] = now_ns
                fill_question_idx[cloid] = q.question_idx
                pos = None

        if decision.action == Action.ENTER and decision.intents and pos is None:
            intent = decision.intents[0]
            asset_no = leg_to_asset.get(intent.symbol)
            if asset_no is not None and intent.symbol in books:
                book = books[intent.symbol]
                slipped = _slipped_buy_price(book, intent.limit_price, cfg.slippage_bps)
                # Submit IOC limit at the slipped/limit price.
                oid = _next_oid()
                hbt.submit_buy_order(
                    asset_no,
                    oid,
                    slipped,
                    intent.size,
                    hb_order.IOC,
                    hb_order.LIMIT,
                    True,
                )
                fill = _record_fill_from_order(
                    oid,
                    asset_no,
                    intent.symbol,
                    "buy",
                    intent.cloid,
                    intent.size,
                )
                if fill is not None:
                    result.fills.append(fill)
                    fill_ts[fill.cloid] = now_ns
                    fill_question_idx[fill.cloid] = q.question_idx
                    if fills_dir is not None and current_diag is not None:
                        edge_chosen = (
                            current_diag.edge_yes
                            if intent.symbol == (q.leg_symbols[0] if q.leg_symbols else "")
                            else current_diag.edge_no
                        )
                        fill_meta[fill.cloid] = {
                            "entry_p_model": current_diag.p_model,
                            "entry_edge_chosen_side": edge_chosen,
                            "entry_sigma": current_diag.sigma,
                            "entry_tau_yr": current_diag.tau_yr,
                        }
                    pos = Position(
                        question_idx=q.question_idx,
                        symbol=intent.symbol,
                        qty=fill.size,
                        avg_entry=fill.price,
                        stop_loss_price=_stop_price(fill.price, stop_pct),
                        last_update_ts_ns=now_ns,
                    )
        elif decision.action == Action.EXIT and decision.intents and pos is not None:
            intent = decision.intents[0]
            asset_no = leg_to_asset.get(intent.symbol)
            if asset_no is not None and intent.symbol in books:
                book = books[intent.symbol]
                size = min(intent.size, abs(pos.qty))
                slipped = _slipped_sell_price(book, intent.limit_price, cfg.slippage_bps)
                oid = _next_oid()
                hbt.submit_sell_order(
                    asset_no,
                    oid,
                    slipped,
                    size,
                    hb_order.IOC,
                    hb_order.LIMIT,
                    True,
                )
                fill = _record_fill_from_order(
                    oid,
                    asset_no,
                    intent.symbol,
                    "sell",
                    intent.cloid,
                    size,
                )
                if fill is not None:
                    result.fills.append(fill)
                    fill_ts[fill.cloid] = now_ns
                    fill_question_idx[fill.cloid] = q.question_idx
                    pos = None

    # --- Settlement -------------------------------------------------------
    if pos is not None:
        outcome = data_source.resolved_outcome(q)
        # For binaries: held leg wins iff outcome matches. Bucket markets emit
        # per-leg SettlementEvents (spec §3.4) but lack a `symbol` on the event;
        # source implementations expose the winning leg via `resolved_outcome`
        # using a stable encoding ("leg_<idx>" for buckets) — task C owns the
        # encoding and task E reconciles. For binaries we use the literal
        # "yes"/"no" path below.
        settle_px = _settle_px_for_outcome(pos, q, outcome)
        settle_fill = Fill(
            cloid="settle",
            symbol=pos.symbol,
            side="sell" if pos.qty > 0 else "buy",
            price=settle_px,
            size=abs(pos.qty),
            fee=0.0,
            partial=False,
        )
        result.fills.append(settle_fill)
        fill_ts["settle"] = q.end_ts_ns
        fill_question_idx["settle"] = pos.question_idx
        if fills_dir is not None:
            fill_meta["settle"] = {"resolved_outcome": outcome}
        pos = None

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
                )
            )
        write_fills(fill_rows, fills_dir / f"{q.question_id}.parquet")

    return result


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
