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
    settlement_payoff_price,
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


# ---------------------------------------------------------------------------
# Order-latency models (SHR-89)
# ---------------------------------------------------------------------------
#
# Spec 1 wired a single constant round-trip latency into every asset. Spec 3
# generalises that into a *pluggable* model so the measured δ distribution from
# the SHR-83 trade journal can be fed in later, without a hard dependency on
# live journal data today. The interface is two methods:
#
#   - ``apply(asset, *, start_ts_ns, end_ts_ns)`` configures the hftbacktest
#     latency model on the asset builder and returns the (chained) builder.
#
# ``ConstantLatency`` (the default) reproduces Spec 1 bit-for-bit. A
# distribution model (``SampledLatency``) draws δ per scan-grid point from a
# supplied sample of millisecond latencies and feeds hftbacktest's interpolated
# order-latency model, so each order fills on ``book(decision_ts + δ)`` with δ
# varying across the question.

# Matches hftbacktest's order-latency record layout (data/utils/feed_order_latency.py).
_ORDER_LATENCY_DTYPE = np.dtype(
    [("req_ts", "i8"), ("exch_ts", "i8"), ("resp_ts", "i8"), ("_padding", "i8")],
    align=True,
)


class LatencyModel:
    """Pluggable order-latency model. Subclasses configure how an order's
    round-trip latency δ is wired into the hftbacktest asset."""

    def apply(self, asset: Any, *, start_ts_ns: int, end_ts_ns: int) -> Any:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class ConstantLatency(LatencyModel):
    """Constant round-trip latency (Spec 1 behaviour). ``latency_ms`` ms entry
    AND response latency, identical for every order."""

    latency_ms: float = 50.0

    def apply(self, asset: Any, *, start_ts_ns: int, end_ts_ns: int) -> Any:
        ns = int(self.latency_ms * 1_000_000)
        return asset.constant_order_latency(ns, ns)


@dataclass(frozen=True, slots=True)
class SampledLatency(LatencyModel):
    """Draw order entry latency δ from an empirical sample of millisecond values
    (e.g. the SHR-83 journal's measured δ distribution, or any synthetic list).

    Builds an hftbacktest interpolated order-latency array spanning the question
    on a ``step_ns`` grid, sampling one δ per grid point. The engine interpolates
    each order's latency from the surrounding grid points, so an order submitted
    at ``t`` reaches the exchange around ``t + δ`` and fills on the book then.

    Deterministic given ``seed`` — sims must be reproducible (no process-salted
    randomness). ``resp_ms`` is the response latency added on top of the entry
    latency (default 0; the entry latency is what governs the fill book)."""

    samples_ms: tuple[float, ...]
    seed: int = 0
    step_ns: int = 1_000_000_000  # 1s grid
    resp_ms: float = 0.0

    def build_latency_array(self, *, start_ts_ns: int, end_ts_ns: int) -> np.ndarray:
        if not self.samples_ms:
            raise ValueError("SampledLatency requires at least one δ sample")
        span = max(0, int(end_ts_ns) - int(start_ts_ns))
        n = max(2, span // int(self.step_ns) + 2)
        rng = np.random.default_rng(self.seed)
        samples = np.asarray(self.samples_ms, dtype=float)
        draws = rng.choice(samples, size=n)
        req_ts = int(start_ts_ns) + np.arange(n, dtype=np.int64) * int(self.step_ns)
        entry_ns = np.rint(draws * 1_000_000).astype(np.int64)
        resp_ns = int(self.resp_ms * 1_000_000)
        out = np.zeros(n, dtype=_ORDER_LATENCY_DTYPE)
        out["req_ts"] = req_ts
        out["exch_ts"] = req_ts + entry_ns
        out["resp_ts"] = req_ts + entry_ns + resp_ns
        return out

    def apply(self, asset: Any, *, start_ts_ns: int, end_ts_ns: int) -> Any:
        arr = self.build_latency_array(start_ts_ns=start_ts_ns, end_ts_ns=end_ts_ns)
        return asset.intp_order_latency(arr)


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
    # Pluggable latency model (SHR-89). None → use the constant ``order_latency_ms``
    # knob above (back-compat). Set to a ``LatencyModel`` (e.g. ``SampledLatency``
    # fed the SHR-83 journal δ distribution) to fill on ``book(decision_ts + δ)``
    # with δ varying per order.
    latency_model: "LatencyModel | None" = None
    # Hedge leg config (used by v5_delta_hedged; ignored by all other strategies)
    hedge_enabled: bool = False
    hedge_symbol: str = ""
    hedge_tick_size: float = 0.1
    hedge_lot_size: float = 0.001
    hedge_slippage_bps: float = 10.0
    hedge_fee_bps: float = 1.0
    # SHR-94: IOC marketability re-check at send+latency.
    # When True (default), the runner inspects the next recorded book snapshot
    # within [now, now + latency_ns] before submitting an IOC. If that snapshot
    # shows the ask has moved above the fill price (for a buy) or the bid has
    # moved below (for a sell), the level is flagged as fleeting and the order
    # is rejected without being submitted — matching live's zero-fill on levels
    # that disappeared within the order's latency window.
    # Set to False to restore the pre-SHR-94 behaviour where fleeting levels
    # can fill (e.g. for isolated tests that don't supply snap_best arrays).
    ioc_marketability_recheck: bool = True
    # SHR-95: event-driven scan mode. "fixed" (default) preserves legacy behaviour
    # (scan every scanner_interval_seconds). "event" mirrors the live engine:
    # evaluate on each book/reference event clamped between scan_min_interval_seconds
    # (floor — never faster than live's 0.2s) and scan_max_interval_seconds (ceiling
    # — quiet market still scans periodically at live's 2s idle-backoff).
    scan_mode: str = "fixed"
    scan_min_interval_seconds: float = 0.2
    scan_max_interval_seconds: float = 2.0
    # SHR-79/SHR-89: minimum inter-order re-fire floor (seconds). Live serializes
    # one order per leg in flight — after dispatching an IOC for a leg the engine
    # cannot dispatch another for that leg until the round-trip (submit -> ack ->
    # reconcile) completes. Measured from the 2026-06-10 live HL fills
    # (docs/research/2026-06-10-hl-live-fills-v1-v31-window.csv): distinct
    # re-fired orders (new cloid) on the same (slot, symbol) are gapped a hard
    # MINIMUM of ~0.73s (p10 ~0.90s; churny bucket #1670 median 1.31s).
    # Without a floor the event-driven sim re-fires at the 0.2s scan cadence
    # (5 Hz) and over-churns persistently-wide bucket books. The floor throttles
    # re-fires on a leg to live cadence. Default 0.0 disables it (back-compat;
    # the legacy no-floor A/B arm). The measured value for HL is ~0.75s.
    min_inter_order_seconds: float = 0.0
    # SHR-89b: wall-clock persistence marketability re-check.
    # The SHR-94 IOC re-check vetoes a fill if the next snapshot within the
    # order-latency window (~50ms) shows the ask retreated. On HL, snapshots
    # are seconds apart so the next snapshot is almost never within 50ms.
    # When ioc_fleeting_persistence_seconds > 0 a complementary check is
    # applied: if the ask level at fill_price appeared in the CURRENT snapshot
    # but NOT the previous snapshot (i.e. it has been present for fewer than
    # persistence_seconds), the IOC is vetoed (the level is "new/fleeting").
    # Default 0.0 disables it (back-compat; same as setting ioc_marketability_recheck
    # to the old latency-window-only behaviour). Measured HL value: 2.0s
    # (docs/research/2026-06-11-ioc-refire-floor-hl-fill-model.md, #2230 analysis).
    ioc_fleeting_persistence_seconds: float = 0.0
    # SHR-91: shared cross-market inventory caps. When set, run_questions_parallel
    # creates a SharedInventoryLedger for the in-process path so the concurrent /
    # total-inventory gates see positions from ALL questions that were live at a
    # given timestamp — matching the live engine's single per-slot ledger. Also
    # passed to run_one_question for the daily-loss cap and existing SHR-85 gates.
    # None (default) disables cross-question state (back-compat).
    sim_risk_caps: "SimRiskCaps | None" = None

    def effective_latency_model(self) -> "LatencyModel":
        """The latency model to wire into assets: the explicit ``latency_model``
        if set, else ``ConstantLatency(order_latency_ms)`` (Spec 1 default)."""
        return self.latency_model or ConstantLatency(self.order_latency_ms)


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


def _sentinel_event_array(ts_ns: int) -> np.ndarray:
    """Single no-op DEPTH_CLEAR event at ``ts_ns``.

    Used to extend hftbacktest's data timeline to the question's end_ts_ns in
    event-driven scan mode (SHR-95). Without at least one event at/near end_ts,
    hbt.elapse() returns rc=1 (end-of-data) as soon as the last real event is
    consumed — preventing the max-interval ceiling from firing over a quiet
    window. This sentinel keeps the timeline alive without affecting book state.
    """
    arr = np.zeros(1, dtype=event_dtype)
    flag = EXCH_EVENT | LOCAL_EVENT
    arr[0]["ev"] = DEPTH_CLEAR_EVENT | flag | BUY_EVENT
    arr[0]["exch_ts"] = ts_ns
    arr[0]["local_ts"] = ts_ns
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


def _latency_span(
    arr: np.ndarray, start_ts_ns: int | None, end_ts_ns: int | None
) -> tuple[int, int]:
    """Resolve the [start, end] ts window the latency model spans. Callers in the
    runner pass the question's start/end; low-level tests may omit them, in which
    case the window is derived from the event array's exch_ts range."""
    if start_ts_ns is not None and end_ts_ns is not None:
        return int(start_ts_ns), int(end_ts_ns)
    ts = arr["exch_ts"]
    derived_start = int(ts.min()) if len(ts) else 0
    derived_end = int(ts.max()) if len(ts) else derived_start
    return (
        int(start_ts_ns) if start_ts_ns is not None else derived_start,
        int(end_ts_ns) if end_ts_ns is not None else derived_end,
    )


def _build_asset(
    arr: np.ndarray, cfg: RunConfig, *,
    start_ts_ns: int | None = None, end_ts_ns: int | None = None,
):
    # SHR-79: partial_fill_exchange replaces no_partial_fill_exchange so IOC
    # orders fill only real recorded book depth and walk levels naturally.
    # SHR-89: the pluggable latency model (constant by default; a sampled δ
    # distribution when configured) wires the order RTT into the sim.
    start, end = _latency_span(arr, start_ts_ns, end_ts_ns)
    asset = hb.BacktestAsset().data(arr).linear_asset(1.0)
    asset = cfg.effective_latency_model().apply(asset, start_ts_ns=start, end_ts_ns=end)
    return (
        asset
        .risk_adverse_queue_model()
        .partial_fill_exchange()
        .trading_value_fee_model(0.0, 0.0)
        .tick_size(cfg.tick_size)
        .lot_size(cfg.lot_size)
        .last_trades_capacity(cfg.last_trades_capacity)
    )


def _build_hedge_asset(
    arr: np.ndarray, cfg: RunConfig, *,
    start_ts_ns: int | None = None, end_ts_ns: int | None = None,
):
    """Like _build_asset but uses hedge-specific tick/lot sizes."""
    start, end = _latency_span(arr, start_ts_ns, end_ts_ns)
    asset = hb.BacktestAsset().data(arr).linear_asset(1.0)
    asset = cfg.effective_latency_model().apply(asset, start_ts_ns=start, end_ts_ns=end)
    return (
        asset
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


def _classify_reject(book: BookState, side: str, submit_px: float) -> bool:
    """SHR-89: was a no-fill IOC a *reject* (limit unmarketable at fill time)?

    An order is a reject when it was submitted as a crossing/marketable order
    against the decision-time book (buy at/through the ask; sell at/through the
    bid) yet recorded no fill — the book moved away (or the queue wasn't swept)
    during the latency δ before the order reached the exchange. The strategy
    then re-fires on the next scan, reproducing live's churn.

    A non-fill on a *non-crossing* order (limit below the ask / above the bid),
    or when there was no opposing liquidity to cross, is a plain no-op — not a
    reject."""
    if side == "buy":
        return book.ask_px is not None and submit_px >= book.ask_px
    return book.bid_px is not None and submit_px <= book.bid_px


def _inter_order_blocked(
    now_ns: int, last_order_ns: int | None, min_inter_order_ns: int
) -> bool:
    """SHR-79/SHR-89: would dispatching an order on this leg now violate the
    minimum inter-order re-fire floor?

    Returns ``True`` when a prior order was dispatched on this leg less than
    ``min_inter_order_ns`` ago — the caller must suppress this submission and
    let the strategy re-fire on a later scan (reproducing live's one-order-in-
    flight serialization). Returns ``False`` when:
    - the floor is disabled (``min_inter_order_ns <= 0``); or
    - this is the first order on the leg (``last_order_ns is None``); or
    - at least ``min_inter_order_ns`` has elapsed since the last dispatch.
    """
    if min_inter_order_ns <= 0 or last_order_ns is None:
        return False
    return (now_ns - last_order_ns) < min_inter_order_ns


def _is_fleeting_ask(
    sym: str,
    fill_price: float,
    now_ns: int,
    latency_ns: int,
    snap_best_ask_per_leg: "dict[str, np.ndarray]",
    book_ts_per_leg: "dict[str, np.ndarray]",
    book_idx: "dict[str, int]",
    persistence_ns: int = 0,
) -> bool:
    """SHR-94 + SHR-89b: was the ask at ``fill_price`` a fleeting level?

    Two complementary checks:

    **SHR-94 (latency-window check):** looks at the *next* snapshot within the
    order-latency window ``[now_ns, now_ns + latency_ns]``. If it shows the
    ask has moved ABOVE ``fill_price``, the level is fleeting. Effective when
    HL snapshots are more frequent than the latency (~50ms).

    **SHR-89b (wall-clock persistence check, ``persistence_ns > 0``):** checks
    the *current* snapshot's prior snapshot. If the ask at ``fill_price`` just
    appeared (present in the current snapshot but ABSENT in the previous one),
    the level has been live for fewer nanoseconds than ``persistence_ns`` →
    fleeting. This catches HL's burst-sampled book where the next snapshot is
    seconds away (making the SHR-94 window never fire).

    Either check may veto the IOC independently. Returns ``False`` when:
    - No snapshot arrays for this leg.
    - persistence_ns=0 AND no next snapshot within the latency window.
    """
    snap_ask = snap_best_ask_per_leg.get(sym)
    ts_arr = book_ts_per_leg.get(sym)
    if snap_ask is None or ts_arr is None or len(snap_ask) == 0 or len(ts_arr) == 0:
        return False
    # book_idx[sym] is the first snapshot index NOT YET consumed at now_ns
    # (the scan loop advances it past all ts <= now_ns).
    idx = book_idx.get(sym, 0)

    # SHR-89b: wall-clock persistence check on the CURRENT (most-recently-
    # consumed) snapshot. current_idx = idx - 1 (last consumed snapshot).
    if persistence_ns > 0:
        current_idx = idx - 1
        if current_idx >= 0:
            current_ask = float(snap_ask[current_idx])
            if not np.isnan(current_ask) and current_ask <= fill_price:
                # The ask is marketable at the current snapshot. Check if it
                # appeared here for the first time (no prior snapshot) or if
                # the previous snapshot had a DIFFERENT ask.
                if current_idx == 0:
                    # First-ever snapshot: the level just appeared → fleeting.
                    return True
                prev_ask = float(snap_ask[current_idx - 1])
                if np.isnan(prev_ask) or prev_ask > fill_price:
                    # Previous snapshot had no marketable ask at fill_price:
                    # the level is new. Check persistence duration.
                    current_ts = int(ts_arr[current_idx])
                    prev_ts = int(ts_arr[current_idx - 1])
                    if current_ts - prev_ts < persistence_ns:
                        return True  # level just appeared; within persistence window

    # SHR-94: forward-looking latency-window check.
    if idx >= len(ts_arr):
        return False  # no next snapshot
    next_ts = int(ts_arr[idx])
    # Only a snapshot that falls STRICTLY WITHIN the latency window can reveal
    # a fleeting level. A snapshot AFTER arrival (next_ts > now_ns + latency_ns)
    # means the book was stable through the latency window — fill is valid.
    arrival_ns = now_ns + latency_ns
    if next_ts > arrival_ns:
        return False
    # The next snapshot is within [now_ns, arrival_ns]. Check if the ask at
    # that snapshot is worse (higher) than fill_price — fleeting!
    next_ask = float(snap_ask[idx])
    if np.isnan(next_ask):
        return False  # no ask recorded at that snapshot — can't tell; allow fill
    return next_ask > fill_price


def _is_fleeting_bid(
    sym: str,
    fill_price: float,
    now_ns: int,
    latency_ns: int,
    snap_best_bid_per_leg: "dict[str, np.ndarray]",
    book_ts_per_leg: "dict[str, np.ndarray]",
    book_idx: "dict[str, int]",
    persistence_ns: int = 0,
) -> bool:
    """SHR-94 + SHR-89b: was the bid at ``fill_price`` a fleeting level?

    Symmetric to ``_is_fleeting_ask`` for the sell side.

    **SHR-94:** next snapshot within latency window shows bid BELOW fill_price.
    **SHR-89b:** bid just appeared in the current snapshot (not in the prior
    one) and has been present for fewer than ``persistence_ns`` nanoseconds.
    """
    snap_bid = snap_best_bid_per_leg.get(sym)
    ts_arr = book_ts_per_leg.get(sym)
    if snap_bid is None or ts_arr is None or len(snap_bid) == 0 or len(ts_arr) == 0:
        return False
    idx = book_idx.get(sym, 0)

    # SHR-89b: wall-clock persistence check (bid side).
    if persistence_ns > 0:
        current_idx = idx - 1
        if current_idx >= 0:
            current_bid = float(snap_bid[current_idx])
            if not np.isnan(current_bid) and current_bid >= fill_price:
                if current_idx == 0:
                    return True
                prev_bid = float(snap_bid[current_idx - 1])
                if np.isnan(prev_bid) or prev_bid < fill_price:
                    current_ts = int(ts_arr[current_idx])
                    prev_ts = int(ts_arr[current_idx - 1])
                    if current_ts - prev_ts < persistence_ns:
                        return True

    # SHR-94: forward-looking latency-window check.
    if idx >= len(ts_arr):
        return False
    next_ts = int(ts_arr[idx])
    arrival_ns = now_ns + latency_ns
    if next_ts > arrival_ns:
        return False
    next_bid = float(snap_bid[idx])
    if np.isnan(next_bid):
        return False
    return next_bid < fill_price


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
    # SHR-79/SHR-89: per-leg timestamp (ns) of the last order DISPATCHED to the
    # venue (filled, rejected, or fleeting-vetoed — any round-trip consumed).
    # Read by the inter-order re-fire floor to throttle re-submission on a leg.
    last_order_ns: dict[str, int] = field(default_factory=dict)
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
    # SHR-94: per-snapshot top-of-book arrays for IOC marketability re-check.
    # Populated from LegArrays.snap_best_ask/bid (parallel to book_ts_per_leg).
    # Empty dict (default) disables the check — synthetic / PM sources that
    # don't supply snap_best arrays fall back to the pre-SHR-94 behaviour.
    snap_best_ask_per_leg: "dict[str, np.ndarray]" = field(default_factory=dict)
    snap_best_bid_per_leg: "dict[str, np.ndarray]" = field(default_factory=dict)
    # book_ts_per_leg: per-leg snapshot timestamps, parallel to snap_best arrays.
    # Injected from run_one_question; used by _is_fleeting_ask/bid.
    book_ts_per_leg: "dict[str, np.ndarray]" = field(default_factory=dict)
    # Per-leg snapshot cursor (shared with stale-book gate in the scan loop).
    # Injected via the runner's book_idx dict — the _RunState doesn't own it
    # directly; it's passed to _is_fleeting_level via the routing helpers.
    _book_idx: "dict[str, int] | None" = None
    # SHR-91: cross-question inventory visible to the entry gate.
    # Populated by run_questions_parallel from the SharedInventoryLedger
    # (positions held in OTHER questions that were live at now_ns). Defaults to
    # zero so all existing callers are unaffected.
    extra_held_notional: float = 0.0
    extra_n_held: int = 0
    # SHR-91: timestamp when the current position opened (None = no position).
    # Used to record position_windows when a position closes.
    _pos_open_ts_ns: "int | None" = None

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
    extra_held_notional: float = 0.0,
    extra_n_held: int = 0,
) -> RunResult:
    """Run one question end-to-end through hftbacktest.

    SHR-85: ``halt_windows`` (live halt periods to replay) and ``sim_risk_caps``
    (daily-loss / inventory caps) make the sim stop ENTERING when live would have
    stopped. Both default to off, so callers that don't pass them are unaffected;
    exits are never suppressed (matching live).

    SHR-91: ``extra_held_notional`` and ``extra_n_held`` carry cross-question
    inventory from the SharedInventoryLedger (positions held in other questions
    that overlapped in real time). Added to the current question's held inventory
    inside ``entry_blocked`` so the concurrent / total-inventory caps see the
    full picture. Default 0 (no cross-question state) preserves existing callers."""

    # SHR-91: the sim_risk_caps kwarg takes precedence over cfg.sim_risk_caps;
    # fall back to the config field when the kwarg is not explicitly supplied.
    # This allows the in-process parallel runner to pass per-question caps that
    # override (or supplement) the config-level caps.
    if sim_risk_caps is None:
        sim_risk_caps = cfg.sim_risk_caps

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

    # SHR-93: True when the reference stream is raw per-tick events rather than
    # pre-bucketed OHLC bars. The runner uses apply_reference_tick instead of
    # apply_reference so last_mark is the instantaneous tick price (live-parity).
    ref_events_are_raw_ticks: bool = False
    # SHR-94: per-snapshot top-of-book arrays for the IOC marketability re-check.
    # Populated from LegArrays when the fast path provides them; empty dict for
    # the legacy path (synthetic / PM sources → recheck is skipped).
    snap_best_ask_per_leg: dict[str, np.ndarray] = {}
    snap_best_bid_per_leg: dict[str, np.ndarray] = {}

    if bundle is not None:
        leg_event_arrays = {sym: legarr.events for sym, legarr in bundle.leg_arrays.items()}
        book_ts_per_leg = {sym: legarr.book_ts for sym, legarr in bundle.leg_arrays.items()}
        ref_events = bundle.reference_events
        settle_events = bundle.settlement_events
        ref_events_are_raw_ticks = getattr(bundle, "reference_events_are_raw_ticks", False)
        # Fast path: trade events are pre-read in the bundle (SHR-78).
        raw: list[TradeEvent] = []
        for evs in bundle.trade_events_per_leg.values():
            raw.extend(evs)
        raw.sort(key=lambda t: t.ts_ns)
        all_trade_events = raw
        # SHR-94: extract per-snapshot best ask/bid for the fleeting-level check.
        for sym, legarr in bundle.leg_arrays.items():
            if len(legarr.snap_best_ask) > 0:
                snap_best_ask_per_leg[sym] = legarr.snap_best_ask
            if len(legarr.snap_best_bid) > 0:
                snap_best_bid_per_leg[sym] = legarr.snap_best_bid
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
        # SHR-93: propagate the data source's reference_ticks mode to the scan loop
        # so the slow events() path also uses apply_reference_tick for raw ticks.
        ref_events_are_raw_ticks = (
            getattr(data_source, "reference_ticks", "bars") == "raw"
        )

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
    # SHR-95: in event-driven mode append a no-op sentinel at end_ts_ns to each
    # leg's event array. This extends hftbacktest's data timeline to the question
    # end so max-interval ceiling scans can fire over quiet windows — without this,
    # hbt.elapse() returns rc=1 (end-of-data) as soon as the last real event is
    # consumed. Fixed-mode is unaffected (sentinel is not appended).
    _event_mode_sentinel = (
        _sentinel_event_array(q.end_ts_ns)
        if cfg.scan_mode == "event"
        else None
    )

    assets = []
    leg_to_asset: dict[str, int] = {}
    _data_keepalive: list[np.ndarray] = []
    for i, sym in enumerate(q.leg_symbols):
        leg_arr = leg_event_arrays[sym]
        clear_arr = _initial_clear_array(q.start_ts_ns)
        parts = [clear_arr]
        if len(leg_arr) > 0:
            parts.append(leg_arr)
        if _event_mode_sentinel is not None:
            parts.append(_event_mode_sentinel)
        full = np.concatenate(parts) if len(parts) > 1 else parts[0]
        _data_keepalive.append(full)
        assets.append(
            _build_asset(full, cfg, start_ts_ns=q.start_ts_ns, end_ts_ns=q.end_ts_ns)
        )
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
        assets.append(
            _build_hedge_asset(
                hedge_full, cfg, start_ts_ns=q.start_ts_ns, end_ts_ns=q.end_ts_ns
            )
        )

    hbt = hb.HashMapMarketDepthBacktest(assets)

    state = MarketState()
    # SHR-96 / SHR-97: always configure the OHLC bucket width, regardless of
    # whether ticks are raw or pre-bucketed bars. In bars mode the shared core's
    # _OhlcBuffer uses the registered bucket period for set_reference_cadence and
    # the bar-append path doesn't call ingest_tick (so no OHLC coalescing occurs),
    # making this registration a no-op for bars — but calling it unconditionally
    # closes the SHR-96 class of bug (cadence left at 60s when ticks are raw)
    # structurally: there is no longer a conditional branch that can be "off".
    _ref_dt_s: int = int(getattr(data_source, "reference_resample_seconds", 60))
    state.set_reference_cadence(_ref_dt_s)
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

    # Per-leg cursor into the book event stream tracks the latest L2 snap ts
    # for the "stale data halt" gate AND (SHR-94) the next snapshot for the
    # fleeting-level check. The cursor advances over book_ts_per_leg (an int64
    # array of snapshot timestamps); the previous BookSnapshot list is no longer
    # needed by the scan loop. Defined BEFORE _RunState so the _RunState's
    # _book_idx field can alias the same dict.
    book_idx: dict[str, int] = {sym: 0 for sym in q.leg_symbols}

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
        snap_best_ask_per_leg=snap_best_ask_per_leg,
        snap_best_bid_per_leg=snap_best_bid_per_leg,
        book_ts_per_leg=book_ts_per_leg,
        _book_idx=book_idx,
        extra_held_notional=extra_held_notional,
        extra_n_held=extra_n_held,
    )

    # --- Scan loop ---------------------------------------------------------
    # Two modes (SHR-95):
    #
    # "fixed" (default): advance hbt by scan_interval_ns every iteration —
    #   the original fixed-grid behaviour. Back-compat: bit-identical to the
    #   pre-SHR-95 scan cadence for all callers that don't opt into "event".
    #
    # "event": mirrors the live engine's event-driven cadence. On each loop
    #   iteration we compute the next scan target as the earliest trigger among:
    #     a) the next book/reference event timestamp (trigger on market activity)
    #     b) last_scan_ns + max_interval_ns  (idle-backoff ceiling)
    #   …then clamp it so it is at least last_scan_ns + min_interval_ns (floor).
    #   This reproduces live's scan_min/scan_max interval semantics without
    #   requiring changes to the hftbacktest asset or event array.
    #
    # Event-mode pre-build: merge all book-event timestamps + ref-event
    # timestamps into one sorted int64 array used as the trigger schedule.

    event_mode = cfg.scan_mode == "event"
    _event_triggers: np.ndarray  # sorted int64, only used in event mode

    if event_mode:
        min_interval_ns = int(cfg.scan_min_interval_seconds * 1_000_000_000)
        max_interval_ns = int(cfg.scan_max_interval_seconds * 1_000_000_000)
        # Merge all leg book-update timestamps + reference event timestamps.
        parts: list[np.ndarray] = []
        for ts_arr in book_ts_per_leg.values():
            if len(ts_arr) > 0:
                parts.append(ts_arr.astype(np.int64))
        if ref_events:
            parts.append(np.array([r.ts_ns for r in ref_events], dtype=np.int64))
        if parts:
            _event_triggers = np.sort(np.concatenate(parts))
        else:
            _event_triggers = np.empty(0, dtype=np.int64)
        _evt_idx: int = 0
        _last_scan_ns: int = q.start_ts_ns
    else:
        min_interval_ns = 0  # unused in fixed mode
        max_interval_ns = 0  # unused in fixed mode
        _event_triggers = np.empty(0, dtype=np.int64)  # unused
        _evt_idx = 0
        _last_scan_ns = q.start_ts_ns

    while True:
        if event_mode:
            # Determine how far to advance hbt to reach the next scan target.
            # The ceiling is: last scan + max_interval (idle-backoff).
            next_target = _last_scan_ns + max_interval_ns
            # Pull the next event trigger that is strictly after last_scan_ns.
            while _evt_idx < len(_event_triggers) and _event_triggers[_evt_idx] <= _last_scan_ns:
                _evt_idx += 1
            if _evt_idx < len(_event_triggers):
                next_event_ts = int(_event_triggers[_evt_idx])
                # Clamp to the max-interval ceiling.
                next_target = min(next_target, next_event_ts)
            # Enforce the min-interval floor.
            next_target = max(next_target, _last_scan_ns + min_interval_ns)
            # Advance hbt by the delta from its current position.
            current_ts = int(hbt.current_timestamp)
            delta_ns = max(1, next_target - current_ts)
            rc = hbt.elapse(delta_ns)
        else:
            rc = hbt.elapse(scan_interval_ns)
        if rc != 0:
            break

        now_ns = int(hbt.current_timestamp)
        if event_mode:
            _last_scan_ns = now_ns
        if now_ns >= q.end_ts_ns:
            break

        # Drain reference events up to now into MarketState.
        # SHR-93: raw-tick mode (reference_ticks="raw") routes ticks through
        # apply_reference_tick so last_mark is the instantaneous raw price
        # (live-parity). Bar mode uses apply_reference (pre-bucketed OHLC bars).
        if ref_events_are_raw_ticks:
            while ref_idx < len(ref_events) and ref_events[ref_idx].ts_ns <= now_ns:
                state.apply_reference_tick(ref_events[ref_idx])
                ref_idx += 1
        else:
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
    """Binary-leg settlement payoff, via the shared
    :func:`position_math.settlement_payoff_price` (SHR-88).

    The held leg's index (leg[0] = YES, leg[1] = NO) is the position side; the
    venue/recorded resolved ``outcome`` supplies the winning (settled) side
    index — ``yes`` -> 0, ``no`` -> 1. Routing through the shared function makes
    the sim book payoffs identically to the live engine and, critically, derives
    the winner from the resolved outcome rather than re-deriving a YES winner.
    An unrecognised outcome, or a position whose leg isn't one of the question's
    legs, can never match the settled side and settles worthless (0.0) — exactly
    as before.
    """
    legs = q.leg_symbols
    yes_leg = legs[0] if legs else ""
    no_leg = legs[1] if len(legs) > 1 else ""
    if pos.symbol == yes_leg:
        position_side_idx = 0
    elif pos.symbol == no_leg:
        position_side_idx = 1
    else:
        return 0.0  # held leg isn't part of this question — can't win
    if outcome == "yes":
        settled_side_idx = 0
    elif outcome == "no":
        settled_side_idx = 1
    else:
        return 0.0  # unresolved / unknown outcome settles worthless
    return settlement_payoff_price(position_side_idx, settled_side_idx)


__all__ = [
    "RunConfig",
    "run_one_question",
    "LatencyModel",
    "ConstantLatency",
    "SampledLatency",
]
