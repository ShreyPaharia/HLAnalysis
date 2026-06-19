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

Sub-module layout
-----------------
Logic extracted from this 1755-LOC file into cohesive sub-modules:

- ``_latency.py``  — pluggable order-latency models (``LatencyModel``,
                     ``ConstantLatency``, ``SampledLatency``)
- ``_fees.py``     — binary fee models (``_binary_fee``)
- ``_fills.py``    — fill/slippage heuristics, IOC marketability checks,
                     hedge math, settlement payoff helpers
- ``_assets.py``   — hftbacktest asset-building helpers and book-state reader
- ``_routing.py``  — ``_RunState`` dataclass + order-routing helpers

All moved names are re-exported here so the public import surface is unchanged.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import hftbacktest as hb
import numpy as np
from loguru import logger

from hlanalysis.marketdata.position_math import (
    STOP_DISABLED_SENTINEL,
)
from hlanalysis.strategy.base import Strategy
from hlanalysis.strategy.types import Action, BookState, Position  # noqa: F401 (re-exported)

from ..core.data_source import DataSource, QuestionDescriptor
from ..core.events import BookSnapshot, ReferenceEvent, SettlementEvent, TradeEvent
from ..core.question import build_question_view
from ..data._fastpath_core import build_leg_event_array_from_snapshots
from ..halt_replay import (
    HaltWindow,
    SimRiskCaps,
)
from ._assets import (  # noqa: F401
    _book_state,
    _build_asset,
    _build_hedge_asset,
    _initial_clear_array,
    _latency_span,
    _sentinel_event_array,
)
from ._decision_trace import DecisionTraceWriter, build_trace_row  # noqa: F401
from ._fees import _binary_fee  # noqa: F401
from ._fills import (  # noqa: F401
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

# ---------------------------------------------------------------------------
# Re-exports from sub-modules (preserve the public import surface)
# ---------------------------------------------------------------------------
from ._latency import (  # noqa: F401
    _ORDER_LATENCY_DTYPE,
    ConstantLatency,
    LatencyModel,
    SampledLatency,
)
from ._routing import (  # noqa: F401
    _route_enter,
    _route_exit,
    _route_hedge,
    _route_stop_loss,
    _RunState,
    _settle,
)
from .market_state import MarketState
from .result import (
    DiagnosticRow,
    Fill,  # noqa: F401 (re-exported)
    FillRow,
    RunResult,
    build_diagnostic_row,
    write_diagnostics,
    write_fills,
)

# ---------------------------------------------------------------------------
# Helpers retained in this module (used only by RunConfig / run_one_question)
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
    from hlanalysis.marketdata.position_math import stop_price

    return stop_price(fill_price, stop_pct)


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
#
# NOTE: LatencyModel / ConstantLatency / SampledLatency are defined in
# _latency.py and re-exported above.


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
    latency_model: LatencyModel | None = None
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
    sim_risk_caps: SimRiskCaps | None = None

    def effective_latency_model(self) -> LatencyModel:
        """The latency model to wire into assets: the explicit ``latency_model``
        if set, else ``ConstantLatency(order_latency_ms)`` (Spec 1 default)."""
        return self.latency_model or ConstantLatency(self.order_latency_ms)


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
    decision_trace_writer: DecisionTraceWriter | None = None,
    decision_trace_dir: Path | None = None,
    decision_trace_strategy_id: str = "",
    decision_trace_config_hash: str = "",
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

    fast_path_fn = None if os.environ.get("HLBT_DISABLE_FASTPATH") else getattr(data_source, "events_arrays", None)
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
            leg_event_arrays[sym] = build_leg_event_array_from_snapshots(book_events[sym], trade_events[sym])
            book_ts_per_leg[sym] = np.asarray([b.ts_ns for b in book_events[sym]], dtype=np.int64)
        # Legacy path: merge and sort all per-leg trade events (SHR-78).
        raw_legacy: list[TradeEvent] = []
        for evs in trade_events.values():
            raw_legacy.extend(evs)
        raw_legacy.sort(key=lambda t: t.ts_ns)
        all_trade_events = raw_legacy
        # SHR-93: propagate the data source's reference_ticks mode to the scan loop
        # so the slow events() path also uses apply_reference_tick for raw ticks.
        ref_events_are_raw_ticks = getattr(data_source, "reference_ticks", "bars") == "raw"

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
    _event_mode_sentinel = _sentinel_event_array(q.end_ts_ns) if cfg.scan_mode == "event" else None

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
        assets.append(_build_asset(full, cfg, start_ts_ns=q.start_ts_ns, end_ts_ns=q.end_ts_ns))
        leg_to_asset[sym] = i

    # Hedge leg: build a third BacktestAsset when hedge_enabled and events provided.
    hedge_asset_no: int | None = None
    if cfg.hedge_enabled and hedge_events is not None:
        hedge_arr = build_leg_event_array_from_snapshots(hedge_events, [])
        hedge_clear = _initial_clear_array(q.start_ts_ns)
        hedge_full = np.concatenate([hedge_clear, hedge_arr]) if len(hedge_arr) > 0 else hedge_clear
        _data_keepalive.append(hedge_full)
        hedge_asset_no = len(assets)
        assets.append(_build_hedge_asset(hedge_full, cfg, start_ts_ns=q.start_ts_ns, end_ts_ns=q.end_ts_ns))

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
    # Skip materialising recent_hl_bars for strategies that never read it
    # (theta_harvester). Building that ~1350-element tuple-of-tuples every scan
    # tick is ~79% of the 1s-cadence scan-loop runtime; passing () is bit-identical
    # for a non-consumer. Range-based σ strategies (late_resolution/Parkinson)
    # leave consumes_hl_bars=True and get the full tuple. See Strategy.consumes_hl_bars.
    _consumes_hl_bars = getattr(strategy, "consumes_hl_bars", True)

    # Bound the per-tick recent_returns/HL window to the seconds the strategy
    # actually consumes, instead of the RunConfig default (86_400s ≈ a full day,
    # which at dt=5 is a 17 280-element array converted to a tuple EVERY scan
    # tick — the single biggest sim cost: ~32% of a 1s-cadence PM theta run).
    # ``Strategy.decision_lookback_seconds`` reports that need (None → legacy:
    # use the full RunConfig window, e.g. for Parkinson/late_resolution which we
    # leave unbounded). We provision 2× the reported need: ``slice_window`` is a
    # time-bounded window but the strategy re-slices it by COUNT (``[-n_keep:]``),
    # and the window's ``lo_idx+1`` boundary offset (plus any NaN-return gaps)
    # can leave the exact-need window one return short of n_keep — empirically
    # the 1× window matched the full tail on only ~16% of ticks while the 2×
    # window matched on 100% (95k ticks, v31_pm binary). Capped at the RunConfig
    # default so this never *widens* the window vs legacy. Bit-identical: the
    # most-recent returns are suffix-stable in lookback (a wider window only
    # prepends OLDER returns, which the strategy's tail slice discards).
    _need_lb = getattr(strategy, "decision_lookback_seconds", lambda: None)()
    if _need_lb is not None:
        _returns_lookback_s = min(cfg.vol_lookback_seconds, 2 * int(_need_lb))
    else:
        _returns_lookback_s = cfg.vol_lookback_seconds

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

    # Per-tick book build only needs legs that EVER carry a recorded book
    # snapshot. A leg with no book_ts entries keeps an empty hftbacktest depth,
    # so _book_state returns None every tick and it never enters `books` — so
    # skipping its per-tick depth read (a numba jitclass boundary crossing) +
    # apply_l2 is bit-identical. This removes the dominant per-tick cost on
    # many-leg bucket questions, where most legs are untradeable (no/empty book)
    # and, with leg-pruning on, are emitted with empty arrays on purpose. Binary
    # legs (both quoted) are unaffected. The hedge leg is handled separately.
    _active_leg_to_asset: dict[str, int] = {
        sym: asset_no for sym, asset_no in leg_to_asset.items() if len(book_ts_per_leg.get(sym, ())) > 0
    }

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

    # Per-question decision-trace shard (parallel-safe). When ``decision_trace_dir``
    # is set and no explicit writer was passed, open a per-question JSONL file
    # ``<dir>/<question_id>.jsonl`` (mirrors how diagnostics_dir / fills_dir write
    # one parquet per question). The CLI concatenates these shards into the final
    # ``--decision-trace-out`` path, so subprocess workers each write their own
    # shard. Closed before return. Zero overhead when neither is provided.
    _owned_trace_writer: DecisionTraceWriter | None = None
    if decision_trace_writer is None and decision_trace_dir is not None:
        _owned_trace_writer = DecisionTraceWriter(decision_trace_dir / f"{q.question_id}.jsonl").open()
        decision_trace_writer = _owned_trace_writer

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

    # PERF: identity-keyed memo for the per-scan tuple conversion of the σ
    # windows (see the build site below). Seeded with sentinels that can never
    # equal a real array object, so the first scan always computes.
    _prev_rets_arr: np.ndarray | None = None
    _prev_recent_returns: tuple[float, ...] = ()
    _prev_hl_arr: np.ndarray | None = None
    _prev_recent_hl_bars: tuple[tuple[float, float], ...] = ()

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
        # Convention (aligned with build_question_view in core/question.py:~37):
        # ``settled = now_ns > q.end_ts_ns``. A tick at EXACTLY end_ts_ns is
        # therefore the last valid scan (settled=False) and must be processed.
        # Using ``>=`` would break one tick early, silently skipping the final
        # scan and missing any edge/exit decision at expiry (#41).
        if now_ns > q.end_ts_ns:
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

        # Build per-leg books. Iterate only legs that ever have a book snapshot
        # (see _active_leg_to_asset): an empty leg's depth is always empty so
        # _book_state would return None — skipping it is bit-identical.
        books: dict[str, BookState] = {}
        for sym, asset_no in _active_leg_to_asset.items():
            ts_arr = book_ts_per_leg.get(sym)
            i = book_idx[sym]
            if ts_arr is not None and i > 0:
                last_l2_ts = int(ts_arr[i - 1])
            else:
                last_l2_ts = 0
            bs = _book_state(hbt, asset_no, sym, last_l2_ts_ns=last_l2_ts)
            if bs is not None:
                books[sym] = bs
                # #31: stamp apply_l2 with the snapshot's own exchange/recv
                # timestamp (last_l2_ts), NOT the current scan tick (now_ns).
                # Using now_ns makes last_l2_ts_ns in the book always equal to
                # the scan time, so the stale-data gate can never detect a stale
                # book (last_l2_ts << now_ns would be the correct signal).
                state.apply_l2(
                    BookSnapshot(
                        ts_ns=last_l2_ts,
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
        _rets_arr, _hl_arr = state.recent_returns_and_hl(now_ns=now_ns, lookback_seconds=_returns_lookback_s)
        # Match the live engine's tuple contract (scanner.py:428-430): convert
        # numpy arrays to tuples so sim and live pass the identical container
        # type to strategy.evaluate.
        #
        # PERF: the tuple build is O(window) pure-Python and runs every scan —
        # the dominant backtest cost for HL-bar consumers (e.g. v1 Parkinson over
        # a 720-bar window). ``slice_window`` returns the SAME array object across
        # consecutive scans whose window content is unchanged (its bit-identical
        # memo), so an identity check lets us reuse the previously built tuples
        # verbatim. Bit-identical: same object ⇒ same content ⇒ same tuple.
        if _rets_arr is _prev_rets_arr:
            recent_returns = _prev_recent_returns
        else:
            recent_returns = tuple(_rets_arr.tolist())
            _prev_rets_arr = _rets_arr
            _prev_recent_returns = recent_returns
        # FIX B: bulk C-level unbox via .tolist() then build inner tuples —
        # bit-identical to float(h)/float(lo) but avoids per-element numpy
        # scalar boxing in the Python loop (~68% of backtest runtime). Built
        # ONLY when the strategy consumes it (skip for theta — see above).
        if not _consumes_hl_bars:
            recent_hl_bars = ()
        elif _hl_arr is _prev_hl_arr:
            recent_hl_bars = _prev_recent_hl_bars
        else:
            recent_hl_bars = tuple((h, lo) for h, lo in _hl_arr.tolist())
            _prev_hl_arr = _hl_arr
            _prev_recent_hl_bars = recent_hl_bars
        ref_close = state.latest_btc_close() or qv.strike

        decision = strategy.evaluate(
            question=qv,
            books=books,
            reference_price=float(ref_close),
            recent_returns=recent_returns,
            recent_volume_usd=state.recent_volume_usd(q.leg_symbols, now_ns=now_ns),
            position=st.pos,
            now_ns=now_ns,
            recent_hl_bars=recent_hl_bars,
        )
        st.result.n_decisions += 1

        # Per-scan decision trace (--decision-trace-out). Zero overhead when the
        # writer is None (gate is a single None-check, no serialisation occurs).
        if decision_trace_writer is not None:
            _trace_row = build_trace_row(
                ts_ns=now_ns,
                question=qv,
                strategy_id=decision_trace_strategy_id,
                decision=decision,
                reference_price=float(ref_close),
                books=books,
                position=st.pos,
                config_hash=decision_trace_config_hash,
            )
            decision_trace_writer.write(_trace_row)

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

    # Flush + close the per-question trace shard we opened above (if any).
    if _owned_trace_writer is not None:
        _owned_trace_writer.close()

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


__all__ = [
    "DecisionTraceWriter",
    "RunConfig",
    "run_one_question",
    "LatencyModel",
    "ConstantLatency",
    "SampledLatency",
]
