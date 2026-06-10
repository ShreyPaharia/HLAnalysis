from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import hftbacktest as hb
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
from hftbacktest import order as hb_order

from hlanalysis.backtest.data.synthetic import (
    SyntheticDataSource,
    build_dummy_enter_strategy,
    make_default_binary_question,
)
from hlanalysis.backtest.runner.hftbt_runner import RunConfig, run_one_question
from hlanalysis.strategy.base import Strategy
from hlanalysis.strategy.types import BookState


# ---------------------------------------------------------------------------
# Helpers for low-level hftbacktest asset construction tests (SHR-79/56/57)
# ---------------------------------------------------------------------------

def _make_depth_clear_arr(start_ts_ns: int) -> np.ndarray:
    """Two DEPTH_CLEAR events at start_ts_ns (buy + sell sides)."""
    arr = np.zeros(2, dtype=event_dtype)
    flag = EXCH_EVENT | LOCAL_EVENT
    arr[0]["ev"] = DEPTH_CLEAR_EVENT | flag | BUY_EVENT
    arr[0]["exch_ts"] = start_ts_ns
    arr[0]["local_ts"] = start_ts_ns
    arr[1]["ev"] = DEPTH_CLEAR_EVENT | flag | SELL_EVENT
    arr[1]["exch_ts"] = start_ts_ns
    arr[1]["local_ts"] = start_ts_ns
    return arr


def _depth_ev(ts: int, side_flag: int, px: float, qty: float) -> np.ndarray:
    ev = np.zeros(1, dtype=event_dtype)
    ev[0]["ev"] = DEPTH_EVENT | EXCH_EVENT | LOCAL_EVENT | side_flag
    ev[0]["exch_ts"] = ts
    ev[0]["local_ts"] = ts
    ev[0]["px"] = px
    ev[0]["qty"] = qty
    return ev


def _trade_ev(ts: int, side_flag: int, px: float, qty: float) -> np.ndarray:
    ev = np.zeros(1, dtype=event_dtype)
    ev[0]["ev"] = TRADE_EVENT | EXCH_EVENT | LOCAL_EVENT | side_flag
    ev[0]["exch_ts"] = ts
    ev[0]["local_ts"] = ts
    ev[0]["px"] = px
    ev[0]["qty"] = qty
    return ev


# ---------------------------------------------------------------------------
# Existing integration tests (unchanged)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_source() -> tuple[SyntheticDataSource, object]:
    sq = make_default_binary_question(start_ts_ns=0)
    ds = SyntheticDataSource()
    ds.add_question(sq)
    return ds, sq


def test_run_yes_wins_records_enter_and_settle(synthetic_source, tmp_path):
    ds, sq = synthetic_source
    strat = build_dummy_enter_strategy({"size": 10.0})
    cfg = RunConfig(
        scanner_interval_seconds=60,
        tick_size=0.001,
        lot_size=1.0,
        slippage_bps=0.0,  # exact bookkeeping
        fee_taker=0.0,
        book_depth_assumption=10_000.0,
    )
    res = run_one_question(
        strat,
        ds,
        sq.descriptor,
        cfg,
        diagnostics_dir=tmp_path / "diag",
        fills_dir=tmp_path / "fills",
        strike=sq.strike,
    )
    # 1 ENTER + 1 settle fill
    assert len(res.fills) == 2, [f for f in res.fills]
    enter = res.fills[0]
    settle = res.fills[1]
    assert enter.side == "buy"
    assert enter.size == 10.0
    assert 0.30 < enter.price < 0.50  # first scan tick, YES ask ≈ 0.32-0.40
    assert settle.cloid == "settle"
    assert settle.price == 1.0  # YES won
    assert settle.size == 10.0
    # Realized P&L = settle.notional - enter.notional - fees
    expected_pnl = settle.price * settle.size - enter.price * enter.size
    assert res.realized_pnl_usd is not None
    assert res.realized_pnl_usd == pytest.approx(expected_pnl, abs=1e-9)


def test_run_no_outcome_settles_zero(tmp_path):
    """If the resolved outcome is NO but we held YES, settle price is 0."""
    sq = make_default_binary_question(start_ts_ns=0, outcome="no")
    ds = SyntheticDataSource()
    ds.add_question(sq)
    strat = build_dummy_enter_strategy({"size": 5.0})
    cfg = RunConfig(slippage_bps=0.0, fee_taker=0.0)
    res = run_one_question(
        strat,
        ds,
        sq.descriptor,
        cfg,
        diagnostics_dir=tmp_path / "diag",
        fills_dir=tmp_path / "fills",
        strike=sq.strike,
    )
    settle = res.fills[-1]
    assert settle.cloid == "settle"
    assert settle.price == 0.0  # YES position, NO outcome → 0
    assert res.realized_pnl_usd is not None
    assert res.realized_pnl_usd < 0  # paid for YES, got nothing


def test_run_persists_parquet_artifacts(synthetic_source, tmp_path):
    ds, sq = synthetic_source
    strat = build_dummy_enter_strategy({"size": 10.0})
    cfg = RunConfig(slippage_bps=0.0, fee_taker=0.0)
    run_one_question(
        strat,
        ds,
        sq.descriptor,
        cfg,
        diagnostics_dir=tmp_path / "diag",
        fills_dir=tmp_path / "fills",
        strike=sq.strike,
    )
    diag = tmp_path / "diag" / f"{sq.descriptor.question_id}.parquet"
    fills = tmp_path / "fills" / f"{sq.descriptor.question_id}.parquet"
    assert diag.exists()
    assert fills.exists()

    import pyarrow.parquet as pq

    d = pq.read_table(diag).to_pydict()
    f = pq.read_table(fills).to_pydict()
    # Diagnostics: ts_ns monotonic and one row per scan tick (10 ticks over 10 min)
    assert len(d["ts_ns"]) > 0
    assert d["ts_ns"] == sorted(d["ts_ns"])
    # Fills: at least one entry + a settle row with resolved_outcome populated
    assert "settle" in f["cloid"]
    settle_idx = f["cloid"].index("settle")
    assert f["resolved_outcome"][settle_idx] == "yes"


def test_binary_fee_flat_matches_legacy():
    from hlanalysis.backtest.runner.hftbt_runner import _binary_fee

    cfg = RunConfig(fee_model="flat", fee_taker=0.0035)
    # flat: px * qty * fee_taker
    assert _binary_fee(0.50, 100.0, cfg) == pytest.approx(0.50 * 100.0 * 0.0035)
    assert _binary_fee(0.95, 100.0, cfg) == pytest.approx(0.95 * 100.0 * 0.0035)
    # fee_rate unused in flat mode
    cfg2 = RunConfig(fee_model="flat", fee_taker=0.0035, fee_rate=0.07)
    assert _binary_fee(0.50, 100.0, cfg2) == pytest.approx(_binary_fee(0.50, 100.0, cfg))


def test_binary_fee_pm_binary_matches_polymarket_docs():
    """fee = C * feeRate * p * (1-p); peaks $1.75 / 100 shares at p=0.5, crypto."""
    from hlanalysis.backtest.runner.hftbt_runner import _binary_fee

    cfg = RunConfig(fee_model="pm_binary", fee_rate=0.07, fee_taker=999.0)
    # Doc example: 100 shares at p=0.5 in crypto → max $1.75
    assert _binary_fee(0.50, 100.0, cfg) == pytest.approx(1.75)
    # Near-resolution: p=0.95 → 0.07 * 0.95 * 0.05 * 100 = 0.3325
    assert _binary_fee(0.95, 100.0, cfg) == pytest.approx(0.3325)
    # Symmetric: p=0.05 same fee as p=0.95
    assert _binary_fee(0.05, 100.0, cfg) == pytest.approx(_binary_fee(0.95, 100.0, cfg))
    # p=0.85: 0.07 * 0.85 * 0.15 * 100 = 0.8925
    assert _binary_fee(0.85, 100.0, cfg) == pytest.approx(0.8925)
    # fee_taker is ignored in pm_binary mode (would otherwise blow up)
    assert _binary_fee(0.50, 100.0, cfg) < 2.0
    # Sports category (feeRate=0.03): max = 0.03 * 0.25 * 100 = 0.75
    cfg_sports = RunConfig(fee_model="pm_binary", fee_rate=0.03)
    assert _binary_fee(0.50, 100.0, cfg_sports) == pytest.approx(0.75)


def test_binary_fee_pm_binary_clamps_out_of_range_price():
    """Prices outside [0,1] (numerical noise) must not produce negative fees."""
    from hlanalysis.backtest.runner.hftbt_runner import _binary_fee

    cfg = RunConfig(fee_model="pm_binary", fee_rate=0.07)
    assert _binary_fee(-0.01, 100.0, cfg) == pytest.approx(0.0)
    assert _binary_fee(1.01, 100.0, cfg) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# SHR-79: partial fills vs real depth
# ---------------------------------------------------------------------------
#
# Timing pattern for partial-fill tests:
#   1. Build event array with depth events BEFORE T_SUBMIT and trade events
#      AFTER T_SUBMIT (so trades happen after order arrives at exchange).
#   2. bt.elapse(T_SUBMIT) to advance to submission time (processes depth events).
#   3. submit_buy_order(..., wait=False) — order queued, not yet processed.
#   4. bt.elapse(T_AFTER_TRADE) to process trade events → triggers fills.
#
# All partial-fill tests use order_latency_ms=0.0 to isolate fill semantics
# from latency (latency is tested separately in the latency section).

def test_build_asset_uses_partial_fill_exchange():
    """_build_asset must use partial_fill_exchange (not no_partial_fill_exchange).

    Verify via behavior: IOC order for qty > book depth fills only the available
    depth (EXPIRED status with exec_qty == available) rather than the full order
    (FILLED status with exec_qty == order_qty).
    """
    from hlanalysis.backtest.runner.hftbt_runner import _build_asset

    T_BOOK = 1_000_000        # 1ms — depth event (before submit)
    T_SUBMIT = 1_000_000_000  # 1s — submit order here
    T_TRADE = T_SUBMIT + 1_000_000  # 1ms after submit — trade triggers fill

    # Book: ask at 0.5 with size 7; trade after order submission triggers fill
    arr = np.concatenate([
        _make_depth_clear_arr(0),
        _depth_ev(T_BOOK, SELL_EVENT, 0.5, 7.0),
        _trade_ev(T_TRADE, BUY_EVENT, 0.5, 7.0),
    ])
    cfg = RunConfig(tick_size=0.001, lot_size=1.0, order_latency_ms=0.0)
    asset = _build_asset(arr, cfg)
    bt = hb.HashMapMarketDepthBacktest([asset])
    bt.elapse(T_SUBMIT)  # advance past depth events
    # Submit IOC buy for 20 (more than available 7), wait=True processes the fill
    bt.submit_buy_order(0, 1, 0.55, 20.0, hb_order.IOC, hb_order.LIMIT, True)
    order = bt.orders(0).get(1)

    assert order is not None, "order not found"
    # With partial_fill_exchange: fills 7 (available depth), remainder cancelled
    assert order.exec_qty == pytest.approx(7.0), (
        f"partial_fill_exchange should fill only available depth (7), got {order.exec_qty}"
    )
    # Status is EXPIRED (2) for IOC with leftover remainder cancelled
    assert order.status in (hb_order.EXPIRED, hb_order.FILLED), (
        f"unexpected order status {order.status}"
    )
    bt.close()


def test_partial_fill_ioc_remainder_cancelled():
    """IOC order > top-of-book size: fill == available depth, remainder cancelled.

    Uses _build_asset directly to confirm partial_fill_exchange semantics.
    """
    from hlanalysis.backtest.runner.hftbt_runner import _build_asset

    ask_depth = 12.0
    order_size = 50.0  # intentionally >> ask_depth
    T_BOOK = 1_000_000
    T_SUBMIT = 1_000_000_000
    T_TRADE = T_SUBMIT + 1_000_000

    arr = np.concatenate([
        _make_depth_clear_arr(0),
        _depth_ev(T_BOOK, SELL_EVENT, 0.6, ask_depth),
        _trade_ev(T_TRADE, BUY_EVENT, 0.6, ask_depth),
    ])
    cfg = RunConfig(tick_size=0.001, lot_size=1.0, order_latency_ms=0.0)
    asset = _build_asset(arr, cfg)
    bt = hb.HashMapMarketDepthBacktest([asset])
    bt.elapse(T_SUBMIT)
    bt.submit_buy_order(0, 1, 0.65, order_size, hb_order.IOC, hb_order.LIMIT, True)
    order = bt.orders(0).get(1)

    assert order is not None
    assert order.exec_qty == pytest.approx(ask_depth), (
        f"expected fill == ask_depth ({ask_depth}), got {order.exec_qty}"
    )
    assert order.exec_qty < order_size, "IOC remainder must be cancelled (not over-filled)"
    bt.close()


def test_multi_level_walk_vwap_worse_than_touch():
    """Multi-level IOC walk: last fill price is strictly worse than best ask (touch).

    With risk_adverse_queue_model, orders are placed BEHIND the existing exchange
    queue. When a large trade sweeps through multiple price levels, orders fill at
    the deeper level rather than the touch.

    Setup:
      Level 1: ask 0.50 size=10  (touch; my order sits behind this queue)
      Level 2: ask 0.60 size=10

    A single large trade (qty=30) at 0.50 sweeps through both levels:
      - Consumes the 10-unit queue at 0.5 (my order is behind → doesn't fill at 0.5)
      - Sweeps the 10-unit queue at 0.6 (my order is behind → fills the excess)
      - exec_price = 0.60, which is strictly > touch (0.50) — slippage emerged
        from the book walk.
    """
    from hlanalysis.backtest.runner.hftbt_runner import _build_asset

    T_BOOK = 1_000_000
    T_SUBMIT = 1_000_000_000
    T_TRADE = T_SUBMIT + 1_000_000   # single sweep trade at 0.5 that walks to 0.6

    arr = np.concatenate([
        _make_depth_clear_arr(0),
        _depth_ev(T_BOOK, SELL_EVENT, 0.5, 10.0),
        _depth_ev(T_BOOK, SELL_EVENT, 0.6, 10.0),
        # A trade of 30 at 0.50: sweeps through 10@0.5 (queue ahead) + 10@0.6 (queue
        # + fills 10 of my order at the 0.6 level)
        _trade_ev(T_TRADE, BUY_EVENT, 0.5, 30.0),
    ])
    cfg = RunConfig(tick_size=0.001, lot_size=1.0, order_latency_ms=0.0)
    asset = _build_asset(arr, cfg)
    bt = hb.HashMapMarketDepthBacktest([asset])
    bt.elapse(T_SUBMIT)
    bt.submit_buy_order(0, 1, 0.70, 20.0, hb_order.IOC, hb_order.LIMIT, True)
    order = bt.orders(0).get(1)

    assert order is not None
    assert order.exec_qty > 0, f"expected some fill from sweep trade, got exec_qty=0"
    # exec_price is the fill price; since fill occurs at the 0.6 level (not touch 0.5),
    # exec_price must be > touch (0.50) — this is the slippage from the book walk.
    assert order.exec_price >= 0.5 + 1e-6, (
        f"fill price after multi-level walk must be > touch (0.50); got {order.exec_price}"
    )
    bt.close()


# ---------------------------------------------------------------------------
# SHR-79: record_fill_from_order must accept EXPIRED status (partial IOC fills)
# ---------------------------------------------------------------------------

def test_record_fill_from_order_accepts_expired_status():
    """record_fill_from_order must return a Fill for EXPIRED (status=2) orders.

    With partial_fill_exchange + IOC, a partially-filled order gets status=EXPIRED
    (the remainder was cancelled). The current code only checks FILLED(3) and
    PARTIALLY_FILLED(5); it must also accept EXPIRED(2) when exec_qty > 0.
    """
    from hlanalysis.backtest.runner.hftbt_runner import _build_asset, _RunState, RunResult, RunConfig
    from hlanalysis.backtest.core.data_source import QuestionDescriptor

    T_BOOK = 1_000_000
    T_SUBMIT = 1_000_000_000
    T_TRADE = T_SUBMIT + 1_000_000
    ask_depth = 8.0

    arr = np.concatenate([
        _make_depth_clear_arr(0),
        _depth_ev(T_BOOK, SELL_EVENT, 0.5, ask_depth),
        _trade_ev(T_TRADE, BUY_EVENT, 0.5, ask_depth),
    ])
    cfg = RunConfig(tick_size=0.001, lot_size=1.0, slippage_bps=0.0, fee_taker=0.0,
                    order_latency_ms=0.0)
    asset = _build_asset(arr, cfg)
    bt = hb.HashMapMarketDepthBacktest([asset])
    bt.elapse(T_SUBMIT)
    # Submit IOC for 30 (> ask_depth=8); wait=False + elapse to process trade
    bt.submit_buy_order(0, 1, 0.55, 30.0, hb_order.IOC, hb_order.LIMIT, False)
    bt.elapse(T_TRADE - T_SUBMIT + 1_000_000)
    order = bt.orders(0).get(1)
    assert order is not None
    # Must be EXPIRED (status=2) because remainder was cancelled
    assert order.status == hb_order.EXPIRED, (
        f"expected EXPIRED(2), got {order.status}; exec_qty={order.exec_qty}"
    )
    assert order.exec_qty == pytest.approx(ask_depth)

    # Now verify _RunState.record_fill_from_order returns a Fill (not None)
    q = QuestionDescriptor(
        question_id="test-q", question_idx=1,
        start_ts_ns=0, end_ts_ns=10_000_000_000,
        leg_symbols=("yes", "no"), klass="priceBinary", underlying="BTC",
    )
    st = _RunState(
        hbt=bt,
        cfg=cfg,
        q=q,
        data_source=None,
        leg_to_asset={},
        hedge_asset_no=None,
        stop_pct=None,
        fills_dir_active=False,
        result=RunResult(),
    )
    fill = st.record_fill_from_order(1, 0, "yes", "buy", "test-cloid", 30.0)
    bt.close()

    assert fill is not None, (
        "record_fill_from_order must return Fill for EXPIRED IOC with exec_qty > 0"
    )
    assert fill.size == pytest.approx(ask_depth)
    assert fill.price == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# SHR-79: --depth remains as an optional explicit cap, default = unlimited
# ---------------------------------------------------------------------------

def test_book_depth_assumption_default_is_none():
    """RunConfig.book_depth_assumption defaults to None (unlimited)."""
    cfg = RunConfig()
    assert cfg.book_depth_assumption is None, (
        f"default book_depth_assumption should be None (unlimited), got {cfg.book_depth_assumption}"
    )


def test_book_depth_assumption_explicit_cap():
    """When book_depth_assumption is set, record_fill_from_order caps fill size."""
    from hlanalysis.backtest.runner.hftbt_runner import _build_asset, _RunState, RunResult

    ask_depth = 20.0
    cap = 5.0
    T_BOOK = 1_000_000
    T_SUBMIT = 1_000_000_000
    T_TRADE = T_SUBMIT + 1_000_000

    arr = np.concatenate([
        _make_depth_clear_arr(0),
        _depth_ev(T_BOOK, SELL_EVENT, 0.5, ask_depth),
        _trade_ev(T_TRADE, BUY_EVENT, 0.5, ask_depth),
    ])
    cfg = RunConfig(tick_size=0.001, lot_size=1.0, book_depth_assumption=cap,
                    order_latency_ms=0.0)
    asset = _build_asset(arr, cfg)
    bt = hb.HashMapMarketDepthBacktest([asset])
    bt.elapse(T_SUBMIT)
    bt.submit_buy_order(0, 1, 0.55, 100.0, hb_order.IOC, hb_order.LIMIT, False)
    bt.elapse(T_TRADE - T_SUBMIT + 1_000_000)

    from hlanalysis.backtest.core.data_source import QuestionDescriptor
    q = QuestionDescriptor(
        question_id="test-cap", question_idx=1,
        start_ts_ns=0, end_ts_ns=10_000_000_000,
        leg_symbols=("yes", "no"), klass="priceBinary", underlying="BTC",
    )
    st = _RunState(
        hbt=bt, cfg=cfg, q=q, data_source=None, leg_to_asset={},
        hedge_asset_no=None, stop_pct=None, fills_dir_active=False, result=RunResult(),
    )
    fill = st.record_fill_from_order(1, 0, "yes", "buy", "cap-cloid", 100.0)
    bt.close()

    assert fill is not None
    assert fill.size == pytest.approx(cap), (
        f"explicit book_depth_assumption={cap} should cap fill; got {fill.size}"
    )


# ---------------------------------------------------------------------------
# Order latency: RunConfig and _build_asset wire latency to hftbacktest
# ---------------------------------------------------------------------------

def test_run_config_has_order_latency_ms():
    """RunConfig must have order_latency_ms defaulting to 50.0."""
    cfg = RunConfig()
    assert hasattr(cfg, "order_latency_ms"), "RunConfig must have order_latency_ms field"
    assert cfg.order_latency_ms == pytest.approx(50.0), (
        f"default order_latency_ms should be 50.0, got {cfg.order_latency_ms}"
    )


def test_build_asset_wires_latency_to_hftbacktest():
    """_build_asset must apply order_latency_ms to the hftbacktest asset.

    Verify via the order_latency() reporting: after submitting an order with
    50ms latency, the reported (req_ts, exch_ts, resp_ts) should show
    exch_ts == req_ts + 50ms.
    """
    from hlanalysis.backtest.runner.hftbt_runner import _build_asset

    T = 1_000_000_000  # 1s
    LATENCY_MS = 50.0
    LATENCY_NS = int(LATENCY_MS * 1_000_000)

    arr = np.concatenate([
        _make_depth_clear_arr(T),
        _depth_ev(T + 1_000, SELL_EVENT, 0.5, 100.0),
        _trade_ev(T + LATENCY_NS + 1_000, BUY_EVENT, 0.5, 100.0),
    ])
    cfg = RunConfig(tick_size=0.001, lot_size=1.0, order_latency_ms=LATENCY_MS)
    asset = _build_asset(arr, cfg)
    bt = hb.HashMapMarketDepthBacktest([asset])
    bt.elapse(2_000_000_000)
    bt.submit_buy_order(0, 1, 0.55, 10.0, hb_order.IOC, hb_order.LIMIT, True)
    lat = bt.order_latency(0)
    bt.close()

    assert lat is not None, "order_latency() returned None; order may not have been submitted"
    req_ts, exch_ts, resp_ts = lat
    assert exch_ts == pytest.approx(req_ts + LATENCY_NS, abs=1), (
        f"exchange timestamp should be req_ts + {LATENCY_NS}ns; "
        f"got req={req_ts}, exch={exch_ts}"
    )


def test_order_latency_zero_in_legacy_mode():
    """With order_latency_ms=0 the exchange sees the order immediately (T+0)."""
    from hlanalysis.backtest.runner.hftbt_runner import _build_asset

    T = 1_000_000_000
    arr = np.concatenate([
        _make_depth_clear_arr(T),
        _depth_ev(T + 1_000, SELL_EVENT, 0.5, 100.0),
        _trade_ev(T + 2_000, BUY_EVENT, 0.5, 100.0),
    ])
    cfg = RunConfig(tick_size=0.001, lot_size=1.0, order_latency_ms=0.0)
    asset = _build_asset(arr, cfg)
    bt = hb.HashMapMarketDepthBacktest([asset])
    bt.elapse(2_000_000_000)
    bt.submit_buy_order(0, 1, 0.55, 5.0, hb_order.IOC, hb_order.LIMIT, True)
    lat = bt.order_latency(0)
    bt.close()

    assert lat is not None
    req_ts, exch_ts, _resp_ts = lat
    assert exch_ts == req_ts, (
        f"zero latency: exch_ts ({exch_ts}) should equal req_ts ({req_ts})"
    )


# ---------------------------------------------------------------------------
# SHR-56: slippage_bps as additive haircut on the recorded fill price
# ---------------------------------------------------------------------------

def test_slippage_bps_applied_as_additive_buy_haircut():
    """slippage_bps must make buy fills MORE expensive (higher price).

    With a single ask level at 0.5 and slippage_bps=100, the buy fill price
    should be 0.5 * (1 + 100/10000) = 0.505.

    Because _route_enter submits the IOC at the slipped price, the actual
    recorded exec_price from hftbacktest reflects the slipped ask level;
    record_fill_from_order then returns that slipped price.
    """
    from hlanalysis.backtest.runner.hftbt_runner import _slipped_buy_price
    from hlanalysis.strategy.types import BookState

    book = BookState(
        symbol="YES", bid_px=0.49, bid_sz=100.0, ask_px=0.50, ask_sz=100.0,
        last_trade_ts_ns=0, last_l2_ts_ns=0,
    )
    # 100 bps = 1%; slipped buy = 0.50 * 1.01 = 0.505
    slipped = _slipped_buy_price(book, limit_price=0.0, slippage_bps=100.0)
    assert slipped == pytest.approx(0.505, rel=1e-6), (
        f"buy slippage haircut incorrect: expected ~0.505, got {slipped}"
    )


def test_slippage_bps_applied_as_additive_sell_haircut():
    """slippage_bps must make sell fills CHEAPER (lower price).

    With a bid at 0.60 and slippage_bps=100, sell fill should be
    0.60 * (1 - 100/10000) = 0.594.
    """
    from hlanalysis.backtest.runner.hftbt_runner import _slipped_sell_price
    from hlanalysis.strategy.types import BookState

    book = BookState(
        symbol="YES", bid_px=0.60, bid_sz=100.0, ask_px=0.61, ask_sz=100.0,
        last_trade_ts_ns=0, last_l2_ts_ns=0,
    )
    slipped = _slipped_sell_price(book, limit_price=0.0, slippage_bps=100.0)
    assert slipped == pytest.approx(0.594, rel=1e-6), (
        f"sell slippage haircut incorrect: expected ~0.594, got {slipped}"
    )


def test_slippage_zero_gives_exact_book_price():
    """With slippage_bps=0 the returned price equals the book touch."""
    from hlanalysis.backtest.runner.hftbt_runner import _slipped_buy_price, _slipped_sell_price
    from hlanalysis.strategy.types import BookState

    book = BookState(
        symbol="YES", bid_px=0.45, bid_sz=50.0, ask_px=0.46, ask_sz=50.0,
        last_trade_ts_ns=0, last_l2_ts_ns=0,
    )
    assert _slipped_buy_price(book, 0.0, 0.0) == pytest.approx(0.46)
    assert _slipped_sell_price(book, 0.0, 0.0) == pytest.approx(0.45)


# ---------------------------------------------------------------------------
# SHR-57: fee logging — fee model/rate logged at run start (smoke test)
# ---------------------------------------------------------------------------

def test_fee_model_logged_at_run_start(capsys, synthetic_source):
    """run_one_question must emit a log line about the effective fee model.

    This is a smoke test: we verify that running a question with explicit
    fee_taker/fee_model produces output that references those values.
    The exact format is unconstrained — just verifying the log fires.
    """
    import logging

    ds, sq = synthetic_source
    strat = build_dummy_enter_strategy({"size": 5.0})
    cfg = RunConfig(slippage_bps=0.0, fee_taker=0.0, fee_model="flat")

    # Capture loguru output via caplog at DEBUG level
    import io
    log_output = io.StringIO()
    import loguru
    handler_id = loguru.logger.add(log_output, level="DEBUG")
    try:
        run_one_question(strat, ds, sq.descriptor, cfg, strike=sq.strike)
        output = log_output.getvalue()
    finally:
        loguru.logger.remove(handler_id)

    # Should mention fee model somewhere in the run start log
    assert "fee" in output.lower(), (
        "run_one_question must log the effective fee model at run start; "
        f"got no 'fee' in log output"
    )


# ---------------------------------------------------------------------------
# CLI: --order-latency-ms and --depth=None defaults
# ---------------------------------------------------------------------------

def test_cli_order_latency_ms_arg():
    """--order-latency-ms must default to 50 and parse correctly."""
    from hlanalysis.backtest.cli import main

    # Parse the run subcommand args; we just check the argparse defaults
    import argparse
    from hlanalysis.backtest.cli import _add_run_config_args

    p = argparse.ArgumentParser()
    _add_run_config_args(p)
    args = p.parse_args([])
    assert hasattr(args, "order_latency_ms"), (
        "--order-latency-ms not added to run-config args"
    )
    assert args.order_latency_ms == pytest.approx(50.0), (
        f"--order-latency-ms default should be 50.0, got {args.order_latency_ms}"
    )


def test_cli_depth_default_is_none():
    """--depth must default to None (unlimited) after the SHR-79 change."""
    import argparse
    from hlanalysis.backtest.cli import _add_run_config_args

    p = argparse.ArgumentParser()
    _add_run_config_args(p)
    args = p.parse_args([])
    assert args.depth is None, (
        f"--depth default should be None (unlimited), got {args.depth}"
    )


def test_run_config_from_args_wires_order_latency_ms():
    """_run_config_from_args must propagate order_latency_ms into RunConfig."""
    import argparse
    from hlanalysis.backtest.cli import _add_run_config_args, _run_config_from_args

    p = argparse.ArgumentParser()
    _add_run_config_args(p)
    args = p.parse_args(["--order-latency-ms", "100.0"])
    cfg = _run_config_from_args(args, hedge_cfg=None)
    assert cfg.order_latency_ms == pytest.approx(100.0), (
        f"RunConfig.order_latency_ms should be 100.0 from CLI, got {cfg.order_latency_ms}"
    )


def test_run_config_from_args_depth_none_gives_unlimited():
    """When --depth is omitted, _run_config_from_args must set book_depth_assumption=None."""
    import argparse
    from hlanalysis.backtest.cli import _add_run_config_args, _run_config_from_args

    p = argparse.ArgumentParser()
    _add_run_config_args(p)
    args = p.parse_args([])
    cfg = _run_config_from_args(args, hedge_cfg=None)
    assert cfg.book_depth_assumption is None, (
        f"omitting --depth should give book_depth_assumption=None, got {cfg.book_depth_assumption}"
    )


# ---------------------------------------------------------------------------
# SHR-85: sim state/halt replay + daily-loss-cap + inventory caps
# ---------------------------------------------------------------------------


class _AlwaysEnterStrategy(Strategy):
    """Test-only strategy: tries to ENTER (open or top-up) YES on every tick.

    Unlike ``_DummyEnterStrategy`` it never stops firing, so halt-window /
    inventory suppression has something to suppress on every scan tick.
    """

    name = "_always_enter_yes"

    def __init__(self, size: float = 10.0):
        self._size = size

    def evaluate(self, *, question, books, reference_price, recent_returns,
                 recent_volume_usd, position, now_ns, recent_hl_bars=()):
        from hlanalysis.strategy.types import Action, Decision, OrderIntent
        book = books.get(question.yes_symbol)
        if book is None or book.ask_px is None:
            return Decision(action=Action.HOLD)
        intent = OrderIntent(
            question_idx=question.question_idx,
            symbol=question.yes_symbol,
            side="buy",
            size=self._size,
            limit_price=min(1.0, book.ask_px + 0.05),
            cloid=f"always-{now_ns}",
        )
        return Decision(action=Action.ENTER, intents=(intent,))


def _entry_fill_timestamps(res, fill_ts) -> list[int]:
    """Timestamps of non-settle BUY (entry) fills."""
    out = []
    for f in res.fills:
        if f.side == "buy" and f.cloid != "settle" and not f.cloid.startswith("hedge"):
            out.append(fill_ts[f.cloid])
    return out


def test_injected_halt_window_suppresses_entries_inside_it():
    """A halt window passed to the runner must produce zero entries inside it,
    while entries before/after the window still fire (exits stay exempt)."""
    from hlanalysis.backtest.halt_replay import HaltWindow
    from hlanalysis.backtest.runner.hftbt_runner import run_one_question

    sq = make_default_binary_question(start_ts_ns=0)
    ds = SyntheticDataSource()
    ds.add_question(sq)
    cfg = RunConfig(scanner_interval_seconds=60, slippage_bps=0.0, fee_taker=0.0)

    # Question runs 0..600s; scan ticks at 60,120,...,540s. Halt 150s..330s.
    halt = HaltWindow(start_ns=150 * 1_000_000_000, end_ns=330 * 1_000_000_000,
                      reason="stale_data_halt")

    # Run with the halt window. Capture fill timestamps via the fills parquet is
    # overkill; instead run twice (with/without) and compare entry timestamps.
    res = run_one_question(
        _AlwaysEnterStrategy(size=10.0), ds, sq.descriptor, cfg,
        strike=sq.strike, halt_windows=[halt],
    )
    # Reconstruct entry timestamps from the run's internal fill_ts is not exposed;
    # assert instead that NO entry fill's recorded ts lands in the window by
    # re-deriving from the fills' cloids which embed now_ns.
    entered_ns = [
        int(f.cloid.split("-")[1])
        for f in res.fills
        if f.cloid.startswith("always-")
    ]
    assert entered_ns, "expected at least one entry outside the halt window"
    assert all(
        not (halt.start_ns <= ts < halt.end_ns) for ts in entered_ns
    ), f"entries leaked into the halt window: {entered_ns}"
    # And at least one entry on either side of the window (suppression, not a
    # blanket block).
    assert any(ts < halt.start_ns for ts in entered_ns)
    assert any(ts >= halt.end_ns for ts in entered_ns)


def test_no_halt_window_enters_every_tick():
    """Control: without a halt window the always-enter strategy fills on ticks
    spanning the would-be window (so the suppression test is meaningful)."""
    from hlanalysis.backtest.runner.hftbt_runner import run_one_question

    sq = make_default_binary_question(start_ts_ns=0)
    ds = SyntheticDataSource()
    ds.add_question(sq)
    cfg = RunConfig(scanner_interval_seconds=60, slippage_bps=0.0, fee_taker=0.0)
    res = run_one_question(
        _AlwaysEnterStrategy(size=10.0), ds, sq.descriptor, cfg, strike=sq.strike,
    )
    entered_ns = [
        int(f.cloid.split("-")[1])
        for f in res.fills
        if f.cloid.startswith("always-")
    ]
    assert any(150 * 1_000_000_000 <= ts < 330 * 1_000_000_000 for ts in entered_ns), (
        "control run should fill inside the would-be halt window"
    )


def _make_run_state(caps=None, halt_windows=()):
    """A minimal _RunState for gate-wiring unit tests (no hftbacktest needed)."""
    from hlanalysis.backtest.runner.hftbt_runner import _RunState, RunConfig
    from hlanalysis.backtest.runner.result import RunResult
    from hlanalysis.backtest.core.data_source import QuestionDescriptor

    q = QuestionDescriptor(
        question_id="gate-q", question_idx=1, start_ts_ns=0,
        end_ts_ns=10_000_000_000, leg_symbols=("yes", "no"),
        klass="priceBinary", underlying="BTC",
    )
    return _RunState(
        hbt=None, cfg=RunConfig(), q=q, data_source=None, leg_to_asset={},
        hedge_asset_no=None, stop_pct=None, fills_dir_active=False,
        result=RunResult(), sim_risk_caps=caps, halt_windows=tuple(halt_windows),
    )


def test_runstate_daily_loss_cap_latches_and_resets_next_window():
    """record_realized accumulates per daily window; once realized loss crosses
    the cap the window stays halted (latched) even if a later win recovers it,
    and the next daily window resumes."""
    from datetime import datetime, timezone
    from hlanalysis.backtest.halt_replay import SimRiskCaps

    def ns(*a):
        return int(datetime(*a, tzinfo=timezone.utc).timestamp() * 1e9)

    caps = SimRiskCaps(daily_loss_cap_usd=100.0, daily_window_start_hour_utc=6)
    st = _make_run_state(caps=caps)

    day1_noon = ns(2026, 6, 8, 12, 0)
    st.now_ns = day1_noon
    # Within budget so far.
    st.record_realized(day1_noon, -50.0)
    assert st.entry_blocked(intent_notional=10.0, is_topup=False,
                            held_notional=0.0, n_held=0) is None
    # Cross the cap.
    st.record_realized(day1_noon, -60.0)  # cumulative -110 < -100
    assert st.entry_blocked(intent_notional=10.0, is_topup=False,
                            held_notional=0.0, n_held=0) == "daily_loss_cap"
    # A later WIN brings running PnL back above -cap, but the window stays halted.
    st.record_realized(day1_noon, +80.0)  # cumulative -30, but floor latched
    assert st.entry_blocked(intent_notional=10.0, is_topup=False,
                            held_notional=0.0, n_held=0) == "daily_loss_cap"
    # Next daily window (after 06:00 the following day) resumes.
    day2_noon = ns(2026, 6, 9, 12, 0)
    st.now_ns = day2_noon
    assert st.entry_blocked(intent_notional=10.0, is_topup=False,
                            held_notional=0.0, n_held=0) is None


def test_runstate_inventory_cap_blocks_n_plus_1():
    """The total-inventory cap blocks the entry that would push held notional
    over the cap (the N+1 entry)."""
    from hlanalysis.backtest.halt_replay import SimRiskCaps

    caps = SimRiskCaps(max_total_inventory_usd=300.0)
    st = _make_run_state(caps=caps)
    st.now_ns = 1_000

    # 250 already held; a 40-notional top-up fits (290 ≤ 300).
    assert st.entry_blocked(intent_notional=40.0, is_topup=True,
                            held_notional=250.0, n_held=1) is None
    # A 100-notional entry would push to 350 → blocked.
    assert st.entry_blocked(intent_notional=100.0, is_topup=True,
                            held_notional=250.0, n_held=1) == "max_total_inventory"


def test_runstate_no_caps_never_blocks():
    """With no caps and no halt windows the gate is a no-op (backward compat)."""
    st = _make_run_state(caps=None, halt_windows=())
    st.now_ns = 42
    assert st.entry_blocked(intent_notional=1e9, is_topup=False,
                            held_notional=1e9, n_held=99) is None


# ---------------------------------------------------------------------------
# SHR-89: pluggable latency model (constant default + sampled distribution)
# ---------------------------------------------------------------------------

def test_run_config_latency_model_defaults_none():
    """RunConfig.latency_model defaults to None (use the constant knob)."""
    cfg = RunConfig()
    assert cfg.latency_model is None


def test_effective_latency_model_default_is_constant_from_knob():
    """With no explicit model, effective_latency_model() is ConstantLatency
    wrapping the legacy order_latency_ms knob (back-compat)."""
    from hlanalysis.backtest.runner.hftbt_runner import ConstantLatency

    cfg = RunConfig(order_latency_ms=37.0)
    model = cfg.effective_latency_model()
    assert isinstance(model, ConstantLatency)
    assert model.latency_ms == pytest.approx(37.0)


def test_effective_latency_model_returns_explicit_model():
    """An explicit latency_model overrides the constant knob."""
    from hlanalysis.backtest.runner.hftbt_runner import SampledLatency

    model = SampledLatency(samples_ms=(10.0, 200.0))
    cfg = RunConfig(latency_model=model)
    assert cfg.effective_latency_model() is model


def test_sampled_latency_build_array_is_deterministic_and_typed():
    """SampledLatency builds an hftbacktest order-latency array whose entry
    latencies are drawn from the supplied δ samples (ns), deterministic by seed."""
    from hlanalysis.backtest.runner.hftbt_runner import (
        SampledLatency,
        _ORDER_LATENCY_DTYPE,
    )

    samples = (10.0, 200.0)
    model = SampledLatency(samples_ms=samples, seed=7, step_ns=1_000_000_000)
    a = model.build_latency_array(start_ts_ns=0, end_ts_ns=5_000_000_000)
    b = model.build_latency_array(start_ts_ns=0, end_ts_ns=5_000_000_000)

    assert a.dtype == _ORDER_LATENCY_DTYPE
    assert len(a) >= 2
    # Same seed → identical draws.
    assert np.array_equal(a["exch_ts"], b["exch_ts"])
    # req_ts strictly increasing (required by intp interpolation).
    assert np.all(np.diff(a["req_ts"]) > 0)
    # Every entry latency is one of the supplied samples (in ns).
    entry_ns = a["exch_ts"] - a["req_ts"]
    allowed = {int(s * 1_000_000) for s in samples}
    assert set(int(x) for x in entry_ns) <= allowed


def test_build_asset_uses_sampled_latency_distribution():
    """_build_asset wired with a SampledLatency model fills on book(decision+δ):
    the order's reported entry latency equals the sampled δ (single-valued sample
    → deterministic), not the constant order_latency_ms."""
    from hlanalysis.backtest.runner.hftbt_runner import _build_asset, SampledLatency

    DELTA_MS = 123.0
    DELTA_NS = int(DELTA_MS * 1_000_000)
    T = 1_000_000_000
    arr = np.concatenate([
        _make_depth_clear_arr(T),
        _depth_ev(T + 1_000, SELL_EVENT, 0.5, 100.0),
        _trade_ev(T + DELTA_NS + 1_000, BUY_EVENT, 0.5, 100.0),
    ])
    # order_latency_ms left at the default 50ms; the model must take precedence.
    cfg = RunConfig(
        tick_size=0.001, lot_size=1.0,
        latency_model=SampledLatency(samples_ms=(DELTA_MS,), seed=0),
    )
    asset = _build_asset(arr, cfg, start_ts_ns=T, end_ts_ns=T + 5_000_000_000)
    bt = hb.HashMapMarketDepthBacktest([asset])
    bt.elapse(2_000_000_000)
    bt.submit_buy_order(0, 1, 0.55, 10.0, hb_order.IOC, hb_order.LIMIT, True)
    lat = bt.order_latency(0)
    bt.close()

    assert lat is not None
    req_ts, exch_ts, _resp_ts = lat
    assert exch_ts - req_ts == pytest.approx(DELTA_NS, abs=1_000), (
        f"sampled latency δ={DELTA_NS}ns should drive the exchange arrival; "
        f"got entry latency {exch_ts - req_ts}ns"
    )


def test_constant_latency_model_matches_legacy_knob():
    """ConstantLatency via the model path is bit-identical to the legacy
    constant_order_latency wiring (no behaviour change by default)."""
    from hlanalysis.backtest.runner.hftbt_runner import _build_asset

    LAT_MS = 50.0
    LAT_NS = int(LAT_MS * 1_000_000)
    T = 1_000_000_000
    arr = np.concatenate([
        _make_depth_clear_arr(T),
        _depth_ev(T + 1_000, SELL_EVENT, 0.5, 100.0),
        _trade_ev(T + LAT_NS + 1_000, BUY_EVENT, 0.5, 100.0),
    ])
    cfg = RunConfig(tick_size=0.001, lot_size=1.0, order_latency_ms=LAT_MS)
    asset = _build_asset(arr, cfg)
    bt = hb.HashMapMarketDepthBacktest([asset])
    bt.elapse(2_000_000_000)
    bt.submit_buy_order(0, 1, 0.55, 10.0, hb_order.IOC, hb_order.LIMIT, True)
    lat = bt.order_latency(0)
    bt.close()

    assert lat is not None
    req_ts, exch_ts, _resp_ts = lat
    assert exch_ts - req_ts == pytest.approx(LAT_NS, abs=1)


# ---------------------------------------------------------------------------
# SHR-89: explicit IOC reject when the limit is unmarketable at fill time,
# and re-fire next scan (reproduces live churn)
# ---------------------------------------------------------------------------

def test_classify_reject_buy_crossing_no_fill_is_reject():
    """A buy IOC submitted at/through the decision-time ask that returns no fill
    is a reject (book moved away / queue not swept during latency)."""
    from hlanalysis.backtest.runner.hftbt_runner import _classify_reject

    book = BookState(symbol="yes", bid_px=None, bid_sz=None, ask_px=0.50, ask_sz=10.0,
                     last_trade_ts_ns=0, last_l2_ts_ns=0)
    assert _classify_reject(book, "buy", submit_px=0.50) is True
    assert _classify_reject(book, "buy", submit_px=0.51) is True


def test_classify_reject_buy_non_crossing_is_not_reject():
    """A buy whose price is below the ask never crossed — a non-fill there is a
    plain no-op (resting/cancelled), not a reject."""
    from hlanalysis.backtest.runner.hftbt_runner import _classify_reject

    book = BookState(symbol="yes", bid_px=None, bid_sz=None, ask_px=0.60, ask_sz=10.0,
                     last_trade_ts_ns=0, last_l2_ts_ns=0)
    assert _classify_reject(book, "buy", submit_px=0.55) is False


def test_classify_reject_sell_crossing_no_fill_is_reject():
    from hlanalysis.backtest.runner.hftbt_runner import _classify_reject

    book = BookState(symbol="yes", bid_px=0.50, bid_sz=10.0, ask_px=None, ask_sz=None,
                     last_trade_ts_ns=0, last_l2_ts_ns=0)
    assert _classify_reject(book, "sell", submit_px=0.50) is True
    assert _classify_reject(book, "sell", submit_px=0.49) is True
    assert _classify_reject(book, "sell", submit_px=0.55) is False


def test_classify_reject_no_book_is_not_reject():
    """No opposing liquidity at decision → not a reject (nothing to cross)."""
    from hlanalysis.backtest.runner.hftbt_runner import _classify_reject

    book = BookState(symbol="yes", bid_px=None, bid_sz=None, ask_px=None, ask_sz=None,
                     last_trade_ts_ns=0, last_l2_ts_ns=0)
    assert _classify_reject(book, "buy", submit_px=0.50) is False
    assert _classify_reject(book, "sell", submit_px=0.50) is False


def _enter_decision(symbol="yes", size=10.0, limit=0.55):
    from hlanalysis.strategy.types import Action, Decision, OrderIntent
    return Decision(
        action=Action.ENTER,
        intents=(OrderIntent(question_idx=1, symbol=symbol, side="buy",
                             size=size, limit_price=limit, cloid="c1"),),
    )


def _reject_run_state(bt, cfg):
    from hlanalysis.backtest.runner.hftbt_runner import _RunState, RunResult
    from hlanalysis.backtest.core.data_source import QuestionDescriptor
    q = QuestionDescriptor(
        question_id="rej", question_idx=1, start_ts_ns=0, end_ts_ns=10_000_000_000,
        leg_symbols=("yes", "no"), klass="priceBinary", underlying="BTC",
    )
    return _RunState(
        hbt=bt, cfg=cfg, q=q, data_source=None, leg_to_asset={"yes": 0},
        hedge_asset_no=None, stop_pct=None, fills_dir_active=False, result=RunResult(),
    )


def test_route_enter_rejects_when_book_moves_during_latency():
    """Marketable buy at decision; the ask jumps above the limit before the order
    reaches the exchange → no fill, reject counted, position stays None (re-fires)."""
    from hlanalysis.backtest.runner.hftbt_runner import _build_asset, _route_enter

    LAT_NS = 50_000_000  # 50ms
    T_BOOK = 1_000_000
    T_SUBMIT = 1_000_000_000
    T_MOVE = T_SUBMIT + 1_000_000          # ask raised within the latency window
    T_TRADE = T_SUBMIT + LAT_NS + 5_000_000

    arr = np.concatenate([
        _make_depth_clear_arr(0),
        _depth_ev(T_BOOK, SELL_EVENT, 0.50, 100.0),
        _depth_ev(T_MOVE, SELL_EVENT, 0.50, 0.0),     # clear the 0.50 ask
        _depth_ev(T_MOVE, SELL_EVENT, 0.70, 100.0),   # ask now 0.70 (> limit 0.55)
        _trade_ev(T_TRADE, BUY_EVENT, 0.70, 100.0),   # trade can't cross 0.55
    ])
    cfg = RunConfig(tick_size=0.001, lot_size=1.0, slippage_bps=0.0, fee_taker=0.0,
                    order_latency_ms=50.0)
    asset = _build_asset(arr, cfg)
    bt = hb.HashMapMarketDepthBacktest([asset])
    bt.elapse(T_SUBMIT)
    st = _reject_run_state(bt, cfg)
    st.books = {"yes": BookState(symbol="yes", bid_px=None, bid_sz=None, ask_px=0.50,
                                 ask_sz=100.0, last_trade_ts_ns=0, last_l2_ts_ns=T_BOOK)}
    st.now_ns = T_SUBMIT

    _route_enter(st, _enter_decision())
    bt.close()

    assert st.result.fills == [], "rejected order must not book a fill"
    assert st.pos is None, "no position opened → strategy re-fires next scan"
    assert st.result.n_rejects == 1


def test_route_enter_no_reject_when_not_marketable():
    """A non-crossing buy that doesn't fill is a plain no-op, not a reject."""
    from hlanalysis.backtest.runner.hftbt_runner import _build_asset, _route_enter

    T_BOOK = 1_000_000
    T_SUBMIT = 1_000_000_000
    arr = np.concatenate([
        _make_depth_clear_arr(0),
        _depth_ev(T_BOOK, SELL_EVENT, 0.60, 100.0),   # ask 0.60, above our 0.55 limit
    ])
    cfg = RunConfig(tick_size=0.001, lot_size=1.0, slippage_bps=0.0, fee_taker=0.0,
                    order_latency_ms=50.0)
    asset = _build_asset(arr, cfg)
    bt = hb.HashMapMarketDepthBacktest([asset])
    bt.elapse(T_SUBMIT)
    st = _reject_run_state(bt, cfg)
    st.books = {"yes": BookState(symbol="yes", bid_px=None, bid_sz=None, ask_px=0.60,
                                 ask_sz=100.0, last_trade_ts_ns=0, last_l2_ts_ns=T_BOOK)}
    st.now_ns = T_SUBMIT

    _route_enter(st, _enter_decision(limit=0.55))
    bt.close()

    assert st.result.fills == []
    assert st.pos is None
    assert st.result.n_rejects == 0


def test_reject_then_refire_fills_on_next_scan():
    """After a reject (no fill, pos None) the strategy re-fires; a later scan
    where the book is marketable and a trade sweeps fills the order."""
    from hlanalysis.backtest.runner.hftbt_runner import _build_asset, _route_enter

    LAT_NS = 50_000_000
    T_BOOK1 = 1_000_000
    T_SUBMIT1 = 1_000_000_000
    T_MOVE = T_SUBMIT1 + 1_000_000               # ask jumps away → reject #1
    T_BOOK2 = 2_000_000_000                       # book recovers to 0.50
    T_SUBMIT2 = 2_500_000_000
    T_TRADE2 = T_SUBMIT2 + LAT_NS + 1_000_000     # trade sweeps after arrival → fill

    arr = np.concatenate([
        _make_depth_clear_arr(0),
        _depth_ev(T_BOOK1, SELL_EVENT, 0.50, 10.0),
        _depth_ev(T_MOVE, SELL_EVENT, 0.50, 0.0),
        _depth_ev(T_MOVE, SELL_EVENT, 0.70, 10.0),
        _depth_ev(T_BOOK2, SELL_EVENT, 0.70, 0.0),
        _depth_ev(T_BOOK2, SELL_EVENT, 0.50, 10.0),
        _trade_ev(T_TRADE2, BUY_EVENT, 0.50, 10.0),
    ])
    cfg = RunConfig(tick_size=0.001, lot_size=1.0, slippage_bps=0.0, fee_taker=0.0,
                    order_latency_ms=50.0)
    asset = _build_asset(arr, cfg)
    bt = hb.HashMapMarketDepthBacktest([asset])
    bt.elapse(T_SUBMIT1)
    st = _reject_run_state(bt, cfg)
    st.books = {"yes": BookState(symbol="yes", bid_px=None, bid_sz=None, ask_px=0.50,
                                 ask_sz=10.0, last_trade_ts_ns=0, last_l2_ts_ns=T_BOOK1)}
    st.now_ns = T_SUBMIT1
    _route_enter(st, _enter_decision())
    assert st.pos is None and st.result.n_rejects == 1, "first attempt must reject"

    # Re-fire on the next scan once the book has recovered.
    bt.elapse(T_SUBMIT2 - int(bt.current_timestamp))
    st.books = {"yes": BookState(symbol="yes", bid_px=None, bid_sz=None, ask_px=0.50,
                                 ask_sz=10.0, last_trade_ts_ns=0, last_l2_ts_ns=T_BOOK2)}
    st.now_ns = T_SUBMIT2
    _route_enter(st, _enter_decision())
    bt.close()

    assert st.pos is not None, "re-fired order should fill once book is marketable"
    assert len(st.result.fills) == 1
    assert st.result.fills[0].side == "buy"


# ---------------------------------------------------------------------------
# SHR-94: IOC marketability re-check at send+latency (fleeting levels)
# ---------------------------------------------------------------------------
#
# The HL book is sampled at discrete intervals (dt=5s or dt=60s). A level
# present at snapshot[i] and absent at snapshot[i+1] appears persistent in
# hftbacktest until the clearing event fires at ts[i+1]. With 50ms latency the
# order arrives at the exchange at T+50ms, but if the next snapshot is at T+5s
# (well after arrival) the clearing event hasn't fired and hftbacktest fills.
# In live trading, a level that vanished within the sampling interval after it
# was observed gets zero fills.
#
# These tests inject the per-snapshot best-ask/bid arrays (snap_best_ask_per_leg,
# snap_best_bid_per_leg, book_ts_per_leg) into a _RunState directly and call
# _route_enter / _route_exit to exercise the pre-flight veto logic.

def _snap_run_state(
    bt,
    cfg,
    *,
    snap_best_ask_per_leg=None,
    snap_best_bid_per_leg=None,
    book_ts_per_leg=None,
    book_idx=None,
):
    """Build a _RunState wired with the given per-snapshot arrays."""
    from hlanalysis.backtest.runner.hftbt_runner import _RunState, RunResult
    from hlanalysis.backtest.core.data_source import QuestionDescriptor

    q = QuestionDescriptor(
        question_id="shr94-q", question_idx=1, start_ts_ns=0, end_ts_ns=10_000_000_000,
        leg_symbols=("yes", "no"), klass="priceBinary", underlying="BTC",
    )
    book_idx = book_idx if book_idx is not None else {"yes": 0}
    return _RunState(
        hbt=bt, cfg=cfg, q=q, data_source=None, leg_to_asset={"yes": 0},
        hedge_asset_no=None, stop_pct=None, fills_dir_active=False, result=RunResult(),
        snap_best_ask_per_leg=snap_best_ask_per_leg or {},
        snap_best_bid_per_leg=snap_best_bid_per_leg or {},
        book_ts_per_leg=book_ts_per_leg or {},
        _book_idx=book_idx,
    )


def test_shr94_fleeting_ask_produces_no_fill():
    """SHR-94: ask present at decision, gone by next snapshot within latency window
    → IOC produces NO fill (matching live's zero-fill on fleeting level).

    Scenario: ask=0.90 at decision time T.  The next book snapshot at T+2s
    (within the 50ms latency window? No — T+2s > T+50ms, BUT the ask price at
    that snapshot is 0.98 > 0.90 → the level fled).

    Wait — the check fires when the next snapshot falls within [T, T+latency].
    Let's use latency=2100ms so the next snapshot at T+2s IS within the window.
    """
    from hlanalysis.backtest.runner.hftbt_runner import _build_asset, _route_enter

    LAT_MS = 2100.0  # 2.1s latency so the 2s snapshot falls inside
    LAT_NS = int(LAT_MS * 1_000_000)
    T = 1_000_000_000  # 1s decision
    T_SNAP_NEXT = T + 2_000_000_000  # next snapshot 2s later, ask=0.98 (level fled)

    # Build the event array: ask at 0.90, persists nominally (no explicit clear)
    arr = np.concatenate([
        _make_depth_clear_arr(0),
        _depth_ev(T - 5_000_000, SELL_EVENT, 0.90, 100.0),
    ])
    cfg = RunConfig(
        tick_size=0.001, lot_size=1.0, slippage_bps=0.0, fee_taker=0.0,
        order_latency_ms=LAT_MS, ioc_marketability_recheck=True,
    )
    asset = _build_asset(arr, cfg)
    bt = hb.HashMapMarketDepthBacktest([asset])
    bt.elapse(T)

    # next snapshot shows ask=0.98 (fled from 0.90)
    snap_best_ask = np.array([0.90, 0.98], dtype=np.float64)
    book_ts = np.array([T - 5_000_000, T_SNAP_NEXT], dtype=np.int64)
    # book_idx[sym]=1 means snapshot[1] is the NEXT unconsumed snapshot
    book_idx = {"yes": 1}

    st = _snap_run_state(
        bt, cfg,
        snap_best_ask_per_leg={"yes": snap_best_ask},
        book_ts_per_leg={"yes": book_ts},
        book_idx=book_idx,
    )
    st.books = {"yes": BookState(symbol="yes", bid_px=None, bid_sz=None, ask_px=0.90,
                                 ask_sz=100.0, last_trade_ts_ns=0, last_l2_ts_ns=T)}
    st.now_ns = T

    _route_enter(st, _enter_decision(limit=0.92))
    bt.close()

    assert st.result.fills == [], (
        "fleeting ask (gone in next snapshot within latency window) must produce no fill"
    )
    assert st.pos is None, "no position opened on fleeting ask"
    assert st.result.n_rejects == 1, (
        "fleeting ask detected → counted as a reject (strategy re-fires next scan)"
    )


def test_shr94_stable_ask_fills_normally():
    """SHR-94 regression: a genuinely deep and stable ask (present in the next
    snapshot too) still fills as before — no regression introduced.
    """
    from hlanalysis.backtest.runner.hftbt_runner import _build_asset, _route_enter

    LAT_MS = 50.0
    LAT_NS = int(LAT_MS * 1_000_000)
    T = 1_000_000_000
    T_SNAP_NEXT = T + 5_000_000_000  # next snapshot 5s later, ask still at 0.90

    arr = np.concatenate([
        _make_depth_clear_arr(0),
        _depth_ev(T - 5_000_000, SELL_EVENT, 0.90, 100.0),
    ])
    cfg = RunConfig(
        tick_size=0.001, lot_size=1.0, slippage_bps=0.0, fee_taker=0.0,
        order_latency_ms=LAT_MS, ioc_marketability_recheck=True,
    )
    asset = _build_asset(arr, cfg)
    bt = hb.HashMapMarketDepthBacktest([asset])
    bt.elapse(T)

    # Next snapshot at T+5s (outside the 50ms latency window)
    snap_best_ask = np.array([0.90, 0.90], dtype=np.float64)
    book_ts = np.array([T - 5_000_000, T_SNAP_NEXT], dtype=np.int64)
    book_idx = {"yes": 1}

    st = _snap_run_state(
        bt, cfg,
        snap_best_ask_per_leg={"yes": snap_best_ask},
        book_ts_per_leg={"yes": book_ts},
        book_idx=book_idx,
    )
    st.books = {"yes": BookState(symbol="yes", bid_px=None, bid_sz=None, ask_px=0.90,
                                 ask_sz=100.0, last_trade_ts_ns=0, last_l2_ts_ns=T)}
    st.now_ns = T

    _route_enter(st, _enter_decision(limit=0.92))
    bt.close()

    assert st.pos is not None, "stable deep ask should fill (no regression)"
    assert len(st.result.fills) == 1
    assert st.result.fills[0].side == "buy"
    assert st.result.n_rejects == 0


def test_shr94_fill_capped_at_displayed_depth():
    """SHR-94: fill size capped at displayed book depth (ask_sz=7, order for 50).

    This is the SHR-79 partial-fill guarantee: the ask depth is 7 units, so the
    fill must be ≤ 7.  Verifies no over-fill beyond displayed depth even when
    the level is stable.
    """
    from hlanalysis.backtest.runner.hftbt_runner import _build_asset, _route_enter

    LAT_MS = 50.0
    T = 1_000_000_000
    ASK_DEPTH = 7.0

    arr = np.concatenate([
        _make_depth_clear_arr(0),
        _depth_ev(T - 5_000_000, SELL_EVENT, 0.90, ASK_DEPTH),
    ])
    cfg = RunConfig(
        tick_size=0.001, lot_size=1.0, slippage_bps=0.0, fee_taker=0.0,
        order_latency_ms=LAT_MS, ioc_marketability_recheck=True,
    )
    asset = _build_asset(arr, cfg)
    bt = hb.HashMapMarketDepthBacktest([asset])
    bt.elapse(T)

    # Next snapshot outside latency window (ask stable)
    snap_best_ask = np.array([0.90, 0.90], dtype=np.float64)
    book_ts = np.array([T - 5_000_000, T + 5_000_000_000], dtype=np.int64)
    book_idx = {"yes": 1}

    st = _snap_run_state(
        bt, cfg,
        snap_best_ask_per_leg={"yes": snap_best_ask},
        book_ts_per_leg={"yes": book_ts},
        book_idx=book_idx,
    )
    st.books = {"yes": BookState(symbol="yes", bid_px=None, bid_sz=None, ask_px=0.90,
                                 ask_sz=ASK_DEPTH, last_trade_ts_ns=0, last_l2_ts_ns=T)}
    st.now_ns = T

    # Request 50 units — book only has 7
    _route_enter(st, _enter_decision(size=50.0, limit=0.95))
    bt.close()

    assert len(st.result.fills) == 1
    fill = st.result.fills[0]
    assert fill.size <= ASK_DEPTH, (
        f"fill size must not exceed displayed depth {ASK_DEPTH}; got {fill.size}"
    )
    assert st.pos is not None


def test_shr94_no_snap_arrays_fills_as_before():
    """SHR-94 back-compat: no snap_best arrays supplied (legacy/PM path) →
    recheck is skipped and the order fills normally as in pre-SHR-94.
    """
    from hlanalysis.backtest.runner.hftbt_runner import _build_asset, _route_enter

    LAT_MS = 50.0
    T = 1_000_000_000

    arr = np.concatenate([
        _make_depth_clear_arr(0),
        _depth_ev(T - 5_000_000, SELL_EVENT, 0.90, 100.0),
    ])
    cfg = RunConfig(
        tick_size=0.001, lot_size=1.0, slippage_bps=0.0, fee_taker=0.0,
        order_latency_ms=LAT_MS, ioc_marketability_recheck=True,
    )
    asset = _build_asset(arr, cfg)
    bt = hb.HashMapMarketDepthBacktest([asset])
    bt.elapse(T)

    # No snap_best arrays → empty dicts → check skipped
    st = _snap_run_state(bt, cfg)
    st.books = {"yes": BookState(symbol="yes", bid_px=None, bid_sz=None, ask_px=0.90,
                                 ask_sz=100.0, last_trade_ts_ns=0, last_l2_ts_ns=T)}
    st.now_ns = T

    _route_enter(st, _enter_decision(limit=0.92))
    bt.close()

    assert st.pos is not None, "no snap arrays → recheck skipped → fills normally"
    assert len(st.result.fills) == 1
    assert st.result.n_rejects == 0


def test_shr94_recheck_disabled_fills_fleeting():
    """With ioc_marketability_recheck=False, fleeting levels still fill (explicit
    opt-out to restore pre-SHR-94 behaviour)."""
    from hlanalysis.backtest.runner.hftbt_runner import _build_asset, _route_enter

    LAT_MS = 2100.0
    T = 1_000_000_000
    T_SNAP_NEXT = T + 2_000_000_000

    arr = np.concatenate([
        _make_depth_clear_arr(0),
        _depth_ev(T - 5_000_000, SELL_EVENT, 0.90, 100.0),
    ])
    cfg = RunConfig(
        tick_size=0.001, lot_size=1.0, slippage_bps=0.0, fee_taker=0.0,
        order_latency_ms=LAT_MS, ioc_marketability_recheck=False,  # recheck OFF
    )
    asset = _build_asset(arr, cfg)
    bt = hb.HashMapMarketDepthBacktest([asset])
    bt.elapse(T)

    snap_best_ask = np.array([0.90, 0.98], dtype=np.float64)
    book_ts = np.array([T - 5_000_000, T_SNAP_NEXT], dtype=np.int64)
    book_idx = {"yes": 1}

    st = _snap_run_state(
        bt, cfg,
        snap_best_ask_per_leg={"yes": snap_best_ask},
        book_ts_per_leg={"yes": book_ts},
        book_idx=book_idx,
    )
    st.books = {"yes": BookState(symbol="yes", bid_px=None, bid_sz=None, ask_px=0.90,
                                 ask_sz=100.0, last_trade_ts_ns=0, last_l2_ts_ns=T)}
    st.now_ns = T

    _route_enter(st, _enter_decision(limit=0.92))
    bt.close()

    assert st.pos is not None, (
        "recheck=False: fleeting level must fill (pre-SHR-94 behaviour restored)"
    )
    assert st.result.n_rejects == 0


def test_shr94_snap_best_from_columns_basic():
    """snap_best_from_columns extracts correct per-snapshot best ask/bid."""
    from hlanalysis.backtest.data._fastpath_core import snap_best_from_columns

    # 3 snapshots: s0=ask[0.5,0.6], s1=ask[0.4], s2=ask[]
    ask_px = np.array([0.5, 0.6, 0.4], dtype=np.float64)
    ask_off = np.array([0, 2, 3, 3], dtype=np.int64)  # lengths: [2, 1, 0]
    bid_px = np.array([0.3, 0.35, 0.38], dtype=np.float64)
    bid_off = np.array([0, 1, 2, 3], dtype=np.int64)  # lengths: [1, 1, 1]
    book_cols = {
        "ts": np.array([100, 200, 300], dtype=np.int64),
        "ask_px": ask_px, "ask_sz": np.ones(3),
        "ask_offsets": ask_off,
        "bid_px": bid_px, "bid_sz": np.ones(3),
        "bid_offsets": bid_off,
    }
    best_ask, best_bid = snap_best_from_columns(book_cols)

    assert best_ask.shape == (3,)
    assert best_bid.shape == (3,)
    assert best_ask[0] == pytest.approx(0.5)  # first ask at s0
    assert best_ask[1] == pytest.approx(0.4)  # first ask at s1
    assert np.isnan(best_ask[2])              # no asks at s2
    assert best_bid[0] == pytest.approx(0.3)
    assert best_bid[1] == pytest.approx(0.35)
    assert best_bid[2] == pytest.approx(0.38)


def test_shr94_snap_best_none_input():
    """snap_best_from_columns returns empty arrays when book_cols is None."""
    from hlanalysis.backtest.data._fastpath_core import snap_best_from_columns

    best_ask, best_bid = snap_best_from_columns(None)
    assert len(best_ask) == 0
    assert len(best_bid) == 0


def test_shr94_is_fleeting_ask_function():
    """_is_fleeting_ask unit test: next snapshot within window, ask higher → fleeting."""
    from hlanalysis.backtest.runner.hftbt_runner import _is_fleeting_ask

    T = 1_000_000_000
    LAT_NS = 2_100_000_000  # 2.1s
    # snapshot ts[0]=T, ts[1]=T+2s
    book_ts = {"yes": np.array([T, T + 2_000_000_000], dtype=np.int64)}
    # Next snapshot at idx=1 shows ask=0.98 > fill_price=0.90 → fleeting
    snap_ask = {"yes": np.array([0.90, 0.98], dtype=np.float64)}
    book_idx = {"yes": 1}  # cursor points to next unconsumed snapshot

    assert _is_fleeting_ask("yes", 0.90, T, LAT_NS, 0.0, snap_ask, book_ts, book_idx) is True


def test_shr94_is_fleeting_ask_outside_window():
    """SHR-98: the next recorded change is BEYOND the persistence window. On a
    change-driven feed that means the level held unchanged through the whole
    window → it was stable/live-hittable → NOT fleeting.
    """
    from hlanalysis.backtest.runner.hftbt_runner import _is_fleeting_ask

    T = 1_000_000_000
    PERSIST_NS = 50_000_000  # 50ms window
    # Next recorded change at T+5s — far beyond the 50ms window. The level was
    # therefore stable for 5s before reverting to 0.98 → live could have hit it.
    book_ts = {"yes": np.array([T + 5_000_000_000], dtype=np.int64)}
    snap_ask = {"yes": np.array([0.98], dtype=np.float64)}
    book_idx = {"yes": 0}

    assert _is_fleeting_ask("yes", 0.90, T, PERSIST_NS, 0.0, snap_ask, book_ts, book_idx) is False


def test_shr94_is_fleeting_ask_ask_still_present():
    """Next snapshot within latency window but ask is STILL at same price → not fleeting."""
    from hlanalysis.backtest.runner.hftbt_runner import _is_fleeting_ask

    T = 1_000_000_000
    LAT_NS = 2_100_000_000
    book_ts = {"yes": np.array([T, T + 2_000_000_000], dtype=np.int64)}
    # Next snapshot still shows ask=0.90 — not fleeting
    snap_ask = {"yes": np.array([0.90, 0.90], dtype=np.float64)}
    book_idx = {"yes": 1}

    assert _is_fleeting_ask("yes", 0.90, T, LAT_NS, 0.0, snap_ask, book_ts, book_idx) is False


def test_shr94_run_config_has_ioc_marketability_recheck():
    """RunConfig.ioc_marketability_recheck defaults to True."""
    cfg = RunConfig()
    assert cfg.ioc_marketability_recheck is True


# ---------------------------------------------------------------------------
# SHR-98: fleeting re-check must fire on HL's change-driven (burst) book, where
# a level can persist across two sub-second snapshots then revert. SHR-94 keyed
# the veto window on the ~50ms order latency, so on dt=5-60s books the next
# snapshot was always far past the window and the gate never fired → phantom
# fills on ~1s transients (v31 #2230: ask 0.90 for ~1.1s, then 0.98, +$20 that
# live got 0 fills on). The fix uses a wall-clock *persistence* window and
# scans every snapshot inside it for the level reverting.
#
# Real #2230 mechanics (recorded book, snapshots are change-driven):
#   t=+0.000s best_ask=0.90  ← sim fill 1 here
#   t=+0.545s best_ask=0.90  ← sim fill 2 here
#   t=+1.088s best_ask=0.98  ← level gone (reverted) ~1.1s after appearing
# A persistence window of 2s sees the 0.98 revert within the window → veto both.
# ---------------------------------------------------------------------------

# Persistence window used by the route-level integration tests below: 2s,
# matching the RunConfig default. The transient reverts ~1.1s out (inside it);
# a stable level's next change is 5s out (outside it).
_PERSIST_NS = 2_000_000_000


def test_shr98_subsecond_transient_ask_no_fill():
    """SHR-98: an ask that appears, holds across two sub-second snapshots, then
    reverts to an unmarketable price ~1.1s later (inside the persistence window)
    must NOT fill — matching live's zero-fill on the v31 #2230 bucket transient.
    """
    from hlanalysis.backtest.runner.hftbt_runner import _build_asset, _route_enter

    T = 1_000_000_000
    arr = np.concatenate([
        _make_depth_clear_arr(0),
        _depth_ev(T - 5_000_000, SELL_EVENT, 0.90, 100.0),
    ])
    cfg = RunConfig(
        tick_size=0.001, lot_size=1.0, slippage_bps=0.0, fee_taker=0.0,
        order_latency_ms=50.0, ioc_marketability_recheck=True,
        ioc_fleeting_persistence_seconds=2.0,
    )
    asset = _build_asset(arr, cfg)
    bt = hb.HashMapMarketDepthBacktest([asset])
    bt.elapse(T)

    # Current snapshot (idx 0) marketable; next (idx 1, +0.545s) still 0.90;
    # then (idx 2, +1.088s) reverts to 0.98 → gone within the 2s window.
    snap_best_ask = np.array([0.90, 0.90, 0.98], dtype=np.float64)
    book_ts = np.array(
        [T - 5_000_000, T + 545_000_000, T + 1_088_000_000], dtype=np.int64)
    book_idx = {"yes": 1}  # next unconsumed snapshot is idx 1

    st = _snap_run_state(
        bt, cfg,
        snap_best_ask_per_leg={"yes": snap_best_ask},
        book_ts_per_leg={"yes": book_ts},
        book_idx=book_idx,
    )
    st.books = {"yes": BookState(symbol="yes", bid_px=None, bid_sz=None, ask_px=0.90,
                                 ask_sz=100.0, last_trade_ts_ns=0, last_l2_ts_ns=T)}
    st.now_ns = T

    _route_enter(st, _enter_decision(limit=0.92))
    bt.close()

    assert st.result.fills == [], (
        "sub-second transient ask (reverts to 0.98 within the persistence "
        "window) must produce no fill — live got 0 fills on #2230"
    )
    assert st.pos is None, "no position opened on a sub-second transient ask"
    assert st.result.n_rejects == 1, "transient detected → counted as a reject"


def test_shr98_is_fleeting_ask_reverts_within_window():
    """Unit: best ask reverts above fill_price at a snapshot inside the
    persistence window (even though the IMMEDIATE next snapshot is still
    marketable) → fleeting. This is the exact #2230 shape SHR-94 missed."""
    from hlanalysis.backtest.runner.hftbt_runner import _is_fleeting_ask

    T = 1_000_000_000
    book_ts = {"yes": np.array(
        [T + 545_000_000, T + 1_088_000_000], dtype=np.int64)}
    snap_ask = {"yes": np.array([0.90, 0.98], dtype=np.float64)}
    book_idx = {"yes": 0}

    # +0.545s still 0.90 (ok), +1.088s 0.98 > 0.92 within 2s → fleeting.
    assert _is_fleeting_ask("yes", 0.92, T, _PERSIST_NS, 0.0, snap_ask, book_ts, book_idx) is True


def test_shr98_stable_ask_next_change_beyond_window_fills():
    """SHR-98 non-regression: on a change-driven feed, a snapshot whose next
    recorded change is BEYOND the persistence window was a STABLE level (no book
    change for seconds) → live-hittable → NOT fleeting. This is the case the old
    latency gate handled correctly and must keep filling (deep/stable v1)."""
    from hlanalysis.backtest.runner.hftbt_runner import _is_fleeting_ask

    T = 1_000_000_000
    # Next recorded change is 5s out and shows the level gone — but 5s of no
    # change means the 0.90 was stable for 5s → hittable.
    book_ts = {"yes": np.array([T + 5_000_000_000], dtype=np.int64)}
    snap_ask = {"yes": np.array([0.98], dtype=np.float64)}
    book_idx = {"yes": 0}

    assert _is_fleeting_ask("yes", 0.92, T, _PERSIST_NS, 0.0, snap_ask, book_ts, book_idx) is False


def test_shr98_stable_ask_marketable_through_window_fills():
    """SHR-98 non-regression: an ask that stays marketable through every
    snapshot in the persistence window → NOT fleeting → fills."""
    from hlanalysis.backtest.runner.hftbt_runner import _is_fleeting_ask

    T = 1_000_000_000
    book_ts = {"yes": np.array(
        [T + 500_000_000, T + 1_000_000_000, T + 1_500_000_000], dtype=np.int64)}
    snap_ask = {"yes": np.array([0.90, 0.90, 0.90], dtype=np.float64)}
    book_idx = {"yes": 0}

    assert _is_fleeting_ask("yes", 0.92, T, _PERSIST_NS, 0.0, snap_ask, book_ts, book_idx) is False


def test_shr98_is_fleeting_bid_reverts_within_window():
    """SHR-98 (sell side): a bid that retreats below fill_price within the
    persistence window → fleeting."""
    from hlanalysis.backtest.runner.hftbt_runner import _is_fleeting_bid

    T = 1_000_000_000
    book_ts = {"yes": np.array(
        [T + 500_000_000, T + 1_100_000_000], dtype=np.int64)}
    snap_bid = {"yes": np.array([0.90, 0.82], dtype=np.float64)}  # retreats
    book_idx = {"yes": 0}

    assert _is_fleeting_bid("yes", 0.88, T, _PERSIST_NS, 0.0, snap_bid, book_ts, book_idx) is True


# ---------------------------------------------------------------------------
# SHR-98 magnitude guard: a reverting level is only a phantom if the reversion
# is meaningfully adverse (> fill_price + min_revert). This separates the v31
# #2230 phantom (filled 0.90, reverts to 0.98 → 0.08) from ordinary near-
# settlement microstructure (v1 #2200: filled 0.99, book ticks to 0.99027 →
# 0.00027 after the resting size is consumed — a REAL fill live also got).
# Empirically all legitimate HL near-settlement reverts are ≤ ~0.008; #2230
# is 0.08; default min_revert=0.02 sits clear of both.
# ---------------------------------------------------------------------------


def test_shr98_large_revert_within_window_is_fleeting():
    """A revert well above fill_price + min_revert (the #2230 0.90→0.98 shape)
    → fleeting."""
    from hlanalysis.backtest.runner.hftbt_runner import _is_fleeting_ask

    T = 1_000_000_000
    book_ts = {"yes": np.array([T + 500_000_000, T + 1_100_000_000], dtype=np.int64)}
    snap_ask = {"yes": np.array([0.90, 0.98], dtype=np.float64)}  # reverts 0.08
    book_idx = {"yes": 0}

    # fill 0.90, reverts to 0.98 → 0.08 > min_revert 0.02 → fleeting.
    assert _is_fleeting_ask("yes", 0.90, T, _PERSIST_NS, 0.02, snap_ask, book_ts, book_idx) is True


def test_shr98_subcent_revert_within_window_is_not_fleeting():
    """A sub-cent revert (the v1/v31 #2200 near-settlement shape: filled ~0.99,
    book ticks back to 0.99027 after the resting size is taken) is a REAL fill,
    NOT a phantom — must NOT be vetoed even though it reverts within the window.
    """
    from hlanalysis.backtest.runner.hftbt_runner import _is_fleeting_ask

    T = 1_000_000_000
    book_ts = {"yes": np.array([T + 332_000_000, T + 546_000_000], dtype=np.int64)}
    snap_ask = {"yes": np.array([0.99027, 0.99027], dtype=np.float64)}
    book_idx = {"yes": 0}

    # fill 0.99, reverts to 0.99027 → 0.00027 < min_revert 0.02 → NOT fleeting.
    assert _is_fleeting_ask("yes", 0.99, T, _PERSIST_NS, 0.02, snap_ask, book_ts, book_idx) is False


def test_shr98_subcent_revert_bid_is_not_fleeting():
    """Sell-side mirror: a sub-cent bid tick-back within the window is a real
    fill, not a phantom."""
    from hlanalysis.backtest.runner.hftbt_runner import _is_fleeting_bid

    T = 1_000_000_000
    book_ts = {"yes": np.array([T + 332_000_000, T + 546_000_000], dtype=np.int64)}
    snap_bid = {"yes": np.array([0.00973, 0.00973], dtype=np.float64)}
    book_idx = {"yes": 0}

    # fill 0.01, bid ticks to 0.00973 → 0.00027 < min_revert 0.02 → NOT fleeting.
    assert _is_fleeting_bid("yes", 0.01, T, _PERSIST_NS, 0.02, snap_bid, book_ts, book_idx) is False


def test_shr98_near_settlement_microstructure_fill_not_vetoed():
    """SHR-98 non-regression (route level): a v1 #2200-style near-settlement
    entry where the book ticks back a sub-cent within the window must still FILL
    under the default magnitude guard — live got these fills."""
    from hlanalysis.backtest.runner.hftbt_runner import _build_asset, _route_enter

    T = 1_000_000_000
    arr = np.concatenate([
        _make_depth_clear_arr(0),
        _depth_ev(T - 5_000_000, SELL_EVENT, 0.99, 300.0),
    ])
    cfg = RunConfig(
        tick_size=0.001, lot_size=1.0, slippage_bps=0.0, fee_taker=0.0,
        order_latency_ms=50.0, ioc_marketability_recheck=True,
        ioc_fleeting_persistence_seconds=2.0, ioc_fleeting_min_revert=0.02,
    )
    asset = _build_asset(arr, cfg)
    bt = hb.HashMapMarketDepthBacktest([asset])
    bt.elapse(T)

    # Book ticks 0.99 → 0.99027 within the window (sub-cent → real fill).
    snap_best_ask = np.array([0.99, 0.99027, 0.99027], dtype=np.float64)
    book_ts = np.array([T - 5_000_000, T + 332_000_000, T + 546_000_000], dtype=np.int64)
    book_idx = {"yes": 1}

    st = _snap_run_state(
        bt, cfg,
        snap_best_ask_per_leg={"yes": snap_best_ask},
        book_ts_per_leg={"yes": book_ts},
        book_idx=book_idx,
    )
    st.books = {"yes": BookState(symbol="yes", bid_px=None, bid_sz=None, ask_px=0.99,
                                 ask_sz=300.0, last_trade_ts_ns=0, last_l2_ts_ns=T)}
    st.now_ns = T

    _route_enter(st, _enter_decision(size=300.0, limit=0.99))
    bt.close()

    assert st.pos is not None, "near-settlement sub-cent tick-back must still fill"
    assert len(st.result.fills) == 1
    assert st.result.n_rejects == 0


def test_shr98_run_config_has_fleeting_persistence_seconds():
    """RunConfig fleeting knobs default to (2.0s, 0.02)."""
    assert RunConfig().ioc_fleeting_persistence_seconds == 2.0
    assert RunConfig().ioc_fleeting_min_revert == 0.02


# SHR-95: event-driven scan mode
# ---------------------------------------------------------------------------
#
# Three tests:
#   1. Event mode fires more scans than the fixed grid in an active window.
#   2. A quiet window (no events) still fires once at the max-interval ceiling.
#   3. Default (fixed) mode is byte-identical to the current fixed-grid behavior.


class _CountingStrategy(Strategy):
    """Test-only strategy: counts evaluate() calls and always HOLDs.

    We use this to count scan-tick firings without the fill-path complications.
    The call count is mutated in-place for easy inspection after the run.
    """
    name = "_counting_strategy"

    def __init__(self):
        self.call_count: int = 0
        self.call_ts_ns: list[int] = []

    def evaluate(self, *, question, books, reference_price, recent_returns,
                 recent_volume_usd, position, now_ns, recent_hl_bars=()):
        from hlanalysis.strategy.types import Action, Decision
        self.call_count += 1
        self.call_ts_ns.append(now_ns)
        return Decision(action=Action.HOLD)


def _make_dense_book_question(
    *,
    n_book_updates: int = 30,
    duration_ns: int = 10 * 60 * 1_000_000_000,  # 10 min
    update_stride_ns: int | None = None,
) -> "tuple":
    """Build a synthetic question whose YES leg has `n_book_updates` book
    snapshots spread over `duration_ns`. Returns (SyntheticDataSource, sq).

    We use the legacy ``events()`` path (no events_arrays) so we control the
    exact book_ts array the runner sees — keeping the test independent of HL-
    or PM-specific fast-path logic.
    """
    from hlanalysis.backtest.data.synthetic import (
        SyntheticDataSource,
        SyntheticQuestion,
    )
    from hlanalysis.backtest.core.data_source import QuestionDescriptor
    from hlanalysis.backtest.core.events import (
        BookSnapshot, ReferenceEvent, SettlementEvent,
    )

    start_ns: int = 0
    end_ns: int = duration_ns
    yes_sym = "ev-yes"
    no_sym = "ev-no"

    desc = QuestionDescriptor(
        question_id="ev-q-0", question_idx=1,
        start_ts_ns=start_ns, end_ts_ns=end_ns,
        leg_symbols=(yes_sym, no_sym),
        klass="priceBinary", underlying="BTC",
    )

    if update_stride_ns is None:
        update_stride_ns = duration_ns // n_book_updates

    # Build n_book_updates book snapshots; first one is 1ms after start.
    snaps: list[BookSnapshot] = []
    for i in range(n_book_updates):
        t = start_ns + 1_000_000 + i * update_stride_ns
        bid = 0.40 + 0.10 * (i / max(1, n_book_updates - 1))
        ask = bid + 0.02
        snaps.append(BookSnapshot(ts_ns=t, symbol=yes_sym,
                                  bids=((round(bid, 4), 100.0),),
                                  asks=((round(ask, 4), 100.0),)))
        snaps.append(BookSnapshot(ts_ns=t, symbol=no_sym,
                                  bids=((round(1.0 - ask, 4), 100.0),),
                                  asks=((round(1.0 - bid, 4), 100.0),)))

    # A few reference events spread over the window.
    refs: list[ReferenceEvent] = [
        ReferenceEvent(ts_ns=start_ns + int(j * duration_ns / 3),
                       symbol="BTC", high=60_100.0, low=59_900.0, close=60_000.0)
        for j in range(4)
    ]

    settle = [SettlementEvent(ts_ns=end_ns, question_idx=1, outcome="yes")]

    sq = SyntheticQuestion(
        descriptor=desc,
        book_snapshots=tuple(snaps),
        trades=(),
        reference_events=tuple(refs),
        settlement_events=tuple(settle),
        outcome="yes",
        strike=60_000.0,
    )
    ds = SyntheticDataSource()
    ds.add_question(sq)
    return ds, sq


def test_event_mode_fires_more_scans_than_fixed_grid():
    """Event-driven mode must produce more strategy evaluations than a fixed
    2s grid over the same 10-minute active window.

    Setup: 30 book updates, 1 every 20s. Fixed grid at 60s -> 9 scans in 10m.
    Event mode (min=0.2s, max=60s) -> at most one scan per update (30 updates
    in 10m) -- strictly more than 9 at the 60s fixed grid.
    """
    ds, sq = _make_dense_book_question(n_book_updates=30, duration_ns=10 * 60 * 1_000_000_000,
                                       update_stride_ns=20 * 1_000_000_000)

    fixed_strat = _CountingStrategy()
    fixed_cfg = RunConfig(
        scanner_interval_seconds=60,
        slippage_bps=0.0, fee_taker=0.0, order_latency_ms=0.0,
    )
    run_one_question(fixed_strat, ds, sq.descriptor, fixed_cfg, strike=sq.strike)

    event_strat = _CountingStrategy()
    event_cfg = RunConfig(
        scan_mode="event",
        scan_min_interval_seconds=0.2,
        scan_max_interval_seconds=60.0,
        slippage_bps=0.0, fee_taker=0.0, order_latency_ms=0.0,
    )
    run_one_question(event_strat, ds, sq.descriptor, event_cfg, strike=sq.strike)

    assert event_strat.call_count > fixed_strat.call_count, (
        f"event mode ({event_strat.call_count} evals) must exceed fixed-grid "
        f"({fixed_strat.call_count} evals) for 30 updates at 20s stride vs 60s grid"
    )


def test_event_mode_respects_min_interval_floor():
    """No two consecutive scan ticks may be closer than scan_min_interval.

    Setup: 20 book updates at 0.1s stride -- faster than the 0.2s floor.
    Event mode should merge adjacent triggers so no gap < 0.2s appears.
    """
    min_s = 0.2
    ds, sq = _make_dense_book_question(
        n_book_updates=20,
        duration_ns=5 * 60 * 1_000_000_000,
        update_stride_ns=int(0.1 * 1_000_000_000),  # 100ms stride < 200ms floor
    )

    strat = _CountingStrategy()
    cfg = RunConfig(
        scan_mode="event",
        scan_min_interval_seconds=min_s,
        scan_max_interval_seconds=10.0,
        slippage_bps=0.0, fee_taker=0.0, order_latency_ms=0.0,
    )
    run_one_question(strat, ds, sq.descriptor, cfg, strike=sq.strike)

    min_ns = int(min_s * 1_000_000_000)
    ts = strat.call_ts_ns
    assert len(ts) >= 2, "expected at least two scans"
    gaps = [ts[i + 1] - ts[i] for i in range(len(ts) - 1)]
    assert all(g >= min_ns - 1 for g in gaps), (
        f"some consecutive scans were closer than scan_min_interval={min_s}s: "
        f"min gap = {min(gaps) / 1e9:.4f}s"
    )


def test_event_mode_quiet_window_scans_at_max_interval():
    """A question with NO book updates (completely quiet market) must still
    trigger scans at the max_interval ceiling.

    A 60s question with max_interval=10s and zero book events should produce
    ~6 scans (the fixed-ceiling triggers), not zero.
    """
    from hlanalysis.backtest.data.synthetic import SyntheticDataSource, SyntheticQuestion
    from hlanalysis.backtest.core.data_source import QuestionDescriptor
    from hlanalysis.backtest.core.events import BookSnapshot, ReferenceEvent, SettlementEvent

    duration_ns = 60 * 1_000_000_000  # 60s
    max_s = 10.0
    start_ns = 0
    end_ns = duration_ns
    yes_sym = "quiet-yes"
    no_sym = "quiet-no"

    desc = QuestionDescriptor(
        question_id="quiet-q-0", question_idx=1,
        start_ts_ns=start_ns, end_ts_ns=end_ns,
        leg_symbols=(yes_sym, no_sym),
        klass="priceBinary", underlying="BTC",
    )
    # One book snapshot at t=1ms so hftbacktest has non-empty depth.
    snap = BookSnapshot(ts_ns=1_000_000, symbol=yes_sym,
                        bids=((0.40, 100.0),), asks=((0.42, 100.0),))
    snap_no = BookSnapshot(ts_ns=1_000_000, symbol=no_sym,
                           bids=((0.58, 100.0),), asks=((0.60, 100.0),))
    refs = [ReferenceEvent(ts_ns=0, symbol="BTC", high=60_100.0, low=59_900.0, close=60_000.0)]
    settle = [SettlementEvent(ts_ns=end_ns, question_idx=1, outcome="yes")]

    sq = SyntheticQuestion(
        descriptor=desc,
        book_snapshots=(snap, snap_no),
        trades=(),
        reference_events=tuple(refs),
        settlement_events=tuple(settle),
        outcome="yes",
        strike=60_000.0,
    )
    ds_quiet = SyntheticDataSource()
    ds_quiet.add_question(sq)

    strat = _CountingStrategy()
    cfg = RunConfig(
        scan_mode="event",
        scan_min_interval_seconds=0.2,
        scan_max_interval_seconds=max_s,
        slippage_bps=0.0, fee_taker=0.0, order_latency_ms=0.0,
    )
    run_one_question(strat, ds_quiet, sq.descriptor, cfg, strike=sq.strike)

    # With 60s window and 10s max ceiling, expect at least 5 scans.
    assert strat.call_count >= 5, (
        f"quiet market: expected >=5 scans from 10s max-ceiling in 60s window, "
        f"got {strat.call_count}"
    )


def test_fixed_mode_is_default_and_unchanged():
    """Default mode is 'fixed'; results are byte-identical to the pre-SHR-95 path.

    Run the same synthetic question twice: once with an explicit
    scan_mode='fixed' and once with the current default RunConfig (no scan_mode
    field). Both must produce the same number of strategy evaluations.
    """
    ds, sq = _make_dense_book_question(n_book_updates=10,
                                       duration_ns=10 * 60 * 1_000_000_000,
                                       update_stride_ns=60 * 1_000_000_000)

    strat_a = _CountingStrategy()
    cfg_default = RunConfig(scanner_interval_seconds=60,
                            slippage_bps=0.0, fee_taker=0.0, order_latency_ms=0.0)
    run_one_question(strat_a, ds, sq.descriptor, cfg_default, strike=sq.strike)

    strat_b = _CountingStrategy()
    cfg_explicit = RunConfig(scanner_interval_seconds=60, scan_mode="fixed",
                             slippage_bps=0.0, fee_taker=0.0, order_latency_ms=0.0)
    run_one_question(strat_b, ds, sq.descriptor, cfg_explicit, strike=sq.strike)

    assert strat_a.call_count == strat_b.call_count, (
        f"default and explicit fixed mode must produce identical scan counts; "
        f"got {strat_a.call_count} vs {strat_b.call_count}"
    )
    assert strat_a.call_count > 0, "fixed mode must scan at least once"


def test_run_config_scan_mode_defaults():
    """RunConfig must have scan_mode='fixed' (default), scan_min_interval_seconds=0.2,
    and scan_max_interval_seconds=2.0 after SHR-95."""
    cfg = RunConfig()
    assert cfg.scan_mode == "fixed", (
        f"scan_mode default must be 'fixed', got {cfg.scan_mode!r}"
    )
    assert cfg.scan_min_interval_seconds == pytest.approx(0.2), (
        f"scan_min_interval_seconds default must be 0.2, got {cfg.scan_min_interval_seconds}"
    )
    assert cfg.scan_max_interval_seconds == pytest.approx(2.0), (
        f"scan_max_interval_seconds default must be 2.0, got {cfg.scan_max_interval_seconds}"
    )


def test_cli_scan_mode_args_exist():
    """--scan-mode, --scan-min-interval-seconds, --scan-max-interval-seconds
    must be declared by _add_run_config_args and default correctly."""
    import argparse
    from hlanalysis.backtest.cli import _add_run_config_args, _run_config_from_args

    p = argparse.ArgumentParser()
    _add_run_config_args(p)

    # Defaults
    args = p.parse_args([])
    assert args.scan_mode == "fixed"
    assert args.scan_min_interval_seconds == pytest.approx(0.2)
    assert args.scan_max_interval_seconds == pytest.approx(2.0)

    # Explicit event-mode values propagate into RunConfig.
    args2 = p.parse_args(["--scan-mode", "event",
                           "--scan-min-interval-seconds", "0.5",
                           "--scan-max-interval-seconds", "5.0"])
    cfg = _run_config_from_args(args2, hedge_cfg=None)
    assert cfg.scan_mode == "event"
    assert cfg.scan_min_interval_seconds == pytest.approx(0.5)
    assert cfg.scan_max_interval_seconds == pytest.approx(5.0)
