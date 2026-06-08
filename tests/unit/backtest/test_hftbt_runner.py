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
