"""Regression guard: realized_pnl_since MUST include settlement payouts.
HIP-4 binaries close via settlement, not HL fills (SHR-49/53). If this
breaks, the daily-loss gate and the reconciliation report both go blind to
the dominant PnL component of the binary strategy."""
from pathlib import Path

from hlanalysis.engine.state import Fill, Position, StateDAL


def test_realized_pnl_since_includes_settlement(tmp_path: Path) -> None:
    dal = StateDAL(tmp_path / "state.db")
    dal.run_migrations()

    # A settlement payout with NO corresponding fill row.
    dal.record_settlement(
        question_idx=7, symbol="BTC", realized_pnl=120.0, ts_ns=1_000,
    )

    # realized_pnl_since(0) must surface the settlement even with zero fills.
    assert dal.realized_pnl_since(0) == 120.0
    # And it must respect the since window.
    assert dal.realized_pnl_since(2_000) == 0.0
    # settlement_pnl_since is the isolated component.
    assert dal.settlement_pnl_since(0) == 120.0


def test_realized_pnl_since_partial_reduce_not_double_counted(tmp_path: Path) -> None:
    """SHR-72: a position partially reduced but STILL OPEN must count its
    partial-reduce realized PnL exactly once. Router._book_fill writes a Fill
    row (closed_pnl on the reduce) AND persists the still-open position's
    accumulated realized_pnl; summing both double-counts the partial PnL."""
    dal = StateDAL(tmp_path / "state.db")
    dal.run_migrations()

    # --- Open: buy 10 @ 100 (no realized PnL on an open) ---
    dal.append_fill(Fill(
        fill_id="c1-1000", cloid="c1", question_idx=1, symbol="BTC",
        side="buy", price=100.0, size=10.0, fee=0.0, ts_ns=1_000,
        closed_pnl=0.0,
    ))
    dal.upsert_position(Position(
        question_idx=1, symbol="BTC", qty=10.0, avg_entry=100.0,
        realized_pnl=0.0, last_update_ts_ns=1_000, stop_loss_price=0.0,
        closed_qty=0.0,
    ))

    # --- Partial reduce: sell 4 @ 110 → realized (110-100)*4 = 40, still open ---
    realized = 40.0
    dal.append_fill(Fill(
        fill_id="c2-2000", cloid="c2", question_idx=1, symbol="BTC",
        side="sell", price=110.0, size=4.0, fee=0.0, ts_ns=2_000,
        closed_pnl=realized,
    ))
    dal.upsert_position(Position(
        question_idx=1, symbol="BTC", qty=6.0, avg_entry=100.0,
        realized_pnl=realized, last_update_ts_ns=2_000, stop_loss_price=0.0,
        closed_qty=4.0,
    ))

    # The windowed realized PnL is the SINGLE partial-reduce amount, not 2x.
    assert dal.realized_pnl_since(0) == realized


def test_realized_pnl_since_full_close_after_partial_reduce(tmp_path: Path) -> None:
    """SHR-72 no-regression: open → partial reduce → FULL close. Once closed the
    Position row is deleted, and the whole-trade realized PnL must equal the sum
    of every fill's closed_pnl (entry 0 + partial 40 + closing lot 120 = 160),
    counted exactly once. This is the path the fix must NOT break: dropping the
    open-position term still leaves the Fill sum authoritative."""
    dal = StateDAL(tmp_path / "state.db")
    dal.run_migrations()

    # --- Open: buy 10 @ 100 ---
    dal.append_fill(Fill(
        fill_id="c1-1000", cloid="c1", question_idx=1, symbol="BTC",
        side="buy", price=100.0, size=10.0, fee=0.0, ts_ns=1_000,
        closed_pnl=0.0,
    ))
    dal.upsert_position(Position(
        question_idx=1, symbol="BTC", qty=10.0, avg_entry=100.0,
        realized_pnl=0.0, last_update_ts_ns=1_000, stop_loss_price=0.0,
        closed_qty=0.0,
    ))

    # --- Partial reduce: sell 4 @ 110 → realized (110-100)*4 = 40, still open ---
    dal.append_fill(Fill(
        fill_id="c2-2000", cloid="c2", question_idx=1, symbol="BTC",
        side="sell", price=110.0, size=4.0, fee=0.0, ts_ns=2_000,
        closed_pnl=40.0,
    ))
    dal.upsert_position(Position(
        question_idx=1, symbol="BTC", qty=6.0, avg_entry=100.0,
        realized_pnl=40.0, last_update_ts_ns=2_000, stop_loss_price=0.0,
        closed_qty=4.0,
    ))

    # --- Full close: sell 6 @ 120 → closing-lot realized (120-100)*6 = 120 ---
    # _book_fill writes the closing lot's PnL (not the cumulative) on the Fill
    # and DELETES the Position row.
    dal.append_fill(Fill(
        fill_id="c3-3000", cloid="c3", question_idx=1, symbol="BTC",
        side="sell", price=120.0, size=6.0, fee=0.0, ts_ns=3_000,
        closed_pnl=120.0,
    ))
    dal.delete_position(1)

    # Whole-trade PnL = 0 + 40 + 120 = 160, counted once (bought 10@100=1000,
    # sold 4@110 + 6@120 = 440 + 720 = 1160; profit 160).
    assert dal.realized_pnl_since(0) == 160.0
    # Windowing still holds: only the closing lot lands after ts 2_500.
    assert dal.realized_pnl_since(2_500) == 120.0
