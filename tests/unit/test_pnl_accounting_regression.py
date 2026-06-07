"""Regression guard: realized_pnl_since MUST include settlement payouts.
HIP-4 binaries close via settlement, not HL fills (SHR-49/53). If this
breaks, the daily-loss gate and the reconciliation report both go blind to
the dominant PnL component of the binary strategy."""
from pathlib import Path

from hlanalysis.engine.state import StateDAL


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
