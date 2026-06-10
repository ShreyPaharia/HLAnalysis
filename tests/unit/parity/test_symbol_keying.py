"""SHR-90 symbol-keyed reconciliation.

The backtest and live engine disagree on ``question_idx`` for the same market
(buckets: live ``#1670→31`` vs sim ``→1000167``) and venue fills carry
``question_idx=-1`` until attributed — so matching sim↔live on question_idx
silently drops markets. The stable HL ``symbol`` (``#NNNN``) is present on both
sides and on every fill, so reconciliation must be able to key on it.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hlanalysis.engine.state import FILL_SOURCE_VENUE, Fill, StateDAL, TradeJournalRow
from hlanalysis.parity.sources import (
    LiveMarket,
    SimMarket,
    build_pairs,
    load_live_markets_from_db,
    reconcile_markets,
)


def test_symbol_key_matches_despite_mismatched_question_idx():
    live = [LiveMarket(question_idx=31, symbol="#1670", realized_pnl=-133.0, traded=True,
                       halt_active=False, sigma=None, reference_price=None, n_fills=48)]
    sim = [SimMarket(question_idx=1000167, symbol="#1670", realized_pnl=-196.0,
                     traded=True, sigma=None, reference_price=None, n_fills=17)]

    # question_idx keying (default) can't pair them → two one-sided pairs.
    assert all(not p.is_matched for p in build_pairs(live, sim))

    # symbol keying pairs them into one matched decision.
    pairs = build_pairs(live, sim, key="symbol")
    assert len(pairs) == 1 and pairs[0].is_matched

    markets = reconcile_markets(live, sim, key="symbol")
    assert len(markets) == 1
    assert markets[0].symbol == "#1670"
    assert markets[0].residual == pytest.approx(-196.0 - (-133.0))


def _dal(tmp_path: Path) -> StateDAL:
    dal = StateDAL(tmp_path / "state.db")
    dal.run_migrations()
    return dal


def test_symbol_rollup_separates_unindexed_venue_fills(tmp_path):
    """Venue fills on DIFFERENT markets both carry question_idx=-1 until
    attributed. A question_idx rollup collapses them into one bogus -1 bucket
    (PnL summed, symbol arbitrary); a symbol rollup keeps them separate and
    credits each market's closed_pnl correctly."""
    dal = _dal(tmp_path)
    dal.append_fill(Fill(
        fill_id="f1", cloid="c1", question_idx=-1, symbol="#2250", side="sell",
        price=0.80, size=100.0, fee=0.0, ts_ns=1_000, closed_pnl=-5.0,
        source=FILL_SOURCE_VENUE,
    ))
    dal.append_fill(Fill(
        fill_id="f2", cloid="c2", question_idx=-1, symbol="#1591", side="sell",
        price=0.95, size=100.0, fee=0.0, ts_ns=2_000, closed_pnl=8.0,
        source=FILL_SOURCE_VENUE,
    ))

    # question_idx rollup collapses both into a single -1 market.
    by_q = load_live_markets_from_db(dal.db_path)
    assert len(by_q) == 1 and by_q[0].realized_pnl == pytest.approx(3.0)  # -5 + 8 merged

    # symbol rollup separates them and credits each market correctly.
    by_sym = {m.symbol: m for m in load_live_markets_from_db(dal.db_path, key="symbol")}
    assert by_sym["#2250"].realized_pnl == pytest.approx(-5.0)
    assert by_sym["#1591"].realized_pnl == pytest.approx(8.0)
    assert by_sym["#2250"].n_fills == 1 and by_sym["#1591"].n_fills == 1
