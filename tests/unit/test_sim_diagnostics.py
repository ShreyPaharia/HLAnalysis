"""Tests for hlanalysis/sim/diagnostics.py — per-decision diagnostics parquet.

TDD: written before implementation. Run with:
    uv run pytest tests/unit/test_sim_diagnostics.py -v
"""
from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import pytest

from hlanalysis.sim.data.binance_klines import Kline
from hlanalysis.sim.data.schemas import PMMarket, PMTrade
from hlanalysis.sim.diagnostics import DiagnosticRow, write_diagnostics
from hlanalysis.sim.fills import FillModelConfig
from hlanalysis.sim.runner import RunnerConfig, run_one_market
from hlanalysis.strategy.model_edge import ModelEdgeConfig, ModelEdgeStrategy


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _market(condition_id: str = "0xabc") -> PMMarket:
    return PMMarket(
        condition_id=condition_id,
        yes_token_id="Y",
        no_token_id="N",
        start_ts_ns=0,
        end_ts_ns=86_400_000_000_000,
        resolved_outcome="yes",
        total_volume_usd=10_000.0,
        n_trades=100,
    )


def _klines(n: int = 60) -> list[Kline]:
    """60 one-minute klines. BTC moves from 100_000 to 100_059 so the strategy
    has non-zero returns and can compute sigma after the vol lookback is warm."""
    return [
        Kline(
            ts_ns=i * 60_000_000_000,
            open=100_000 + i,
            high=100_000 + i + 1,
            low=100_000 + i,
            close=100_000 + i + 1,
            volume=1.0,
        )
        for i in range(n)
    ]


def _trades() -> list[PMTrade]:
    return [
        PMTrade(ts_ns=10 * 60_000_000_000, token_id="Y", side="buy", price=0.6, size=100),
        PMTrade(ts_ns=10 * 60_000_000_000 + 1, token_id="N", side="buy", price=0.4, size=100),
    ]


def _v2_strategy(*, edge_buffer: float = 0.02) -> ModelEdgeStrategy:
    return ModelEdgeStrategy(
        ModelEdgeConfig(
            vol_lookback_seconds=3600,
            vol_sampling_dt_seconds=60,
            vol_clip_min=0.05,
            vol_clip_max=3.0,
            edge_buffer=edge_buffer,
            fee_taker=0.02,
            half_spread_assumption=0.005,
            stop_loss_pct=10.0,
            max_position_usd=100.0,
        )
    )


def _runner_cfg() -> RunnerConfig:
    return RunnerConfig(
        scanner_interval_seconds=60,
        fill_model=FillModelConfig(
            slippage_bps=5.0, fee_taker=0.02, book_depth_assumption=10_000.0
        ),
        synthetic_half_spread=0.005,
        synthetic_depth=10_000.0,
        day_open_btc=100_000.0,
    )


# ---------------------------------------------------------------------------
# Unit tests for DiagnosticRow and write_diagnostics
# ---------------------------------------------------------------------------

class TestDiagnosticRow:
    def test_hold_row_model_fields_none(self):
        """v1-style hold row: model-specific fields should accept None."""
        row = DiagnosticRow(
            ts_ns=1_000_000_000,
            condition_id="0xtest",
            question_idx=42,
            action="hold",
            reason="tte_out_of_window",
            p_model=None,
            edge_yes=None,
            edge_no=None,
            sigma=None,
            tau_yr=None,
            ln_sk=None,
            ref_price=None,
            yes_bid=None,
            yes_ask=None,
            no_bid=None,
            no_ask=None,
        )
        assert row.action == "hold"
        assert row.p_model is None

    def test_enter_row_carries_model_fields(self):
        row = DiagnosticRow(
            ts_ns=2_000_000_000,
            condition_id="0xtest",
            question_idx=7,
            action="enter",
            reason="entry",
            p_model=0.72,
            edge_yes=0.05,
            edge_no=-0.1,
            sigma=0.80,
            tau_yr=0.002739726,
            ln_sk=0.001,
            ref_price=101_000.0,
            yes_bid=0.60,
            yes_ask=0.65,
            no_bid=0.35,
            no_ask=0.40,
        )
        assert row.action == "enter"
        assert abs(row.p_model - 0.72) < 1e-9
        assert abs(row.sigma - 0.80) < 1e-9


class TestWriteDiagnostics:
    def test_write_creates_parquet_readable(self, tmp_path: Path):
        rows = [
            DiagnosticRow(
                ts_ns=1_000_000_000,
                condition_id="0xabc",
                question_idx=1,
                action="hold",
                reason="edge",
                p_model=0.55,
                edge_yes=-0.01,
                edge_no=-0.02,
                sigma=0.7,
                tau_yr=0.003,
                ln_sk=0.0,
                ref_price=100_000.0,
                yes_bid=0.50,
                yes_ask=0.55,
                no_bid=0.45,
                no_ask=0.50,
            ),
        ]
        out_path = tmp_path / "diag.parquet"
        write_diagnostics(rows, out_path)

        table = pq.read_table(out_path)
        assert table.num_rows == 1
        assert "p_model" in table.schema.names
        assert "sigma" in table.schema.names
        assert "tau_yr" in table.schema.names

    def test_write_preserves_none_as_null(self, tmp_path: Path):
        rows = [
            DiagnosticRow(
                ts_ns=1_000_000_000,
                condition_id="0xabc",
                question_idx=1,
                action="hold",
                reason="tte_out_of_window",
                p_model=None,
                edge_yes=None,
                edge_no=None,
                sigma=None,
                tau_yr=None,
                ln_sk=None,
                ref_price=100_000.0,
                yes_bid=None,
                yes_ask=None,
                no_bid=None,
                no_ask=None,
            ),
        ]
        out_path = tmp_path / "diag_nulls.parquet"
        write_diagnostics(rows, out_path)
        table = pq.read_table(out_path)
        assert table.num_rows == 1
        p_model_col = table.column("p_model")
        assert p_model_col[0].as_py() is None


# ---------------------------------------------------------------------------
# Integration: run_one_market writes diagnostics via diagnostics_dir
# ---------------------------------------------------------------------------

class TestRunnerDiagnosticsDir:
    def test_runner_writes_diagnostics_parquet(self, tmp_path: Path):
        """AC1+AC2: run_one_market writes at least one row per market that saw events."""
        market = _market(condition_id="0xcond1")
        diag_dir = tmp_path / "diagnostics"
        result = run_one_market(
            _v2_strategy(),
            market,
            _klines(),
            _trades(),
            _runner_cfg(),
            diagnostics_dir=diag_dir,
        )
        assert result.n_decisions >= 1
        expected_path = diag_dir / "0xcond1.parquet"
        assert expected_path.exists(), f"Expected parquet at {expected_path}"
        table = pq.read_table(expected_path)
        assert table.num_rows >= 1

    def test_runner_no_diagnostics_dir_skips_write(self, tmp_path: Path):
        """If diagnostics_dir is None, no parquet is written."""
        market = _market(condition_id="0xcond2")
        result = run_one_market(
            _v2_strategy(),
            market,
            _klines(),
            _trades(),
            _runner_cfg(),
            diagnostics_dir=None,
        )
        assert result.n_decisions >= 1
        # No parquet anywhere under tmp_path (none was created)
        # Just check none written to any obvious location
        assert not (tmp_path / "diagnostics").exists()

    def test_runner_v2_parquet_has_expected_columns(self, tmp_path: Path):
        """AC3: parquet has p_model, sigma, tau_yr columns."""
        market = _market(condition_id="0xcond3")
        diag_dir = tmp_path / "diagnostics"
        run_one_market(
            _v2_strategy(),
            market,
            _klines(),
            _trades(),
            _runner_cfg(),
            diagnostics_dir=diag_dir,
        )
        table = pq.read_table(diag_dir / "0xcond3.parquet")
        for col in ("p_model", "sigma", "tau_yr", "ts_ns", "action", "reason"):
            assert col in table.schema.names, f"Missing column: {col}"

    def test_runner_v2_enter_row_has_non_null_model_fields(self, tmp_path: Path):
        """AC3: when strategy ENTERs, the row has p_model/sigma/tau_yr populated.

        We configure a very generous edge_buffer=0.0 so the strategy will enter
        as soon as it has enough vol data and sees a positive edge.
        """
        market = _market(condition_id="0xcond4")
        diag_dir = tmp_path / "diagnostics"
        # Use aggressive params: very low edge_buffer and fee so the strategy enters.
        strat = ModelEdgeStrategy(
            ModelEdgeConfig(
                vol_lookback_seconds=600,   # warm after 10 klines
                vol_sampling_dt_seconds=60,
                vol_clip_min=0.01,
                vol_clip_max=5.0,
                edge_buffer=-1.0,           # always enter if we have a book
                fee_taker=0.0,
                half_spread_assumption=0.0,
                stop_loss_pct=None,
                max_position_usd=100.0,
            )
        )
        run_one_market(
            strat,
            market,
            _klines(120),
            _trades(),
            _runner_cfg(),
            diagnostics_dir=diag_dir,
        )
        table = pq.read_table(diag_dir / "0xcond4.parquet")
        rows = table.to_pylist()
        enter_rows = [r for r in rows if r["action"] == "enter"]
        assert len(enter_rows) >= 1, f"Expected at least one ENTER row, got actions: {[r['action'] for r in rows]}"
        for r in enter_rows:
            assert r["p_model"] is not None, "ENTER row missing p_model"
            assert r["sigma"] is not None, "ENTER row missing sigma"
            assert r["tau_yr"] is not None, "ENTER row missing tau_yr"

    def test_runner_v1_strategy_parquet_no_crash(self, tmp_path: Path):
        """v1 strategy: diagnostics still written; model fields are None (no crash)."""
        from hlanalysis.sim.v1_factory import build_v1_strategy_from_params
        market = _market(condition_id="0xcond5")
        diag_dir = tmp_path / "diagnostics"
        strat = build_v1_strategy_from_params({
            "tte_min_seconds": 3600,
            "tte_max_seconds": 86400,
            "price_extreme_threshold": 0.9,
            "distance_from_strike_usd_min": 100.0,
            "vol_max": 999.0,
            "max_position_usd": 100.0,
            "stop_loss_pct": 10.0,
            "max_strike_distance_pct": 0.5,
            "min_recent_volume_usd": 0.0,
            "stale_data_halt_seconds": 86400,
        })
        run_one_market(
            strat,
            market,
            _klines(),
            _trades(),
            _runner_cfg(),
            diagnostics_dir=diag_dir,
        )
        parquet_path = diag_dir / "0xcond5.parquet"
        assert parquet_path.exists()
        table = pq.read_table(parquet_path)
        assert table.num_rows >= 1
        # model fields should all be null for v1
        rows = table.to_pylist()
        for r in rows:
            assert r["p_model"] is None
            assert r["sigma"] is None
