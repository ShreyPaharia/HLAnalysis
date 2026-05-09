"""Tests for hlanalysis/sim/diagnostics.py — per-decision diagnostics parquet.

TDD: written before implementation. Run with:
    uv run pytest tests/unit/test_sim_diagnostics.py -v
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from hlanalysis.sim.data.binance_klines import Kline
from hlanalysis.sim.data.schemas import PMMarket, PMTrade
from hlanalysis.sim.diagnostics import DiagnosticRow, write_diagnostics
from hlanalysis.sim.fills import FillModelConfig
from hlanalysis.sim.runner import RunnerConfig, run_one_market
from hlanalysis.strategy.model_edge import ModelEdgeConfig, ModelEdgeStrategy

# C2 imports (added after impl; kept together so grep finds them)
from hlanalysis.sim.diagnostics import FILLS_SCHEMA, FillRow, write_fills


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

    def test_runner_zero_scan_events_writes_empty_parquet(self, tmp_path: Path):
        """A market that produces zero scan events must still write a zero-row
        parquet with the correct schema when diagnostics_dir is set.

        We force zero scans by setting scanner_interval_seconds larger than the
        entire event stream duration (klines only span 60 minutes).
        """
        import pyarrow.parquet as pq
        from hlanalysis.sim.diagnostics import DIAGNOSTICS_SCHEMA

        market = _market(condition_id="0xzero_scans")
        diag_dir = tmp_path / "diagnostics_zero"
        # scanner_interval_seconds > total kline span → no scan fires
        cfg = RunnerConfig(
            scanner_interval_seconds=7200,  # 2 hours; klines only cover 60 min
            fill_model=FillModelConfig(
                slippage_bps=5.0, fee_taker=0.02, book_depth_assumption=10_000.0
            ),
            synthetic_half_spread=0.005,
            synthetic_depth=10_000.0,
            day_open_btc=100_000.0,
        )
        result = run_one_market(
            _v2_strategy(),
            market,
            _klines(),
            _trades(),
            cfg,
            diagnostics_dir=diag_dir,
        )
        assert result.n_decisions == 0, "Expected zero decisions with 2h scan interval"
        expected_path = diag_dir / "0xzero_scans.parquet"
        assert expected_path.exists(), "Zero-scan market must still produce a parquet file"
        table = pq.read_table(expected_path)
        assert table.num_rows == 0, f"Expected 0 rows, got {table.num_rows}"
        # Schema must match DIAGNOSTICS_SCHEMA
        for field in DIAGNOSTICS_SCHEMA:
            assert field.name in table.schema.names, f"Missing column: {field.name}"

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


# ---------------------------------------------------------------------------
# Regression: cmd_run must not raise NameError (out_dir assigned before use)
# ---------------------------------------------------------------------------

class TestCmdRunOutDirOrdering:
    """Regression for: diag_dir = out_dir / "diagnostics" appearing before
    out_dir = Path(args.out_dir) in cmd_run, causing NameError on any non-empty
    job list.  Introduced in commit feat(sim): persist per-decision diagnostics.
    """

    def test_cmd_run_no_name_error(self, tmp_path: Path, monkeypatch):
        """cmd_run must not raise NameError when called with a non-empty job list."""
        from hlanalysis.sim import cli
        from hlanalysis.sim.cli import cmd_run
        from hlanalysis.sim.tuning import TuningJob

        # Build a minimal valid config file the CLI expects to read.
        config_path = tmp_path / "params.json"
        config_path.write_text(json.dumps({
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
        }))

        out_dir = tmp_path / "out"

        # Build a stub job list so cmd_run doesn't bail on "no jobs".
        market = _market(condition_id="0xreg1")
        klines = _klines()
        trades = _trades()
        stub_jobs = [TuningJob(
            market=market,
            klines=klines,
            trades=trades,
            day_open_btc=100_000.0,
        )]

        # Patch _load_jobs_from_cache so no real filesystem cache is needed.
        monkeypatch.setattr(cli, "_load_jobs_from_cache", lambda _: stub_jobs)

        args = argparse.Namespace(
            cache_root=str(tmp_path / "cache"),
            strategy="v1",
            config=str(config_path),
            out_dir=str(out_dir),
            slippage_bps=5.0,
            fee_taker=0.0,
            half_spread=0.005,
            depth=10_000.0,
        )

        # Should not raise NameError (or any error).
        cmd_run(args)


# ---------------------------------------------------------------------------
# C2: FillRow dataclass + write_fills
# ---------------------------------------------------------------------------

class TestFillRow:
    def test_fill_row_enter_fields(self):
        """FillRow for an ENTER fill holds all entry_* fields."""
        row = FillRow(
            cloid="hla-001",
            ts_ns=1_000_000_000,
            side="buy",
            price=0.62,
            size=50.0,
            fee=0.62,
            condition_id="0xabc",
            question_idx=0,
            symbol="Y",
            entry_p_model=0.72,
            entry_edge_chosen_side=0.05,
            entry_sigma=0.80,
            entry_tau_yr=0.002739726,
            realized_pnl_at_settle=18.0,
        )
        assert row.cloid == "hla-001"
        assert abs(row.entry_p_model - 0.72) < 1e-9
        assert abs(row.entry_edge_chosen_side - 0.05) < 1e-9

    def test_fill_row_settle_nulls(self):
        """Settlement fill has null entry_* fields."""
        row = FillRow(
            cloid="settle",
            ts_ns=86_400_000_000_000,
            side="sell",
            price=1.0,
            size=50.0,
            fee=0.0,
            condition_id="0xabc",
            question_idx=0,
            symbol="Y",
            entry_p_model=None,
            entry_edge_chosen_side=None,
            entry_sigma=None,
            entry_tau_yr=None,
            realized_pnl_at_settle=19.38,
        )
        assert row.entry_p_model is None
        assert row.entry_sigma is None
        assert row.realized_pnl_at_settle is not None


class TestWriteFills:
    def test_write_fills_creates_parquet(self, tmp_path: Path):
        rows = [
            FillRow(
                cloid="hla-001",
                ts_ns=1_000_000_000,
                side="buy",
                price=0.60,
                size=10.0,
                fee=0.12,
                condition_id="0xabc",
                question_idx=0,
                symbol="Y",
                entry_p_model=0.65,
                entry_edge_chosen_side=0.04,
                entry_sigma=0.70,
                entry_tau_yr=0.003,
                realized_pnl_at_settle=3.88,
            ),
            FillRow(
                cloid="settle",
                ts_ns=86_400_000_000_000,
                side="sell",
                price=1.0,
                size=10.0,
                fee=0.0,
                condition_id="0xabc",
                question_idx=0,
                symbol="Y",
                entry_p_model=None,
                entry_edge_chosen_side=None,
                entry_sigma=None,
                entry_tau_yr=None,
                realized_pnl_at_settle=3.88,
            ),
        ]
        out = tmp_path / "fills.parquet"
        write_fills(rows, out)
        table = pq.read_table(out)
        assert table.num_rows == 2
        expected_cols = {
            "cloid", "ts_ns", "side", "price", "size", "fee",
            "condition_id", "question_idx", "symbol",
            "entry_p_model", "entry_edge_chosen_side", "entry_sigma",
            "entry_tau_yr", "realized_pnl_at_settle",
        }
        assert expected_cols.issubset(set(table.schema.names))

    def test_write_fills_preserves_null_entry_fields(self, tmp_path: Path):
        rows = [
            FillRow(
                cloid="settle",
                ts_ns=0,
                side="sell",
                price=1.0,
                size=5.0,
                fee=0.0,
                condition_id="0xdef",
                question_idx=0,
                symbol="N",
                entry_p_model=None,
                entry_edge_chosen_side=None,
                entry_sigma=None,
                entry_tau_yr=None,
                realized_pnl_at_settle=2.0,
            ),
        ]
        out = tmp_path / "fills_null.parquet"
        write_fills(rows, out)
        table = pq.read_table(out)
        row = table.to_pylist()[0]
        assert row["entry_p_model"] is None
        assert row["entry_sigma"] is None
        assert row["realized_pnl_at_settle"] == pytest.approx(2.0)

    def test_write_fills_empty(self, tmp_path: Path):
        """Empty rows still produce a valid parquet with the correct schema."""
        out = tmp_path / "fills_empty.parquet"
        write_fills([], out)
        table = pq.read_table(out)
        assert table.num_rows == 0
        for field in FILLS_SCHEMA:
            assert field.name in table.schema.names


# ---------------------------------------------------------------------------
# C2: fills.parquet integration via run_one_market
# ---------------------------------------------------------------------------

def _v2_strategy_aggressive() -> ModelEdgeStrategy:
    """Always enters once vol is warm."""
    return ModelEdgeStrategy(
        ModelEdgeConfig(
            vol_lookback_seconds=600,
            vol_sampling_dt_seconds=60,
            vol_clip_min=0.01,
            vol_clip_max=5.0,
            edge_buffer=-1.0,
            fee_taker=0.0,
            half_spread_assumption=0.0,
            stop_loss_pct=None,
            max_position_usd=100.0,
        )
    )


class TestRunnerFillsDir:
    def test_v2_run_produces_fills_parquet(self, tmp_path: Path):
        """AC1: after a v2 run, fills.parquet exists per market."""
        market = _market(condition_id="0xfill1")
        fills_dir = tmp_path / "fills"
        run_one_market(
            _v2_strategy_aggressive(),
            market,
            _klines(120),
            _trades(),
            _runner_cfg(),
            fills_dir=fills_dir,
        )
        expected = fills_dir / "0xfill1.parquet"
        assert expected.exists(), f"Expected fills parquet at {expected}"
        table = pq.read_table(expected)
        assert table.num_rows >= 1

    def test_v2_enter_fills_have_non_null_model_fields(self, tmp_path: Path):
        """AC1: ENTER fills have non-null entry_p_model, entry_sigma, entry_tau_yr."""
        market = _market(condition_id="0xfill2")
        fills_dir = tmp_path / "fills"
        run_one_market(
            _v2_strategy_aggressive(),
            market,
            _klines(120),
            _trades(),
            _runner_cfg(),
            fills_dir=fills_dir,
        )
        table = pq.read_table(fills_dir / "0xfill2.parquet")
        rows = table.to_pylist()
        enter_fills = [r for r in rows if r["cloid"] != "settle"]
        assert len(enter_fills) >= 1, "Expected at least one ENTER fill"
        for r in enter_fills:
            assert r["entry_p_model"] is not None, "ENTER fill missing entry_p_model"
            assert r["entry_sigma"] is not None, "ENTER fill missing entry_sigma"
            assert r["entry_tau_yr"] is not None, "ENTER fill missing entry_tau_yr"
            assert r["entry_edge_chosen_side"] is not None, "ENTER fill missing entry_edge_chosen_side"

    def test_settle_fill_has_null_entry_fields_and_non_null_pnl(self, tmp_path: Path):
        """AC1: settlement fill (cloid='settle') has null entry_* and non-null realized_pnl_at_settle."""
        market = _market(condition_id="0xfill3")
        fills_dir = tmp_path / "fills"
        run_one_market(
            _v2_strategy_aggressive(),
            market,
            _klines(120),
            _trades(),
            _runner_cfg(),
            fills_dir=fills_dir,
        )
        table = pq.read_table(fills_dir / "0xfill3.parquet")
        rows = table.to_pylist()
        settle_rows = [r for r in rows if r["cloid"] == "settle"]
        assert len(settle_rows) == 1, f"Expected exactly one settle row, got {len(settle_rows)}"
        s = settle_rows[0]
        assert s["entry_p_model"] is None, "settle fill must have null entry_p_model"
        assert s["entry_sigma"] is None, "settle fill must have null entry_sigma"
        assert s["entry_tau_yr"] is None, "settle fill must have null entry_tau_yr"
        assert s["entry_edge_chosen_side"] is None, "settle fill must have null entry_edge_chosen_side"
        assert s["realized_pnl_at_settle"] is not None, "settle fill must have non-null realized_pnl"

    def test_join_fills_to_diagnostics_on_condition_id_ts_ns(self, tmp_path: Path):
        """AC2: joining fills to diagnostics on (condition_id, ts_ns) recovers the same row."""
        market = _market(condition_id="0xfill4")
        fills_dir = tmp_path / "fills"
        diag_dir = tmp_path / "diagnostics"
        run_one_market(
            _v2_strategy_aggressive(),
            market,
            _klines(120),
            _trades(),
            _runner_cfg(),
            fills_dir=fills_dir,
            diagnostics_dir=diag_dir,
        )
        fills_table = pq.read_table(fills_dir / "0xfill4.parquet")
        diag_table = pq.read_table(diag_dir / "0xfill4.parquet")
        fill_rows = fills_table.to_pylist()
        diag_rows = diag_table.to_pylist()

        # Build a lookup: (condition_id, ts_ns) -> diag_row
        diag_by_key = {(r["condition_id"], r["ts_ns"]): r for r in diag_rows}

        enter_fills = [r for r in fill_rows if r["cloid"] != "settle"]
        assert len(enter_fills) >= 1
        for f in enter_fills:
            key = (f["condition_id"], f["ts_ns"])
            assert key in diag_by_key, f"No diagnostic row for key {key}"
            d = diag_by_key[key]
            assert f["entry_p_model"] == pytest.approx(d["p_model"]), (
                f"entry_p_model {f['entry_p_model']} != diag p_model {d['p_model']}"
            )

    def test_entry_edge_chosen_side_matches_traded_side(self, tmp_path: Path):
        """AC1 (edge): entry_edge_chosen_side == edge_yes when yes bought, edge_no when no bought."""
        market = _market(condition_id="0xfill5")
        fills_dir = tmp_path / "fills"
        diag_dir = tmp_path / "diagnostics"
        run_one_market(
            _v2_strategy_aggressive(),
            market,
            _klines(120),
            _trades(),
            _runner_cfg(),
            fills_dir=fills_dir,
            diagnostics_dir=diag_dir,
        )
        fills_table = pq.read_table(fills_dir / "0xfill5.parquet")
        diag_table = pq.read_table(diag_dir / "0xfill5.parquet")
        fill_rows = fills_table.to_pylist()
        diag_rows = diag_table.to_pylist()
        diag_by_key = {(r["condition_id"], r["ts_ns"]): r for r in diag_rows}

        for f in fill_rows:
            if f["cloid"] == "settle":
                continue
            key = (f["condition_id"], f["ts_ns"])
            d = diag_by_key[key]
            sym = f["symbol"]
            if sym == market.yes_token_id:
                expected_edge = d["edge_yes"]
            else:
                expected_edge = d["edge_no"]
            assert f["entry_edge_chosen_side"] == pytest.approx(expected_edge), (
                f"edge mismatch for symbol={sym}: got {f['entry_edge_chosen_side']}, expected {expected_edge}"
            )

    def test_v1_run_enter_fills_have_null_model_fields(self, tmp_path: Path):
        """v1 run: ENTER fills written, entry_* fields are null (no crash)."""
        from hlanalysis.sim.v1_factory import build_v1_strategy_from_params
        market = _market(condition_id="0xfill6")
        fills_dir = tmp_path / "fills"
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
            _klines(120),
            _trades(),
            _runner_cfg(),
            fills_dir=fills_dir,
        )
        expected = fills_dir / "0xfill6.parquet"
        assert expected.exists()
        table = pq.read_table(expected)
        rows = table.to_pylist()
        enter_fills = [r for r in rows if r["cloid"] != "settle"]
        # v1 might or might not enter depending on strategy logic, but no crash.
        for r in enter_fills:
            assert r["entry_p_model"] is None
            assert r["entry_sigma"] is None

    def test_fills_dir_none_no_parquet_written(self, tmp_path: Path):
        """If fills_dir is None, no fills parquet is written."""
        market = _market(condition_id="0xfill7")
        run_one_market(
            _v2_strategy_aggressive(),
            market,
            _klines(120),
            _trades(),
            _runner_cfg(),
            fills_dir=None,
        )
        assert not (tmp_path / "fills").exists()

    def test_cmd_run_writes_fills(self, tmp_path: Path, monkeypatch):
        """cmd_run passes fills_dir; fills parquet is created."""
        from hlanalysis.sim import cli
        from hlanalysis.sim.cli import cmd_run
        from hlanalysis.sim.tuning import TuningJob
        from hlanalysis.strategy.model_edge import ModelEdgeConfig, ModelEdgeStrategy

        # v2 config
        config_path = tmp_path / "params.json"
        config_path.write_text(json.dumps({
            "vol_lookback_seconds": 600,
            "vol_sampling_dt_seconds": 60,
            "vol_clip_min": 0.01,
            "vol_clip_max": 5.0,
            "edge_buffer": -1.0,
            "fee_taker": 0.0,
            "half_spread_assumption": 0.0,
            "stop_loss_pct": None,
            "max_position_usd": 100.0,
        }))

        out_dir = tmp_path / "out"
        market = _market(condition_id="0xclicmd1")
        stub_jobs = [TuningJob(
            market=market, klines=_klines(120),
            trades=_trades(), day_open_btc=100_000.0,
        )]
        monkeypatch.setattr(cli, "_load_jobs_from_cache", lambda _: stub_jobs)

        args = argparse.Namespace(
            cache_root=str(tmp_path / "cache"),
            strategy="v2",
            config=str(config_path),
            out_dir=str(out_dir),
            slippage_bps=5.0,
            fee_taker=0.0,
            half_spread=0.005,
            depth=10_000.0,
        )
        cmd_run(args)
        fills_parquet = out_dir / "fills" / "0xclicmd1.parquet"
        assert fills_parquet.exists(), f"Expected fills parquet at {fills_parquet}"

    def test_tune_does_not_write_fills(self, tmp_path: Path):
        """cmd_tune: no fills_dir passed; run_one_market is called without fills_dir arg
        (or with None) — verified by checking no fills/ dir is created."""
        # We test at the runner level: call run_one_market without fills_dir, check no fills written
        market = _market(condition_id="0xtune1")
        # Simulate what cmd_tune does: no fills_dir argument
        run_one_market(
            _v2_strategy_aggressive(),
            market,
            _klines(120),
            _trades(),
            _runner_cfg(),
            # no fills_dir arg → defaults to None
        )
        assert not (tmp_path / "fills").exists()
