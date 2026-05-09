from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from hlanalysis.sim.metrics import RunSummary
from hlanalysis.sim.report import write_single_run_report, write_tuning_report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_summary(**kwargs) -> RunSummary:
    defaults = dict(n_markets=10, n_trades=5, total_pnl_usd=42.0,
                    sharpe=1.2, hit_rate=0.6, max_drawdown_usd=10.0)
    defaults.update(kwargs)
    return RunSummary(**defaults)


def _make_fills_parquet(path: Path, fills: list[dict]) -> None:
    """Write a minimal fills.parquet at *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([
        pa.field("cloid",                   pa.string()),
        pa.field("ts_ns",                   pa.int64()),
        pa.field("side",                    pa.string()),
        pa.field("price",                   pa.float64()),
        pa.field("size",                    pa.float64()),
        pa.field("fee",                     pa.float64()),
        pa.field("condition_id",            pa.string()),
        pa.field("question_idx",            pa.int64()),
        pa.field("symbol",                  pa.string()),
        pa.field("entry_p_model",           pa.float64()),
        pa.field("entry_edge_chosen_side",  pa.float64()),
        pa.field("entry_sigma",             pa.float64()),
        pa.field("entry_tau_yr",            pa.float64()),
        pa.field("realized_pnl_at_settle",  pa.float64()),
    ])
    if not fills:
        table = pa.table({f.name: pa.array([], type=f.type) for f in schema}, schema=schema)
    else:
        cols: dict = {f.name: [] for f in schema}
        for row in fills:
            for f in schema:
                cols[f.name].append(row.get(f.name))
        arrays = {name: pa.array(vals, type=schema.field(name).type) for name, vals in cols.items()}
        table = pa.table(arrays, schema=schema)
    pq.write_table(table, path)


@dataclass
class _FakeMarket:
    condition_id: str
    resolved_outcome: str
    start_ts_ns: int
    end_ts_ns: int


# ---------------------------------------------------------------------------
# Existing tests (preserved)
# ---------------------------------------------------------------------------

def test_single_run_report_writes_markdown_and_plot(tmp_path: Path):
    s = _make_summary()
    out = write_single_run_report(
        out_dir=tmp_path,
        strategy_name="model_edge",
        config_summary={"edge_buffer": 0.02, "stop_loss_pct": 10},
        per_market_pnl=[1, -0.5, 2, 0, 1.5, -1, 0.8, 1.1, -0.3, 0.4],
        summary=s,
    )
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "equity_curve.html").exists()


def test_tuning_report_writes_top_k(tmp_path: Path):
    rows = [
        {"params": {"edge_buffer": 0.01, "stop_loss_pct": 10}, "summary": {"sharpe": 0.5, "total_pnl_usd": 100, "n_trades": 10, "hit_rate": 0.6, "n_markets": 30, "max_drawdown_usd": 20}},
        {"params": {"edge_buffer": 0.02, "stop_loss_pct": 10}, "summary": {"sharpe": 1.2, "total_pnl_usd": 200, "n_trades": 8, "hit_rate": 0.7, "n_markets": 30, "max_drawdown_usd": 15}},
    ]
    write_tuning_report(out_dir=tmp_path, strategy_name="model_edge", rows=rows, top_k=2)
    assert (tmp_path / "report.md").exists()


# ---------------------------------------------------------------------------
# C5+C14 tests
# ---------------------------------------------------------------------------

# --- AC1: sections present ------------------------------------------------

def test_run_context_section_present(tmp_path: Path):
    """report.md must contain '## Run context' section."""
    s = _make_summary()
    write_single_run_report(
        out_dir=tmp_path,
        strategy_name="model_edge",
        config_summary={"edge_buffer": 0.02},
        per_market_pnl=[1.0, -0.5],
        summary=s,
    )
    text = (tmp_path / "report.md").read_text()
    assert "## Run context" in text


def test_per_market_section_present(tmp_path: Path):
    """report.md must contain '## Per-market' section."""
    s = _make_summary()
    write_single_run_report(
        out_dir=tmp_path,
        strategy_name="model_edge",
        config_summary={"edge_buffer": 0.02},
        per_market_pnl=[1.0, -0.5],
        summary=s,
    )
    text = (tmp_path / "report.md").read_text()
    assert "## Per-market" in text


# --- AC2: run context contents -------------------------------------------

def test_run_context_contains_data_range(tmp_path: Path):
    """Run context block must contain a data range string in ISO8601 UTC format."""
    ns_start = int(1_713_139_200 * 1e9)  # 2024-04-15 00:00 UTC
    ns_end   = int(1_746_748_800 * 1e9)  # 2025-05-09 00:00 UTC
    markets = [_FakeMarket("cid_a", "yes", ns_start, ns_end)]
    s = _make_summary(n_markets=1)
    write_single_run_report(
        out_dir=tmp_path,
        strategy_name="model_edge",
        config_summary={"edge_buffer": 0.02},
        per_market_pnl=[1.0],
        summary=s,
        markets=markets,
    )
    text = (tmp_path / "report.md").read_text()
    assert "2024-04-15" in text
    assert "UTC" in text


def test_run_context_contains_fee_and_slippage(tmp_path: Path):
    """Run context must include fee_taker, slippage_bps, half_spread."""
    s = _make_summary()
    write_single_run_report(
        out_dir=tmp_path,
        strategy_name="model_edge",
        config_summary={"edge_buffer": 0.02},
        per_market_pnl=[1.0],
        summary=s,
        fee_taker=0.03,
        slippage_bps=7.5,
        half_spread=0.008,
    )
    text = (tmp_path / "report.md").read_text()
    assert "0.03" in text
    assert "7.5" in text
    assert "0.008" in text


def test_run_context_contains_config_hash(tmp_path: Path):
    """Run context must include the first 12 chars of SHA-256 of config JSON."""
    cfg = {"edge_buffer": 0.02, "stop_loss_pct": 10}
    expected_hash = hashlib.sha256(
        json.dumps(cfg, sort_keys=True).encode()
    ).hexdigest()[:12]
    s = _make_summary()
    write_single_run_report(
        out_dir=tmp_path,
        strategy_name="model_edge",
        config_summary=cfg,
        per_market_pnl=[1.0],
        summary=s,
    )
    text = (tmp_path / "report.md").read_text()
    assert expected_hash in text


# --- AC3: per-market table contents --------------------------------------

def test_per_market_table_one_row_per_market(tmp_path: Path):
    """Table has one data row per market."""
    markets = [
        _FakeMarket("cid_aaa", "yes",  int(1e18), int(2e18)),
        _FakeMarket("cid_bbb", "no",   int(1e18), int(2e18)),
    ]
    fills_dir = tmp_path / "fills"
    # cid_aaa: one enter fill
    _make_fills_parquet(fills_dir / "cid_aaa.parquet", [
        dict(cloid="c1", ts_ns=int(1.1e18), side="buy", price=0.6, size=100.0, fee=0.0,
             condition_id="cid_aaa", question_idx=0, symbol="tok_yes",
             entry_p_model=0.65, entry_edge_chosen_side=0.05,
             entry_sigma=0.2, entry_tau_yr=0.003,
             realized_pnl_at_settle=4.0),
        dict(cloid="settle", ts_ns=int(2e18), side="sell", price=1.0, size=100.0, fee=0.0,
             condition_id="cid_aaa", question_idx=0, symbol="tok_yes",
             entry_p_model=None, entry_edge_chosen_side=None,
             entry_sigma=None, entry_tau_yr=None,
             realized_pnl_at_settle=4.0),
    ])
    # cid_bbb: no trades
    _make_fills_parquet(fills_dir / "cid_bbb.parquet", [])

    s = _make_summary(n_markets=2)
    write_single_run_report(
        out_dir=tmp_path,
        strategy_name="model_edge",
        config_summary={"edge_buffer": 0.02},
        per_market_pnl=[4.0, 0.0],
        summary=s,
        markets=markets,
        fills_dir=fills_dir,
    )
    text = (tmp_path / "report.md").read_text()
    assert "cid_aaa" in text
    assert "cid_bbb" in text


def test_per_market_table_column_order(tmp_path: Path):
    """Table header must contain the required columns in order."""
    markets = [_FakeMarket("cid_x", "yes", int(1e18), int(2e18))]
    s = _make_summary(n_markets=1)
    write_single_run_report(
        out_dir=tmp_path,
        strategy_name="model_edge",
        config_summary={"edge_buffer": 0.02},
        per_market_pnl=[0.0],
        summary=s,
        markets=markets,
    )
    text = (tmp_path / "report.md").read_text()
    header_line = next(l for l in text.splitlines() if "condition_id" in l)
    cols = [c.strip() for c in header_line.split("|") if c.strip()]
    expected = ["condition_id", "outcome", "n_trades", "first_entry_ts",
                "last_exit_ts", "realized_pnl_usd", "calibration_residual"]
    assert cols == expected


# --- AC4: calibration residual -------------------------------------------

def test_calibration_residual_non_null_for_v2(tmp_path: Path):
    """calibration_residual is a numeric value when entry_edge_chosen_side is set (v2)."""
    markets = [_FakeMarket("cid_v2", "yes", int(1e18), int(2e18))]
    fills_dir = tmp_path / "fills"
    # realized_pnl_at_settle / (price * size) - entry_edge_chosen_side
    # = 10.0 / (0.60 * 100.0) - 0.05 = 0.1667 - 0.05 = 0.1167
    _make_fills_parquet(fills_dir / "cid_v2.parquet", [
        dict(cloid="c1", ts_ns=int(1.1e18), side="buy", price=0.60, size=100.0, fee=0.0,
             condition_id="cid_v2", question_idx=0, symbol="tok_yes",
             entry_p_model=0.65, entry_edge_chosen_side=0.05,
             entry_sigma=0.2, entry_tau_yr=0.003,
             realized_pnl_at_settle=10.0),
        dict(cloid="settle", ts_ns=int(2e18), side="sell", price=1.0, size=100.0, fee=0.0,
             condition_id="cid_v2", question_idx=0, symbol="tok_yes",
             entry_p_model=None, entry_edge_chosen_side=None,
             entry_sigma=None, entry_tau_yr=None,
             realized_pnl_at_settle=10.0),
    ])
    s = _make_summary(n_markets=1)
    write_single_run_report(
        out_dir=tmp_path,
        strategy_name="v2",
        config_summary={"edge_buffer": 0.02},
        per_market_pnl=[10.0],
        summary=s,
        markets=markets,
        fills_dir=fills_dir,
    )
    text = (tmp_path / "report.md").read_text()
    # Find the row for cid_v2 and check calibration residual is not empty/null
    cid_line = next(l for l in text.splitlines() if "cid_v2" in l)
    cells = [c.strip() for c in cid_line.split("|") if c.strip()]
    # last cell is calibration_residual
    calib_cell = cells[-1]
    assert calib_cell not in ("", "—", "null", "None")
    # should be parseable as float
    float(calib_cell)


def test_calibration_residual_null_for_v1(tmp_path: Path):
    """calibration_residual is '—' (null marker) when entry_edge_chosen_side is absent (v1)."""
    markets = [_FakeMarket("cid_v1", "yes", int(1e18), int(2e18))]
    fills_dir = tmp_path / "fills"
    # v1: entry_edge_chosen_side is None
    _make_fills_parquet(fills_dir / "cid_v1.parquet", [
        dict(cloid="c1", ts_ns=int(1.1e18), side="buy", price=0.60, size=100.0, fee=0.0,
             condition_id="cid_v1", question_idx=0, symbol="tok_yes",
             entry_p_model=None, entry_edge_chosen_side=None,
             entry_sigma=None, entry_tau_yr=None,
             realized_pnl_at_settle=5.0),
        dict(cloid="settle", ts_ns=int(2e18), side="sell", price=1.0, size=100.0, fee=0.0,
             condition_id="cid_v1", question_idx=0, symbol="tok_yes",
             entry_p_model=None, entry_edge_chosen_side=None,
             entry_sigma=None, entry_tau_yr=None,
             realized_pnl_at_settle=5.0),
    ])
    s = _make_summary(n_markets=1)
    write_single_run_report(
        out_dir=tmp_path,
        strategy_name="v1",
        config_summary={"edge_buffer": 0.02},
        per_market_pnl=[5.0],
        summary=s,
        markets=markets,
        fills_dir=fills_dir,
    )
    text = (tmp_path / "report.md").read_text()
    cid_line = next(l for l in text.splitlines() if "cid_v1" in l)
    cells = [c.strip() for c in cid_line.split("|") if c.strip()]
    calib_cell = cells[-1]
    assert calib_cell == "—"


# --- AC5: zero-trade markets ---------------------------------------------

def test_zero_trade_market_renders_row(tmp_path: Path):
    """Market with no trades renders as a row with n_trades=0 and blank entry/exit cols."""
    markets = [_FakeMarket("cid_empty", "unknown", int(1e18), int(2e18))]
    fills_dir = tmp_path / "fills"
    _make_fills_parquet(fills_dir / "cid_empty.parquet", [])

    s = _make_summary(n_markets=1)
    write_single_run_report(
        out_dir=tmp_path,
        strategy_name="v1",
        config_summary={"edge_buffer": 0.02},
        per_market_pnl=[0.0],
        summary=s,
        markets=markets,
        fills_dir=fills_dir,
    )
    text = (tmp_path / "report.md").read_text()
    assert "cid_empty" in text
    cid_line = next(l for l in text.splitlines() if "cid_empty" in l)
    # Split on | and strip each cell, keeping empty strings (don't filter)
    # The line format is: | cell | cell | ... |
    # Splitting on | gives: ['', ' cell ', ' cell ', ..., '']
    parts = [c.strip() for c in cid_line.split("|")]
    # parts[0] is '' (before leading |), parts[-1] is '' (after trailing |)
    cells = parts[1:-1]
    # column order: condition_id | outcome | n_trades | first_entry_ts | last_exit_ts | realized_pnl_usd | calibration_residual
    assert cells[2] == "0"       # n_trades
    assert cells[3] == ""        # first_entry_ts blank
    assert cells[4] == ""        # last_exit_ts blank


# --- backward compat: existing fields still rendered ---------------------

def test_existing_summary_fields_preserved(tmp_path: Path):
    """Existing Summary section and equity curve HTML still produced."""
    s = _make_summary(n_markets=3, n_trades=6, total_pnl_usd=99.0, sharpe=0.8)
    write_single_run_report(
        out_dir=tmp_path,
        strategy_name="model_edge",
        config_summary={"edge_buffer": 0.02},
        per_market_pnl=[1.0, -0.5, 2.0],
        summary=s,
    )
    text = (tmp_path / "report.md").read_text()
    assert "## Summary" in text
    assert "$99.00" in text
    assert (tmp_path / "equity_curve.html").exists()
