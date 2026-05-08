from __future__ import annotations

from pathlib import Path

from hlanalysis.sim.metrics import RunSummary
from hlanalysis.sim.report import write_single_run_report, write_tuning_report


def test_single_run_report_writes_markdown_and_plot(tmp_path: Path):
    s = RunSummary(n_markets=10, n_trades=5, total_pnl_usd=42.0,
                   sharpe=1.2, hit_rate=0.6, max_drawdown_usd=10.0)
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
