from __future__ import annotations

from pathlib import Path

from hlanalysis.sim.tuning_config import load_tuning_yaml


def test_load_tuning_yaml_round_trip(tmp_path: Path):
    p = tmp_path / "t.yaml"
    p.write_text(
        "v2_grid:\n  edge_buffer: [0.01, 0.02]\n  stop_loss_pct: [10, null]\n"
        "run:\n  train_markets: 60\n  test_markets: 15\n  step_markets: 15\n  max_workers: 4\n"
    )
    cfg = load_tuning_yaml(p)
    assert cfg.v2_grid["edge_buffer"] == [0.01, 0.02]
    assert cfg.v2_grid["stop_loss_pct"] == [10, None]
    assert cfg.run["max_workers"] == 4
