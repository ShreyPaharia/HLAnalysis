from __future__ import annotations

import json

from hlanalysis.sim.tuning import _cell_key, _load_completed_cells, iter_grid


def test_iter_grid_yields_cartesian_product():
    g = {"a": [1, 2], "b": [3]}
    out = list(iter_grid(g))
    assert out == [{"a": 1, "b": 3}, {"a": 2, "b": 3}]


def test_iter_grid_with_three_axes_total_count():
    g = {"a": [1, 2, 3], "b": [4, 5], "c": [6, 7]}
    assert len(list(iter_grid(g))) == 12


def test_load_completed_cells_round_trip(tmp_path):
    log = tmp_path / "results.jsonl"
    log.write_text(
        json.dumps({"params": {"a": 1, "b": 2}, "test_ids": ["m1", "m2"], "summary": {}}) + "\n"
        + json.dumps({"params": {"a": 1, "b": 3}, "test_ids": ["m1", "m2"], "summary": {}}) + "\n"
    )
    cells = _load_completed_cells(log)
    assert _cell_key({"a": 1, "b": 2}, ["m1", "m2"]) in cells
    assert _cell_key({"a": 1, "b": 3}, ["m1", "m2"]) in cells
    assert _cell_key({"a": 1, "b": 2}, ["m1", "m3"]) not in cells


def test_load_completed_cells_empty_when_no_log(tmp_path):
    assert _load_completed_cells(tmp_path / "missing.jsonl") == set()
