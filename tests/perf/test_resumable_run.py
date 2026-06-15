"""Tests for the resumable parallel backtest driver's supervisor logic.

Covers the crash-tolerance core (no real backtests run):
  - classify(): success / retryable / deterministic decisions
  - parse_pnl(): reading total PnL + trades from a report.md
  - Driver resume: a (config, question) with a .done marker is skipped
  - Driver retry/fail: retryable failures requeue up to max_retries, then fail
  - 2D sweep: jobs = configs × questions; per-config env/slot_config wiring
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "resumable_run",
    Path(__file__).resolve().parents[2] / "scripts" / "perf" / "resumable_run.py",
)
rr = importlib.util.module_from_spec(_SPEC)
sys.modules["resumable_run"] = rr  # so @dataclass can resolve cls.__module__
_SPEC.loader.exec_module(rr)


# --- classify -------------------------------------------------------------


class TestClassify:
    def test_success(self):
        assert rr.classify(0, "", report_exists=True) == rr.SUCCESS

    def test_exit0_no_report_is_retryable(self):
        assert rr.classify(0, "", report_exists=False) == rr.RETRYABLE

    def test_killed_by_signal_retryable(self):
        assert rr.classify(-9, "", report_exists=False) == rr.RETRYABLE

    def test_oom_codes_retryable(self):
        assert rr.classify(137, "", report_exists=False) == rr.RETRYABLE
        assert rr.classify(143, "", report_exists=False) == rr.RETRYABLE

    def test_timeout_code_retryable(self):
        assert rr.classify(124, "", report_exists=False) == rr.RETRYABLE

    def test_memoryerror_text_retryable(self):
        assert rr.classify(1, "boom\nMemoryError: out of memory", report_exists=False) == rr.RETRYABLE

    def test_traceback_is_deterministic(self):
        log = "Traceback (most recent call last):\n  ...\nKeyError: 'x'"
        assert rr.classify(1, log, report_exists=False) == rr.DETERMINISTIC

    def test_unknown_nonzero_defaults_retryable(self):
        assert rr.classify(2, "some opaque failure", report_exists=False) == rr.RETRYABLE


# --- parse_pnl ------------------------------------------------------------


def _write_report(d: Path, pnl: str, trades: int) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "report.md").write_text(
        f"# run\n\n## Summary\n\n- questions: 1\n- trades: {trades}\n- total PnL: ${pnl}\n- Sharpe (annualized 365): 1.0\n"
    )


class TestParsePnl:
    def test_parses_pnl_and_trades(self, tmp_path):
        _write_report(tmp_path, "1,234.56", 42)
        pnl, ntr = rr.parse_pnl(tmp_path)
        assert pnl == pytest.approx(1234.56)
        assert ntr == 42

    def test_missing_report(self, tmp_path):
        assert rr.parse_pnl(tmp_path) == (None, None)


# --- shared fixtures ------------------------------------------------------


def _args(out_base: Path, **kw) -> SimpleNamespace:
    base = dict(
        kind="binary",
        start="2026-05-06",
        end="2026-06-11",
        out_base=str(out_base),
        configs=None,
        slot="v31",
        slot_config=None,
        slot_class=None,
        strategy=None,
        workers=2,
        max_retries=2,
        timeout=3600.0,
        scan_min=None,
        scan_max=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _one_cfg():
    return [rr.Config(id="base")]


class _P:
    def __init__(self, pid=4242):
        self.pid = pid


# --- Driver resume --------------------------------------------------------


class TestDriverResume:
    def test_completed_cells_are_skipped(self, tmp_path):
        drv0 = rr.Driver(_args(tmp_path), _one_cfg(), n_questions=3)
        d0 = drv0.qdir("base", 0)
        _write_report(d0, "100.00", 5)
        (d0 / ".done").write_text("1")
        drv = rr.Driver(_args(tmp_path), _one_cfg(), n_questions=3)
        assert drv.states[("base", 0)].status == "done"
        assert drv.states[("base", 0)].pnl == pytest.approx(100.0)
        assert ("base", 0) not in drv.queue
        assert drv.queue == [("base", 1), ("base", 2)]

    def test_fresh_run_queues_all(self, tmp_path):
        drv = rr.Driver(_args(tmp_path), _one_cfg(), n_questions=3)
        assert drv.queue == [("base", 0), ("base", 1), ("base", 2)]


# --- Driver retry / fail / success ----------------------------------------


class TestDriverFinish:
    def test_retryable_failure_requeues_then_fails(self, tmp_path):
        drv = rr.Driver(_args(tmp_path, max_retries=1), _one_cfg(), n_questions=1)
        key = ("base", 0)
        out = drv.qdir(*key)
        out.mkdir(parents=True, exist_ok=True)
        (out / "run.log").write_text("opaque crash")

        drv.queue.clear()
        drv.running[key] = (_P(), rr.time.time(), out)
        drv.states[key].status = "running"
        drv.states[key].attempts = 1
        drv._finish(key, returncode=2)
        assert drv.states[key].status == "pending"
        assert key in drv.queue

        drv.queue.clear()
        drv.running[key] = (_P(), rr.time.time(), out)
        drv.states[key].status = "running"
        drv.states[key].attempts = 2
        drv._finish(key, returncode=2)
        assert drv.states[key].status == "failed"
        assert key not in drv.queue

    def test_deterministic_failure_not_retried(self, tmp_path):
        drv = rr.Driver(_args(tmp_path, max_retries=5), _one_cfg(), n_questions=1)
        key = ("base", 0)
        out = drv.qdir(*key)
        out.mkdir(parents=True, exist_ok=True)
        (out / "run.log").write_text("Traceback (most recent call last):\nValueError: bad")
        drv.queue.clear()
        drv.running[key] = (_P(), rr.time.time(), out)
        drv.states[key].status = "running"
        drv.states[key].attempts = 1
        drv._finish(key, returncode=1)
        assert drv.states[key].status == "failed"
        assert key not in drv.queue

    def test_success_marks_done(self, tmp_path):
        drv = rr.Driver(_args(tmp_path), _one_cfg(), n_questions=1)
        key = ("base", 0)
        out = drv.qdir(*key)
        _write_report(out, "55.50", 7)
        drv.running[key] = (_P(), rr.time.time(), out)
        drv.states[key].status = "running"
        drv.states[key].attempts = 1
        drv._finish(key, returncode=0)
        assert drv.states[key].status == "done"
        assert (out / ".done").exists()
        assert drv.states[key].pnl == pytest.approx(55.50)


# --- chunk math -----------------------------------------------------------


class TestChunkMath:
    def test_num_chunks_exact_multiple(self):
        assert rr.num_chunks(50, 25) == 2

    def test_num_chunks_partial_last(self):
        assert rr.num_chunks(51, 25) == 3
        assert rr.num_chunks(1, 25) == 1

    def test_num_chunks_zero(self):
        assert rr.num_chunks(0, 25) == 0

    def test_num_chunks_size_one_is_per_question(self):
        assert rr.num_chunks(7, 1) == 7

    def test_chunk_bounds_full_and_partial(self):
        assert rr.chunk_bounds(0, 51, 25) == (0, 25)
        assert rr.chunk_bounds(1, 51, 25) == (25, 25)
        assert rr.chunk_bounds(2, 51, 25) == (50, 1)


# --- 2D sweep -------------------------------------------------------------


class TestSweep2D:
    def test_jobs_are_config_x_question(self, tmp_path):
        cfgs = [rr.Config(id="a"), rr.Config(id="b")]
        drv = rr.Driver(_args(tmp_path), cfgs, n_questions=3)
        # 2 configs × 3 questions = 6 jobs
        assert len(drv.states) == 6
        assert set(drv.queue) == {(c, i) for c in ("a", "b") for i in range(3)}

    def test_per_config_resume_independent(self, tmp_path):
        cfgs = [rr.Config(id="a"), rr.Config(id="b")]
        drv0 = rr.Driver(_args(tmp_path), cfgs, n_questions=2)
        da = drv0.qdir("a", 1)
        _write_report(da, "10.00", 1)
        (da / ".done").write_text("1")
        drv = rr.Driver(_args(tmp_path), cfgs, n_questions=2)
        assert drv.states[("a", 1)].status == "done"
        assert ("a", 1) not in drv.queue
        # b's questions and a's other question still queued
        assert set(drv.queue) == {("a", 0), ("b", 0), ("b", 1)}

    def test_build_cmd_applies_config_slot_and_cadence(self, tmp_path):
        cfg = rr.Config(id="roi", slot_config="/tmp/variant.yaml", env={"HLBT_DEPTH_BACKEND": "roi"}, scan_min=1.0, scan_max=5.0)
        cmd = rr.build_cmd(_args(tmp_path), cfg, idx=3, out_dir=tmp_path / "o")
        assert "--slot" in cmd and "v31" in cmd
        assert cmd[cmd.index("--slot-config") + 1] == "/tmp/variant.yaml"
        assert cmd[cmd.index("--skip-markets") + 1] == "3"
        assert cmd[cmd.index("--scan-min-interval-seconds") + 1] == "1.0"
        assert cmd[cmd.index("--scan-max-interval-seconds") + 1] == "5.0"

    def test_load_configs_single_vs_sweep(self, tmp_path):
        # single config (no --configs)
        cfgs = rr.load_configs(_args(tmp_path, configs=None, slot_config="/x.yaml"))
        assert len(cfgs) == 1 and cfgs[0].id == "base" and cfgs[0].slot_config == "/x.yaml"
        # sweep from JSON
        import json

        cfg_json = tmp_path / "configs.json"
        cfg_json.write_text(json.dumps([{"id": "a", "slot_config": "/a.yaml"}, {"id": "b", "env": {"K": "v"}}]))
        cfgs = rr.load_configs(_args(tmp_path, configs=str(cfg_json)))
        assert [c.id for c in cfgs] == ["a", "b"]
        assert cfgs[0].slot_config == "/a.yaml"
        assert cfgs[1].env == {"K": "v"}
