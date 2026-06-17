"""Tests for the warm-chunk resumable backtest driver (no real backtests run).

Covers the supervisor + worker-mode logic:
  - classify(): success / retryable / deterministic decisions
  - parse_pnl(): reading total PnL + trades from a report.md
  - chunk math: num_chunks / chunk_bounds
  - chunk-keyed supervisor: queue of chunk indices, chunk done when all cells
    (.done) exist, per-chunk retry/fail
  - worker mode: question-outer/config-inner loop, memo env, per-cell skip,
    per-config env overlay, non-zero rc on cell failure
  - aggregate from report dirs; build_run_argv / config persistence / CLI args
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def _restore_memo_env():
    """run_worker_chunk sets HLBT_INPROC_BUNDLE_MEMO=1 on os.environ (fine in its
    own worker subprocess, but it would leak into the shared pytest process and
    flip the bundle memo ON for unrelated cache tests). Snapshot + restore."""
    keys = ("HLBT_INPROC_BUNDLE_MEMO", "HLBT_INPROC_BUNDLE_MEMO_WORKERS")
    saved = {k: os.environ.get(k) for k in keys}
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


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
        chunk_size=25,
        # PM / fee / data-source knobs (defaults mirror the CLI).
        data_source="hl_hip4",
        pm_flavor="btc_updown",
        pm_book_source="synthetic",
        pm_reference_source="klines",
        pm_binance_bbo_product_type="perp",
        fee_model="flat",
        fee_rate=0.07,
        cache_root=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _one_cfg():
    return [rr.Config(id="base")]


class _P:
    def __init__(self, pid=4242):
        self.pid = pid


# --- Driver chunk queue + resume ------------------------------------------


class TestDriverChunkQueue:
    def test_fresh_run_queues_all_chunks(self, tmp_path):
        # 3 questions, K=2 → 2 chunks; queue holds chunk indices
        drv = rr.Driver(_args(tmp_path, chunk_size=2), _one_cfg(), n_questions=3)
        assert drv.queue == [0, 1]
        assert set(drv.states) == {0, 1}

    def test_chunk_done_when_all_cells_done(self, tmp_path):
        cfgs = [rr.Config(id="a"), rr.Config(id="b")]
        # chunk 0 (K=2) = questions 0,1 × configs a,b = 4 cells
        for cid in ("a", "b"):
            for q in (0, 1):
                d = rr.qdir(Path(tmp_path), cid, q)
                _write_report(d, "1.00", 1)
                (d / ".done").write_text("1")
        drv = rr.Driver(_args(tmp_path, chunk_size=2), cfgs, n_questions=3)
        assert drv.states[0].status == "done"
        assert 0 not in drv.queue
        assert drv.queue == [1]

    def test_chunk_pending_if_one_cell_missing(self, tmp_path):
        cfgs = [rr.Config(id="a"), rr.Config(id="b")]
        # only 3 of 4 cells done
        for cid, q in [("a", 0), ("a", 1), ("b", 0)]:
            d = rr.qdir(Path(tmp_path), cid, q)
            _write_report(d, "1.00", 1)
            (d / ".done").write_text("1")
        drv = rr.Driver(_args(tmp_path, chunk_size=2), cfgs, n_questions=3)
        assert drv.states[0].status == "pending"
        assert 0 in drv.queue

    def test_stale_manifest_done_requeues_when_cell_missing(self, tmp_path):
        # All cells done → first Driver writes a manifest marking chunk 0 "done".
        cfgs = [rr.Config(id="a"), rr.Config(id="b")]
        for cid in ("a", "b"):
            for q in (0, 1):
                d = rr.qdir(Path(tmp_path), cid, q)
                _write_report(d, "1.00", 1)
                (d / ".done").write_text("1")
        drv0 = rr.Driver(_args(tmp_path, chunk_size=2), cfgs, n_questions=3)
        drv0.save_manifest()
        assert drv0.states[0].status == "done"
        # Now a cell's marker disappears; a fresh Driver must re-queue the chunk
        # even though the manifest still says "done" (disk is authoritative).
        (rr.qdir(Path(tmp_path), "b", 1) / ".done").unlink()
        drv = rr.Driver(_args(tmp_path, chunk_size=2), cfgs, n_questions=3)
        assert drv.states[0].status == "pending"
        assert 0 in drv.queue


# --- Driver retry / fail / success (chunk-keyed) --------------------------


class TestDriverFinishChunk:
    def _running(self, drv, chunk_idx, out_log="opaque crash"):
        log = Path(drv.out_base) / f"_chunk{chunk_idx:04d}.log"
        log.write_text(out_log)
        drv.running[chunk_idx] = (_P(), rr.time.time(), log)
        drv.states[chunk_idx].status = "running"
        drv.states[chunk_idx].attempts = 1

    def test_retryable_then_fail(self, tmp_path):
        drv = rr.Driver(_args(tmp_path, max_retries=1, chunk_size=2), _one_cfg(), n_questions=1)
        drv.queue.clear()
        self._running(drv, 0)
        drv._finish(0, returncode=2)  # no cells done → not success
        assert drv.states[0].status == "pending" and 0 in drv.queue

        drv.queue.clear()
        self._running(drv, 0)
        drv.states[0].attempts = 2
        drv._finish(0, returncode=2)
        assert drv.states[0].status == "failed" and 0 not in drv.queue

    def test_deterministic_failure_not_retried(self, tmp_path):
        drv = rr.Driver(_args(tmp_path, max_retries=5, chunk_size=2), _one_cfg(), n_questions=1)
        drv.queue.clear()
        self._running(drv, 0, out_log="Traceback (most recent call last):\nValueError: bad")
        drv._finish(0, returncode=1)
        assert drv.states[0].status == "failed" and 0 not in drv.queue

    def test_success_when_all_cells_done(self, tmp_path):
        drv = rr.Driver(_args(tmp_path, chunk_size=2), _one_cfg(), n_questions=1)
        d = rr.qdir(Path(tmp_path), "base", 0)
        _write_report(d, "5.00", 1)
        (d / ".done").write_text("1")
        self._running(drv, 0, out_log="ok")
        drv._finish(0, returncode=0)
        assert drv.states[0].status == "done"


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
    def test_build_run_argv_applies_config_slot_and_cadence(self, tmp_path):
        cfg = rr.Config(id="roi", slot_config="/tmp/variant.yaml", scan_min=1.0, scan_max=5.0)
        argv = rr.build_run_argv(_args(tmp_path), cfg, q_global=3, out_dir=tmp_path / "o")
        assert argv[0] == "run"
        assert "uv" not in argv and "hl-bt" not in argv
        assert argv[argv.index("--slot") + 1] == "v31"
        assert argv[argv.index("--slot-config") + 1] == "/tmp/variant.yaml"
        assert argv[argv.index("--skip-markets") + 1] == "3"
        assert argv[argv.index("--max-markets") + 1] == "1"
        assert argv[argv.index("--out-dir") + 1] == str(tmp_path / "o")
        assert argv[argv.index("--scan-min-interval-seconds") + 1] == "1.0"
        assert argv[argv.index("--scan-max-interval-seconds") + 1] == "5.0"

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


# --- Polymarket support ---------------------------------------------------


class TestPMSupport:
    def test_hl_argv_byte_identical_to_legacy(self, tmp_path):
        """HL argv must not gain any PM/fee flags — the byte-identical guarantee."""
        cfg = rr.Config(id="roi", slot_config="/tmp/variant.yaml", scan_min=1.0, scan_max=5.0)
        argv = rr.build_run_argv(_args(tmp_path), cfg, q_global=3, out_dir=tmp_path / "o")
        assert argv == [
            "run",
            "--data-source",
            "hl_hip4",
            "--kind",
            "binary",
            "--start",
            "2026-05-06",
            "--end",
            "2026-06-11",
            "--skip-markets",
            "3",
            "--max-markets",
            "1",
            "--workers",
            "1",
            "--out-dir",
            str(tmp_path / "o"),
            "--slot",
            "v31",
            "--slot-config",
            "/tmp/variant.yaml",
            "--scan-mode",
            "event",
            "--scan-min-interval-seconds",
            "1.0",
            "--scan-max-interval-seconds",
            "5.0",
        ]

    def test_pm_argv_emits_pm_and_fee_flags(self, tmp_path):
        a = _args(
            tmp_path,
            data_source="polymarket",
            slot="v31_pm",
            pm_flavor="btc_updown",
            pm_book_source="recorded",
            pm_reference_source="binance_bbo",
            pm_binance_bbo_product_type="spot",
            fee_model="pm_binary",
            fee_rate=0.07,
        )
        argv = rr.build_run_argv(a, rr.Config(id="base"), q_global=0, out_dir=tmp_path / "o")
        assert argv[argv.index("--data-source") + 1] == "polymarket"
        assert argv[argv.index("--slot") + 1] == "v31_pm"
        assert argv[argv.index("--pm-flavor") + 1] == "btc_updown"
        assert argv[argv.index("--pm-book-source") + 1] == "recorded"
        assert argv[argv.index("--pm-reference-source") + 1] == "binance_bbo"
        assert argv[argv.index("--pm-binance-bbo-product-type") + 1] == "spot"
        assert argv[argv.index("--fee-model") + 1] == "pm_binary"
        assert argv[argv.index("--fee-rate") + 1] == "0.07"

    def test_pm_argv_per_config_overrides(self, tmp_path):
        a = _args(
            tmp_path,
            data_source="polymarket",
            slot="v31_pm",
            pm_reference_source="klines",
            pm_book_source="synthetic",
            fee_model="flat",
            fee_rate=0.07,
        )
        cfg = rr.Config(
            id="variant",
            pm_reference_source="binance_bbo",
            pm_book_source="recorded",
            pm_binance_bbo_product_type="spot",
            fee_model="pm_binary",
            fee_rate=0.03,
        )
        argv = rr.build_run_argv(a, cfg, q_global=0, out_dir=tmp_path / "o")
        assert argv[argv.index("--pm-reference-source") + 1] == "binance_bbo"
        assert argv[argv.index("--pm-book-source") + 1] == "recorded"
        assert argv[argv.index("--pm-binance-bbo-product-type") + 1] == "spot"
        assert argv[argv.index("--fee-model") + 1] == "pm_binary"
        assert argv[argv.index("--fee-rate") + 1] == "0.03"

    def test_pm_argv_emits_cache_root_when_set(self, tmp_path):
        a = _args(tmp_path, data_source="polymarket", slot="v31_pm", cache_root="../../data/sim")
        argv = rr.build_run_argv(a, rr.Config(id="base"), q_global=0, out_dir=tmp_path / "o")
        assert argv[argv.index("--cache-root") + 1] == "../../data/sim"

    def test_hl_argv_omits_cache_root_even_when_set(self, tmp_path):
        # cache_root override is PM-only here; HL stays byte-identical (env-driven).
        a = _args(tmp_path, cache_root="../../data")
        argv = rr.build_run_argv(a, rr.Config(id="base"), q_global=0, out_dir=tmp_path / "o")
        assert "--cache-root" not in argv

    def test_cli_data_source_default_is_hl(self):
        ap = rr._build_arg_parser()
        ns = ap.parse_args(["--kind", "binary", "--start", "x", "--end", "y", "--out-base", "/o", "--slot", "v31"])
        assert ns.data_source == "hl_hip4"
        assert ns.pm_flavor == "btc_updown"
        assert ns.fee_model == "flat"

    def test_cli_pm_args_parse(self):
        ap = rr._build_arg_parser()
        ns = ap.parse_args(
            [
                "--data-source",
                "polymarket",
                "--kind",
                "binary",
                "--start",
                "x",
                "--end",
                "y",
                "--out-base",
                "/o",
                "--slot",
                "v31_pm",
                "--pm-book-source",
                "recorded",
                "--pm-reference-source",
                "binance_bbo",
                "--pm-binance-bbo-product-type",
                "spot",
                "--fee-model",
                "pm_binary",
                "--fee-rate",
                "0.07",
            ]
        )
        assert ns.data_source == "polymarket"
        assert ns.pm_book_source == "recorded"
        assert ns.pm_reference_source == "binance_bbo"
        assert ns.pm_binance_bbo_product_type == "spot"
        assert ns.fee_model == "pm_binary"
        assert ns.fee_rate == 0.07

    def test_config_roundtrip_pm_fields(self, tmp_path):
        cfgs = [
            rr.Config(
                id="a",
                pm_book_source="recorded",
                pm_reference_source="binance_bbo",
                pm_binance_bbo_product_type="spot",
                fee_model="pm_binary",
                fee_rate=0.03,
            )
        ]
        p = rr.write_configs_file(tmp_path, cfgs)
        back = rr.load_configs_file(p)
        assert back[0].pm_book_source == "recorded"
        assert back[0].pm_reference_source == "binance_bbo"
        assert back[0].pm_binance_bbo_product_type == "spot"
        assert back[0].fee_model == "pm_binary"
        assert back[0].fee_rate == 0.03

    def test_load_configs_sweep_with_pm_overrides(self, tmp_path):
        import json

        cfg_json = tmp_path / "configs.json"
        cfg_json.write_text(
            json.dumps(
                [
                    {"id": "base"},
                    {"id": "recorded", "pm_book_source": "recorded", "fee_model": "pm_binary"},
                ]
            )
        )
        cfgs = rr.load_configs(_args(tmp_path, configs=str(cfg_json)))
        assert [c.id for c in cfgs] == ["base", "recorded"]
        assert cfgs[0].pm_book_source is None  # falls back to args
        assert cfgs[1].pm_book_source == "recorded"
        assert cfgs[1].fee_model == "pm_binary"

    def test_discover_count_uses_polymarket_source(self, tmp_path, monkeypatch):
        captured = {}

        class _FakePM:
            def __init__(self, **kw):
                captured["ctor"] = kw

            def discover(self, *, start, end, kind="both", **_):
                captured["discover"] = dict(start=start, end=end, kind=kind)
                return ["q0", "q1", "q2"]

        import hlanalysis.backtest.data.polymarket as pm_mod

        monkeypatch.setattr(pm_mod, "PolymarketDataSource", _FakePM)
        a = _args(tmp_path, data_source="polymarket", kind="binary", cache_root="/tmp/sim")
        n = rr.discover_count(a)
        assert n == 3
        assert captured["discover"]["kind"] == "binary"
        assert str(captured["ctor"]["cache_root"]) == "/tmp/sim"

    def test_discover_count_uses_hl_source_by_default(self, tmp_path, monkeypatch):
        captured = {}

        class _FakeHL:
            def __init__(self, data_root):
                captured["data_root"] = data_root

            def discover(self, *, start, end, kinds):
                captured["kinds"] = kinds
                return ["a", "b"]

        import hlanalysis.backtest.data.hl_hip4 as hl_mod

        monkeypatch.setattr(hl_mod, "HLHip4DataSource", _FakeHL)
        n = rr.discover_count(_args(tmp_path, kind="binary"))
        assert n == 2
        assert captured["kinds"] == ("priceBinary",)


# --- config persistence ---------------------------------------------------


class TestConfigPersistence:
    def test_roundtrip_single(self, tmp_path):
        cfgs = [rr.Config(id="base", slot_config="/x.yaml")]
        p = rr.write_configs_file(tmp_path, cfgs)
        assert p == tmp_path / "_configs.json"
        back = rr.load_configs_file(p)
        assert [c.id for c in back] == ["base"]
        assert back[0].slot_config == "/x.yaml"

    def test_roundtrip_sweep_with_env_and_scan(self, tmp_path):
        cfgs = [
            rr.Config(id="a", slot_config="/a.yaml", scan_min=1.0, scan_max=5.0),
            rr.Config(id="b", env={"K": "v"}),
        ]
        p = rr.write_configs_file(tmp_path, cfgs)
        back = rr.load_configs_file(p)
        assert [c.id for c in back] == ["a", "b"]
        assert back[0].scan_min == 1.0 and back[0].scan_max == 5.0
        assert back[1].env == {"K": "v"}


# --- worker mode ----------------------------------------------------------


class TestWorkerChunk:
    def _fake_invoke(self, calls):
        def _inv(argv):
            calls.append(argv)
            out = Path(argv[argv.index("--out-dir") + 1])
            _write_report(out, "1.00", 1)
            return 0

        return _inv

    def test_runs_all_cells_and_marks_done(self, tmp_path, monkeypatch):
        cfgs = [rr.Config(id="a"), rr.Config(id="b")]
        calls = []
        monkeypatch.setattr(rr, "_invoke_run", self._fake_invoke(calls))
        rc = rr.run_worker_chunk(_args(tmp_path, chunk_size=2), cfgs, chunk_idx=0, n_questions=3)
        assert rc == 0
        assert len(calls) == 4
        for cid in ("a", "b"):
            for q in (0, 1):
                assert (rr.qdir(Path(tmp_path), cid, q) / ".done").exists()
        assert not (rr.qdir(Path(tmp_path), "a", 2)).exists()

    def test_sets_inproc_memo_env(self, tmp_path, monkeypatch):
        seen = {}

        def _inv(argv):
            seen["memo"] = rr.os.environ.get("HLBT_INPROC_BUNDLE_MEMO")
            _write_report(Path(argv[argv.index("--out-dir") + 1]), "1.00", 1)
            return 0

        monkeypatch.setattr(rr, "_invoke_run", _inv)
        rr.run_worker_chunk(_args(tmp_path, chunk_size=2), [rr.Config(id="a")], chunk_idx=0, n_questions=1)
        assert seen["memo"] == "1"

    def test_skips_already_done_cells(self, tmp_path, monkeypatch):
        cfgs = [rr.Config(id="a")]
        d = rr.qdir(Path(tmp_path), "a", 0)
        _write_report(d, "9.00", 1)
        (d / ".done").write_text("1")
        calls = []
        monkeypatch.setattr(rr, "_invoke_run", self._fake_invoke(calls))
        rr.run_worker_chunk(_args(tmp_path, chunk_size=2), cfgs, chunk_idx=0, n_questions=1)
        assert calls == []

    def test_applies_per_config_env_during_invoke(self, tmp_path, monkeypatch):
        seen = {}

        def _inv(argv):
            seen["K"] = rr.os.environ.get("K")
            _write_report(Path(argv[argv.index("--out-dir") + 1]), "1.00", 1)
            return 0

        monkeypatch.setattr(rr, "_invoke_run", _inv)
        rr.run_worker_chunk(
            _args(tmp_path, chunk_size=1), [rr.Config(id="a", env={"K": "v"})], chunk_idx=0, n_questions=1
        )
        assert seen["K"] == "v"
        assert rr.os.environ.get("K") is None

    def test_nonzero_rc_when_a_cell_fails(self, tmp_path, monkeypatch):
        def _inv(argv):
            return 1

        monkeypatch.setattr(rr, "_invoke_run", _inv)
        rc = rr.run_worker_chunk(_args(tmp_path, chunk_size=1), [rr.Config(id="a")], chunk_idx=0, n_questions=1)
        assert rc != 0


# --- aggregate + CLI ------------------------------------------------------


class TestAggregateFromDirs:
    def test_sums_per_config_from_report_dirs(self, tmp_path, capsys):
        for cid, q, pnl in [("a", 0, "10.00"), ("a", 1, "5.00"), ("b", 0, "1.00")]:
            d = rr.qdir(Path(tmp_path), cid, q)
            _write_report(d, pnl, 1)
            (d / ".done").write_text("1")
        rr.write_configs_file(Path(tmp_path), [rr.Config(id="a"), rr.Config(id="b")])
        rr.aggregate(Path(tmp_path))
        out = capsys.readouterr().out
        assert "a" in out and "15.00" in out  # 10 + 5
        assert "b" in out and "1.00" in out


class TestChunkSizeArg:
    def test_chunk_size_defaults_to_25(self):
        ap = rr._build_arg_parser()
        ns = ap.parse_args(["--kind", "binary", "--start", "x", "--end", "y", "--out-base", "/o", "--slot", "v31"])
        assert ns.chunk_size == 25

    def test_worker_chunk_args_parse(self):
        ap = rr._build_arg_parser()
        ns = ap.parse_args(
            [
                "--kind",
                "binary",
                "--start",
                "x",
                "--end",
                "y",
                "--out-base",
                "/o",
                "--_worker-chunk",
                "3",
                "--n-questions",
                "40",
                "--configs",
                "/c.json",
            ]
        )
        assert ns.worker_chunk == 3 and ns.n_questions == 40
