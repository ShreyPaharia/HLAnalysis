# Warm-chunk tuning driver — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `scripts/perf/resumable_run.py` amortize backtest cost across a tuning sweep by running each **chunk of K questions against the whole config list in one warm subprocess** (in-process bundle-memo reuse) instead of one cold `uv run hl-bt run` per `(config, question)`.

**Architecture:** `resumable_run.py` gains a hidden **worker mode** (`--_worker-chunk <idx>`) that the supervisor shells into once per chunk (own process group for OOM/crash isolation). Worker mode sets `HLBT_INPROC_BUNDLE_MEMO=1` and loops question-outer / config-inner, calling the existing `hlanalysis.backtest.cli.main(["run", ...])` per cell — bit-identical to today's cold path, but the module-global bundle memo decodes each question once and serves it to the other configs. The supervisor's work-queue is re-keyed from `(config_id, question_idx)` to **chunk index**; per-`(config, question)` report dirs + `.done` markers stay exactly as today.

**Tech Stack:** Python 3, argparse, `subprocess` (process-group isolation), `hlanalysis.backtest.cli.main` (in-process run entry), `HLBT_INPROC_BUNDLE_MEMO*` env (existing `_event_array_cache.py` machinery). Tests: pytest, module loaded via `importlib` (mirror existing `tests/perf/test_resumable_run.py`).

**Reference spec:** `docs/superpowers/specs/2026-06-16-warm-chunk-tuning-driver-design.md`

---

## File Structure

- **Modify** `scripts/perf/resumable_run.py` — the whole change lives here. Keep three clearly separated sections: (1) shared helpers (chunk math, argv builder, config persistence), (2) **supervisor** (`Driver`, queue, launch/finish, aggregate), (3) **worker mode** (`run_worker_chunk`, `_invoke_run`). Add a top-of-file comment marking the three sections.
- **Modify** `tests/perf/test_resumable_run.py` — extend with chunk-math, argv, config-persistence, worker-loop, and chunk-keyed-supervisor tests. The existing `(config, idx)` queue tests are replaced by chunk-keyed equivalents.
- **No** new files; **no** changes to `hlanalysis/` (worker mode only *calls* the existing CLI). This keeps the four-CLI surface and the `--slot` parity guarantee untouched.

### Key signatures defined by this plan (use these exact names everywhere)

```python
# section 1 — shared helpers
def num_chunks(n_questions: int, chunk_size: int) -> int
def chunk_bounds(chunk_idx: int, n_questions: int, chunk_size: int) -> tuple[int, int]  # (start, length)
def build_run_argv(args, cfg: Config, q_global: int, out_dir: Path) -> list[str]        # ["run", ...]
def write_configs_file(out_base: Path, configs: list[Config]) -> Path                   # out_base/_configs.json
def load_configs_file(path: Path) -> list[Config]

# section 3 — worker mode
def _invoke_run(argv: list[str]) -> int                                                 # wraps cli.main
def run_worker_chunk(args, configs: list[Config], chunk_idx: int, n_questions: int) -> int
```

`ChunkState` (replaces the per-cell `JobState`):

```python
@dataclass
class ChunkState:
    chunk_idx: int
    status: str = "pending"   # pending | running | done | failed
    attempts: int = 0
    wall_s: float = 0.0
    last_class: str = ""
    last_error: str = ""
    n_cells: int = 0          # configs × questions-in-chunk
    n_done: int = 0           # cells with a .done marker
```

The supervisor `states` dict is keyed by `int` (chunk index); `qdir(config_id, q_global)` is **unchanged** (`out_base/<config_id>/q{q_global:04d}`).

---

## Task 1: Chunk math helpers

**Files:**
- Modify: `scripts/perf/resumable_run.py` (add `num_chunks`, `chunk_bounds` near the top, after the `classify` block)
- Test: `tests/perf/test_resumable_run.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/perf/test_resumable_run.py`:

```python
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
        # 51 questions, K=25: chunks cover [0,25), [25,50), [50,51)
        assert rr.chunk_bounds(0, 51, 25) == (0, 25)
        assert rr.chunk_bounds(1, 51, 25) == (25, 25)
        assert rr.chunk_bounds(2, 51, 25) == (50, 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/perf/test_resumable_run.py::TestChunkMath -q`
Expected: FAIL with `AttributeError: module 'resumable_run' has no attribute 'num_chunks'`

- [ ] **Step 3: Write minimal implementation**

In `scripts/perf/resumable_run.py`, after the `classify` function add:

```python
# --- chunk math -----------------------------------------------------------


def num_chunks(n_questions: int, chunk_size: int) -> int:
    """ceil(n / K). K==1 → one chunk per question. n==0 → 0."""
    if n_questions <= 0:
        return 0
    return (n_questions + chunk_size - 1) // chunk_size


def chunk_bounds(chunk_idx: int, n_questions: int, chunk_size: int) -> tuple[int, int]:
    """(chunk_start, chunk_len) for chunk c: questions [c*K, min((c+1)*K, n))."""
    start = chunk_idx * chunk_size
    length = min(chunk_size, n_questions - start)
    return start, length
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/perf/test_resumable_run.py::TestChunkMath -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add scripts/perf/resumable_run.py tests/perf/test_resumable_run.py
git commit -m "perf(backtest): chunk math helpers for warm-chunk driver"
```

---

## Task 2: Per-cell run argv builder

Replace the subprocess command builder. Today `build_cmd` returns `["uv","run","hl-bt","run", ...]` for a cold per-cell subprocess. Worker mode runs the cell **in-process** via `cli.main`, so we need just the `["run", ...]` argv. The supervisor no longer shells `hl-bt` directly, so `build_cmd` is removed and replaced by `build_run_argv`.

**Files:**
- Modify: `scripts/perf/resumable_run.py` (replace `build_cmd` with `build_run_argv`)
- Test: `tests/perf/test_resumable_run.py` (replace `test_build_cmd_applies_config_slot_and_cadence`)

- [ ] **Step 1: Write the failing test**

Replace `TestSweep2D.test_build_cmd_applies_config_slot_and_cadence` with:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/perf/test_resumable_run.py::TestSweep2D::test_build_run_argv_applies_config_slot_and_cadence -q`
Expected: FAIL with `AttributeError: ... has no attribute 'build_run_argv'`

- [ ] **Step 3: Write minimal implementation**

In `scripts/perf/resumable_run.py`, delete `build_cmd` and add:

```python
def build_run_argv(args, cfg: Config, q_global: int, out_dir: Path) -> list[str]:
    """The ``hl-bt`` argv (sans the ``uv run hl-bt`` prefix) for ONE cell —
    a single question (``--skip-markets q_global --max-markets 1``) under one
    config. Worker mode passes this straight to ``cli.main``."""
    argv = [
        "run",
        "--data-source", "hl_hip4",
        "--kind", args.kind,
        "--start", args.start,
        "--end", args.end,
        "--skip-markets", str(q_global),
        "--max-markets", "1",
        "--workers", "1",
        "--out-dir", str(out_dir),
    ]
    slot_config = cfg.slot_config or args.slot_config
    if args.slot:
        argv += ["--slot", args.slot]
        if slot_config:
            argv += ["--slot-config", slot_config]
        if args.slot_class:
            argv += ["--slot-class", args.slot_class]
    elif args.strategy:
        argv += ["--strategy", args.strategy]
        if slot_config:
            argv += ["--config", slot_config]
    scan_min = cfg.scan_min if cfg.scan_min is not None else args.scan_min
    scan_max = cfg.scan_max if cfg.scan_max is not None else args.scan_max
    if scan_min is not None:
        argv += [
            "--scan-mode", "event",
            "--scan-min-interval-seconds", str(scan_min),
            "--scan-max-interval-seconds", str(scan_max if scan_max is not None else 2.0),
        ]
    return argv
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/perf/test_resumable_run.py::TestSweep2D::test_build_run_argv_applies_config_slot_and_cadence -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/perf/resumable_run.py tests/perf/test_resumable_run.py
git commit -m "perf(backtest): build_run_argv (in-process run argv) replaces build_cmd"
```

---

## Task 3: Config persistence (worker reads the same config list)

The supervisor resolves the config list once and writes it to `out_base/_configs.json`; every worker subprocess loads it (so single-config and sweep cases share one path).

**Files:**
- Modify: `scripts/perf/resumable_run.py` (add `write_configs_file`, `load_configs_file`)
- Test: `tests/perf/test_resumable_run.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/perf/test_resumable_run.py::TestConfigPersistence -q`
Expected: FAIL with `AttributeError: ... has no attribute 'write_configs_file'`

- [ ] **Step 3: Write minimal implementation**

Add to `scripts/perf/resumable_run.py` (near `load_configs`):

```python
def write_configs_file(out_base: Path, configs: list[Config]) -> Path:
    """Persist the resolved config list so every worker subprocess reads the
    SAME cells (single-config and sweep share one path)."""
    path = out_base / "_configs.json"
    path.write_text(json.dumps([asdict(c) for c in configs], indent=2))
    return path


def load_configs_file(path: Path) -> list[Config]:
    return [Config(**c) for c in json.loads(Path(path).read_text())]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/perf/test_resumable_run.py::TestConfigPersistence -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/perf/resumable_run.py tests/perf/test_resumable_run.py
git commit -m "perf(backtest): persist resolved config list for worker subprocesses"
```

---

## Task 4: Worker mode — warm in-process chunk loop

The heart of the change. One worker process runs all cells of one chunk in-process, reusing the bundle memo. `_invoke_run` is a thin, monkeypatchable wrapper around `cli.main` so the loop is unit-testable without real backtests.

**Files:**
- Modify: `scripts/perf/resumable_run.py` (add `_invoke_run`, `run_worker_chunk`; add `qdir`/`done_marker` module-level helpers if not already module-level — `qdir` currently lives on `Driver`; extract a module-level `qdir(out_base, config_id, q_global)` and have `Driver.qdir` delegate)
- Test: `tests/perf/test_resumable_run.py`

- [ ] **Step 1: Write the failing test**

```python
class TestWorkerChunk:
    def _fake_invoke(self, calls):
        # Records argv, writes a report.md into --out-dir, returns 0 (success).
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
        # 3 questions, K=2 → chunk 0 covers questions 0,1
        rc = rr.run_worker_chunk(_args(tmp_path, chunk_size=2), cfgs, chunk_idx=0, n_questions=3)
        assert rc == 0
        # cells = 2 configs × 2 questions = 4 invocations + 4 .done
        assert len(calls) == 4
        for cid in ("a", "b"):
            for q in (0, 1):
                assert (rr.qdir(Path(tmp_path), cid, q) / ".done").exists()
        # chunk 0 must NOT touch question 2
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
        assert calls == []  # nothing re-run

    def test_applies_per_config_env_during_invoke(self, tmp_path, monkeypatch):
        seen = {}
        def _inv(argv):
            seen["K"] = rr.os.environ.get("K")
            _write_report(Path(argv[argv.index("--out-dir") + 1]), "1.00", 1)
            return 0
        monkeypatch.setattr(rr, "_invoke_run", _inv)
        rr.run_worker_chunk(_args(tmp_path, chunk_size=1), [rr.Config(id="a", env={"K": "v"})], chunk_idx=0, n_questions=1)
        assert seen["K"] == "v"
        assert rr.os.environ.get("K") is None  # restored after the call

    def test_nonzero_rc_when_a_cell_fails(self, tmp_path, monkeypatch):
        def _inv(argv):
            return 1  # no report written → cell failed
        monkeypatch.setattr(rr, "_invoke_run", _inv)
        rc = rr.run_worker_chunk(_args(tmp_path, chunk_size=1), [rr.Config(id="a")], chunk_idx=0, n_questions=1)
        assert rc != 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/perf/test_resumable_run.py::TestWorkerChunk -q`
Expected: FAIL with `AttributeError: ... has no attribute 'run_worker_chunk'` (and `qdir`)

- [ ] **Step 3: Write minimal implementation**

First extract a module-level `qdir` (place near `report_path`/`done_marker`):

```python
def qdir(out_base: Path, config_id: str, q_global: int) -> Path:
    return out_base / config_id / f"q{q_global:04d}"
```

Then change `Driver.qdir` to delegate: `return qdir(self.out_base, config_id, q_global)`.

Add the worker section at the bottom of the file (clearly marked `# === WORKER MODE ===`):

```python
# === WORKER MODE ==========================================================
import contextlib  # noqa: E402  (kept with the worker section for locality)


def _invoke_run(argv: list[str]) -> int:
    """Run one ``hl-bt run`` cell in-process. Thin wrapper so tests can patch it.

    The module-global in-process bundle memo (set via HLBT_INPROC_BUNDLE_MEMO)
    persists across calls within this process — that is the whole point: the
    first config to touch question q decodes its bundle, configs 2..M reuse it.
    """
    from hlanalysis.backtest.cli import main as bt_main

    return bt_main(argv)


@contextlib.contextmanager
def _env_overlay(overrides: dict[str, str]):
    old = {k: os.environ.get(k) for k in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def run_worker_chunk(args, configs: list[Config], chunk_idx: int, n_questions: int) -> int:
    """Run every (config, question) cell of ONE chunk in-process, reusing the
    bundle memo. question-outer / config-inner keeps ~one bundle resident.

    Returns 0 iff every cell ended with a .done marker; non-zero otherwise so the
    supervisor classifies + retries the chunk.
    """
    os.environ["HLBT_INPROC_BUNDLE_MEMO"] = "1"
    out_base = Path(args.out_base)
    chunk_size = max(1, int(getattr(args, "chunk_size", 1) or 1))
    start, length = chunk_bounds(chunk_idx, n_questions, chunk_size)
    failures = 0
    for q_global in range(start, start + length):
        for cfg in configs:
            out_dir = qdir(out_base, cfg.id, q_global)
            if done_marker(out_dir).exists():
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            argv = build_run_argv(args, cfg, q_global, out_dir)
            try:
                with _env_overlay(cfg.env):
                    rc = _invoke_run(argv)
            except Exception as exc:  # noqa: BLE001  one bad cell must not kill the chunk
                rc = 1
                (out_dir / "run.log").write_text(f"worker exception: {exc!r}")
            if rc == 0 and report_path(out_dir).exists():
                done_marker(out_dir).write_text(str(int(time.time())))
            else:
                failures += 1
                print(f"[cell-fail] {cfg.id}/q{q_global:04d} rc={rc}", flush=True)
    return 1 if failures else 0
```

Note: `report_path` and `done_marker` already exist module-level and take a single `out_dir` — reuse them unchanged.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/perf/test_resumable_run.py::TestWorkerChunk -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add scripts/perf/resumable_run.py tests/perf/test_resumable_run.py
git commit -m "perf(backtest): warm-chunk worker mode (in-process bundle-memo reuse)"
```

---

## Task 5: Supervisor — chunk-keyed queue, launch, finish

Re-key the supervisor from `(config_id, idx)` to chunk index; launch worker subprocesses (self-invocation); detect chunk completion from per-cell `.done` markers.

**Files:**
- Modify: `scripts/perf/resumable_run.py` (`Driver.__init__`, `_launch`, `_finish`, add `ChunkState`, `cells_for_chunk`, `chunk_done`; delete `JobState`)
- Test: `tests/perf/test_resumable_run.py` (rewrite `TestDriverResume`, `TestDriverFinish`, `TestSweep2D` queue tests for chunk keys)

- [ ] **Step 1: Write the failing tests**

Replace the `JobState`-based resume/finish/sweep tests with chunk-keyed versions:

```python
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


class TestDriverFinishChunk:
    def _running(self, drv, chunk_idx, out_log="opaque crash"):
        out = Path(drv.out_base)
        (out / "_chunk0.log").write_text(out_log)
        drv.running[chunk_idx] = (_P(), rr.time.time(), out / "_chunk0.log")
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

    def test_success_when_all_cells_done(self, tmp_path):
        drv = rr.Driver(_args(tmp_path, chunk_size=2), _one_cfg(), n_questions=1)
        d = rr.qdir(Path(tmp_path), "base", 0)
        _write_report(d, "5.00", 1)
        (d / ".done").write_text("1")
        out = Path(drv.out_base) / "_chunk0.log"
        out.write_text("ok")
        drv.running[0] = (_P(), rr.time.time(), out)
        drv.states[0].status = "running"
        drv.states[0].attempts = 1
        drv._finish(0, returncode=0)
        assert drv.states[0].status == "done"
```

Also update `_args` to include `chunk_size` (default 25):

```python
def _args(out_base: Path, **kw) -> SimpleNamespace:
    base = dict(
        kind="binary", start="2026-05-06", end="2026-06-11", out_base=str(out_base),
        configs=None, slot="v31", slot_config=None, slot_class=None, strategy=None,
        workers=2, max_retries=2, timeout=3600.0, scan_min=None, scan_max=None,
        chunk_size=25,
    )
    base.update(kw)
    return SimpleNamespace(**base)
```

Delete the now-obsolete `TestDriverResume`, `TestDriverFinish`, and `TestSweep2D.test_jobs_are_config_x_question` / `test_per_config_resume_independent` (their behaviour is superseded by the chunk-keyed tests above). Keep `test_load_configs_single_vs_sweep`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/perf/test_resumable_run.py::TestDriverChunkQueue tests/perf/test_resumable_run.py::TestDriverFinishChunk -q`
Expected: FAIL (Driver still keyed by `(config_id, idx)`; `chunk_size` unknown)

- [ ] **Step 3: Write the implementation**

In `scripts/perf/resumable_run.py`:

(a) Replace `JobState` with `ChunkState` (fields per the File Structure section).

(b) Rewrite `Driver.__init__`:

```python
def __init__(self, args, configs: list[Config], n_questions: int):
    self.args = args
    self.configs = configs
    self.out_base = Path(args.out_base)
    self.out_base.mkdir(parents=True, exist_ok=True)
    self.manifest_path = self.out_base / "manifest.json"
    self.n_questions = n_questions
    self.chunk_size = max(1, int(getattr(args, "chunk_size", 1) or 1))
    self.configs_path = write_configs_file(self.out_base, configs)
    n_chunks = num_chunks(n_questions, self.chunk_size)
    self.states: dict[int, ChunkState] = {c: ChunkState(chunk_idx=c) for c in range(n_chunks)}
    self._load_manifest()
    for c, st in self.states.items():
        st.n_cells = len(self.cells_for_chunk(c))
        st.n_done = sum(1 for (cid, q) in self.cells_for_chunk(c) if done_marker(qdir(self.out_base, cid, q)).exists())
        if st.n_cells > 0 and st.n_done == st.n_cells:
            st.status = "done"
    self.queue: list[int] = [c for c, s in self.states.items() if s.status not in ("done", "failed")]
    self.running: dict[int, tuple[subprocess.Popen, float, Path]] = {}
    self._stop = False

def cells_for_chunk(self, chunk_idx: int) -> list[tuple[str, int]]:
    start, length = chunk_bounds(chunk_idx, self.n_questions, self.chunk_size)
    return [(c.id, q) for q in range(start, start + length) for c in self.configs]

def qdir(self, config_id: str, q_global: int) -> Path:
    return qdir(self.out_base, config_id, q_global)

def chunk_done(self, chunk_idx: int) -> bool:
    cells = self.cells_for_chunk(chunk_idx)
    return bool(cells) and all(done_marker(qdir(self.out_base, cid, q)).exists() for cid, q in cells)
```

Update `_load_manifest`/`save_manifest` to use chunk-keyed `ChunkState` (key on `chunk_idx`; `jobs` → `chunks`).

(c) Rewrite `_launch` to spawn the worker self-invocation:

```python
def _launch(self, chunk_idx: int) -> None:
    log_path = self.out_base / f"_chunk{chunk_idx:04d}.log"
    logf = open(log_path, "w")  # noqa: SIM115
    env = dict(os.environ)
    env.setdefault("LOGURU_LEVEL", "ERROR")
    env["HLBT_INPROC_BUNDLE_MEMO"] = "1"
    env["HLBT_INPROC_BUNDLE_MEMO_WORKERS"] = str(max(1, int(self.args.workers)))
    cmd = [
        sys.executable, os.path.abspath(__file__),
        "--_worker-chunk", str(chunk_idx),
        "--configs", str(self.configs_path),
        "--out-base", str(self.out_base),
        "--kind", self.args.kind,
        "--start", self.args.start,
        "--end", self.args.end,
        "--chunk-size", str(self.chunk_size),
        "--n-questions", str(self.n_questions),
    ]
    if self.args.slot:
        cmd += ["--slot", self.args.slot]
    if self.args.slot_config:
        cmd += ["--slot-config", self.args.slot_config]
    if self.args.slot_class:
        cmd += ["--slot-class", self.args.slot_class]
    if self.args.strategy:
        cmd += ["--strategy", self.args.strategy]
    if self.args.scan_min is not None:
        cmd += ["--scan-min", str(self.args.scan_min)]
    if self.args.scan_max is not None:
        cmd += ["--scan-max", str(self.args.scan_max)]
    p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True, cwd=os.getcwd(), env=env)
    self.running[chunk_idx] = (p, time.time(), log_path)
    st = self.states[chunk_idx]
    st.status = "running"
    st.attempts += 1
    print(f"[launch] chunk{chunk_idx:04d} (attempt {st.attempts}) pid={p.pid}", flush=True)
```

(d) Rewrite `_finish` to classify on chunk completion (success = all cells `.done`):

```python
def _finish(self, chunk_idx: int, returncode: int) -> None:
    _p, t0, log_path = self.running.pop(chunk_idx)
    st = self.states[chunk_idx]
    st.wall_s = round(time.time() - t0, 1)
    log_text = ""
    try:
        log_text = log_path.read_text(errors="replace")[-4000:]
    except OSError:
        pass
    st.n_done = sum(1 for (cid, q) in self.cells_for_chunk(chunk_idx) if done_marker(qdir(self.out_base, cid, q)).exists())
    all_done = self.chunk_done(chunk_idx)
    cls = classify(returncode, log_text, report_exists=all_done)
    st.last_class = cls
    if cls == SUCCESS:
        st.status = "done"
        print(f"[done]   chunk{chunk_idx:04d} rc={returncode} wall={st.wall_s}s cells={st.n_done}/{st.n_cells}", flush=True)
    else:
        last_line = log_text.strip().splitlines()[-1] if log_text.strip() else ""
        st.last_error = f"rc={returncode} class={cls}; {last_line}"
        if cls == RETRYABLE and st.attempts <= self.args.max_retries:
            st.status = "pending"
            self.queue.append(chunk_idx)
            print(f"[retry]  chunk{chunk_idx:04d} rc={returncode} class={cls} attempt={st.attempts}/{self.args.max_retries + 1}", flush=True)
        else:
            st.status = "failed"
            print(f"[FAILED] chunk{chunk_idx:04d} rc={returncode} class={cls} :: {st.last_error}", flush=True)
    self.save_manifest()
```

The `run()` loop body is unchanged except it iterates chunk-index keys (already generic — it uses `self.queue` / `self.running` keys). Update the final summary print to say `chunks` and drop the `q{idx:04d}` formatting (use `chunk{idx:04d}`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/perf/test_resumable_run.py -q`
Expected: PASS (all classes green; the deleted tests are gone)

- [ ] **Step 5: Commit**

```bash
git add scripts/perf/resumable_run.py tests/perf/test_resumable_run.py
git commit -m "perf(backtest): chunk-keyed supervisor (launch worker per chunk, .done completion)"
```

---

## Task 6: aggregate() reads report dirs + CLI wiring (chunk-size, worker dispatch)

Repoint `aggregate` to read per-config report dirs (jobs are now chunks). Add `--chunk-size` (default 25), the hidden `--_worker-chunk` / `--n-questions` args, and worker dispatch in `main`.

**Files:**
- Modify: `scripts/perf/resumable_run.py` (`aggregate`, `main`, arg parser)
- Test: `tests/perf/test_resumable_run.py`

- [ ] **Step 1: Write the failing test**

```python
class TestAggregateFromDirs:
    def test_sums_per_config_from_report_dirs(self, tmp_path, capsys):
        for cid, q, pnl in [("a", 0, "10.00"), ("a", 1, "5.00"), ("b", 0, "1.00")]:
            d = rr.qdir(Path(tmp_path), cid, q)
            _write_report(d, pnl, 1)
            (d / ".done").write_text("1")
        rr.write_configs_file(Path(tmp_path), [rr.Config(id="a"), rr.Config(id="b")])
        rr.aggregate(Path(tmp_path))
        out = capsys.readouterr().out
        assert "a" in out and "15.00" in out   # 10 + 5
        assert "b" in out and "1.00" in out


class TestChunkSizeArg:
    def test_chunk_size_defaults_to_25(self):
        import argparse
        ap = rr._build_arg_parser()
        ns = ap.parse_args(["--kind", "binary", "--start", "x", "--end", "y", "--out-base", "/o", "--slot", "v31"])
        assert ns.chunk_size == 25
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/perf/test_resumable_run.py::TestAggregateFromDirs tests/perf/test_resumable_run.py::TestChunkSizeArg -q`
Expected: FAIL (`aggregate` still reads manifest; `_build_arg_parser` missing)

- [ ] **Step 3: Write the implementation**

(a) Rewrite `aggregate` to read report dirs (use the persisted `_configs.json` for the config-id list):

```python
def aggregate(out_base: Path) -> None:
    """Per-config totals from completed cells, read straight from report dirs."""
    cfg_file = out_base / "_configs.json"
    if cfg_file.exists():
        config_ids = [c.id for c in load_configs_file(cfg_file)]
    else:  # fall back to top-level dirs that contain q* cells
        config_ids = sorted(p.name for p in out_base.iterdir() if p.is_dir() and not p.name.startswith("_"))
    print("=== AGGREGATE (completed cells per config) ===")
    for cid in config_ids:
        cells = sorted((out_base / cid).glob("q*/report.md")) if (out_base / cid).is_dir() else []
        rows = [parse_pnl((out_base / cid / rp.parent.name)) for rp in cells]
        rows = [(p, t) for (p, t) in rows if p is not None]
        tot = sum(p for p, _ in rows)
        tr = sum(t or 0 for _, t in rows)
        print(f"  {cid:16s}: n={len(rows):>3} totalPnL=${tot:>9.2f} trades={tr:>5}")
```

(b) Extract the parser into `_build_arg_parser()` and add the new args. Keep `main()` thin:

```python
def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=["binary", "bucket"], required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out-base", required=True)
    ap.add_argument("--configs", default=None, help="JSON list of config cells for a sweep; omit for single config")
    ap.add_argument("--slot", default=None)
    ap.add_argument("--slot-config", default=None)
    ap.add_argument("--slot-class", default=None)
    ap.add_argument("--strategy", default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--chunk-size", type=int, default=25, help="questions per warm subprocess (amortizes startup + shares the bundle memo across configs)")
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--scan-min", type=float, default=None)
    ap.add_argument("--scan-max", type=float, default=None)
    ap.add_argument("--aggregate-only", action="store_true")
    # hidden worker-mode args (the supervisor self-invokes with these)
    ap.add_argument("--_worker-chunk", type=int, default=None, dest="worker_chunk", help=argparse.SUPPRESS)
    ap.add_argument("--n-questions", type=int, default=None, dest="n_questions", help=argparse.SUPPRESS)
    return ap


def main() -> int:
    args = _build_arg_parser().parse_args()

    # worker mode: run one chunk in-process and exit.
    if args.worker_chunk is not None:
        configs = load_configs_file(Path(args.configs))
        return run_worker_chunk(args, configs, args.worker_chunk, args.n_questions)

    out_base = Path(args.out_base)
    if args.aggregate_only:
        aggregate(out_base)
        return 0
    configs = load_configs(args)
    n = discover_count(args)
    print(f"discovered {n} {args.kind} questions in [{args.start},{args.end}); {len(configs)} config(s); chunk_size={args.chunk_size} → {num_chunks(n, args.chunk_size)} chunks", flush=True)
    if n == 0:
        print("no questions — nothing to do", file=sys.stderr)
        return 2
    rc = Driver(args, configs, n).run()
    aggregate(out_base)
    return rc
```

Note: `run_worker_chunk` reads `args.out_base`, `args.kind`, `args.start`, `args.end`, `args.slot*`, `args.strategy`, `args.scan_min/max`, `args.chunk_size` — all present on the worker arg namespace.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/perf/test_resumable_run.py -q`
Expected: PASS (entire file green)

- [ ] **Step 5: Commit**

```bash
git add scripts/perf/resumable_run.py tests/perf/test_resumable_run.py
git commit -m "perf(backtest): aggregate from report dirs + chunk-size 25 + worker dispatch"
```

---

## Task 7: Update docstring, lint/type gate, full suite

**Files:**
- Modify: `scripts/perf/resumable_run.py` (module docstring → describe warm-chunk model + `--chunk-size`)
- Modify: `docs/architecture/backtest.md` (one line under "Related" or "Gotchas" pointing at the warm-chunk driver) — optional but preferred

- [ ] **Step 1: Update the module docstring**

Rewrite the top-of-file docstring to describe: chunk = K questions × all configs in one warm subprocess; the in-process bundle memo reuse; `--chunk-size` (default 25); per-chunk resume; `--workers N` parallel chunks with `total/N` memo budget. Update the Usage block to drop the `uv run hl-bt run` per-cell framing and show:

```
    HLBT_HL_DATA_ROOT=../../data uv run python scripts/perf/resumable_run.py \
        --slot v31 --kind binary --start 2026-05-06 --end 2026-06-11 \
        --out-base /tmp/run_binary --workers 6 --chunk-size 25 --scan-min 0.5 --scan-max 2.0
```

- [ ] **Step 2: Run the lint/format/type gate**

Run:
```bash
uvx ruff check scripts/perf/resumable_run.py tests/perf/test_resumable_run.py
uvx ruff format --check scripts/perf/resumable_run.py tests/perf/test_resumable_run.py
```
Expected: clean (fix any findings; re-run). `mypy` runs informationally — address obvious type errors.

- [ ] **Step 3: Run the full perf test module + a broad smoke**

Run:
```bash
uv run pytest tests/perf/test_resumable_run.py -q
uv run pytest -q -k "resumable or chunk"
```
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add scripts/perf/resumable_run.py docs/architecture/backtest.md
git commit -m "docs(perf): document warm-chunk driver model + chunk-size knob"
```

---

## Task 8: Recorded-data parity + speedup validation (the correctness oracle)

This is the gate that proves the warm path changed nothing but speed. It runs against the **main checkout's recorded data** (`../../data`) — not synthetic — per CLAUDE.md. It is a **manual** validation (not a CI pytest), because it needs the ~18 GB corpus. Record the outcome in the PR description.

- [ ] **Step 1: Pick a small but real window** (≈6–8 HL binary questions so it runs in a couple minutes), e.g. `--start 2026-06-09 --end 2026-06-11`.

- [ ] **Step 2: Cold baseline (chunk-size 1, one cell per process)**

```bash
export HLBT_HL_DATA_ROOT=../../data
uv run python scripts/perf/resumable_run.py \
  --slot v31 --kind binary --start 2026-06-09 --end 2026-06-11 \
  --out-base /tmp/parity_cold --workers 1 --chunk-size 1 --scan-min 0.5 --scan-max 2.0
```

- [ ] **Step 3: Warm chunk run (chunk-size 25, all questions in one warm process)**

```bash
uv run python scripts/perf/resumable_run.py \
  --slot v31 --kind binary --start 2026-06-09 --end 2026-06-11 \
  --out-base /tmp/parity_warm --workers 1 --chunk-size 25 --scan-min 0.5 --scan-max 2.0
```

- [ ] **Step 4: Assert per-question report.md are byte-identical**

```bash
for q in /tmp/parity_cold/base/q*; do
  diff -q "$q/report.md" "/tmp/parity_warm/base/$(basename "$q")/report.md" || echo "MISMATCH $q"
done
```
Expected: no `MISMATCH` lines. (Reports are deterministic at `--workers 1` per the backtest determinism oracle; only random cloids differ in fills, not in `report.md` totals.) If any mismatch: STOP — this is a real fidelity regression; debug before proceeding (likely a per-config env leak between cells, or the memo serving a wrong-`config_sig` bundle).

- [ ] **Step 5: Eyeball the speedup**

Compare wall time of Step 2 vs Step 3 (the launch/done log lines print `wall=`). Warm should be materially faster per question after the first (bundle decode amortized + no per-cell `uv`/import). Note the rough factor in the PR.

- [ ] **Step 6: Resume check**

Re-run Step 3's command unchanged — expect it to find all chunks `done` and run nothing. Then `rm -rf /tmp/parity_warm/base/q0003` and re-run — expect only chunk 0 to re-launch and recompute the missing cell.

- [ ] **Step 7: Record results** in the PR description (parity = identical, speedup factor, resume verified).

---

## Self-Review notes (addressed)

- **Spec coverage:** worker mode folded into `resumable_run.py` (T4) ✓; chunk-keyed parallel supervisor + `total/N` memo budget (T5) ✓; `--chunk-size 25` default (T6) ✓; `aggregate` from report dirs (T6) ✓; per-chunk resume + per-cell skip (T4 worker skip + T5 `chunk_done`) ✓; event-mode passthrough (T2 argv) ✓; parity oracle on recorded data (T8) ✓.
- **Memo-correctness verification** (spec risk): covered structurally — `config_sig` already folds every bundle-affecting input (`hl_hip4.py:421`), and T8's byte-identical parity test is the empirical proof. No code change needed in `hlanalysis/`.
- **No placeholders:** every code/test step shows complete code; commands have expected output.
- **Type/name consistency:** `qdir(out_base, config_id, q_global)` module-level used uniformly; `ChunkState` replaces `JobState`; `run_worker_chunk` / `_invoke_run` / `build_run_argv` / `write_configs_file` / `load_configs_file` names match across tasks.
