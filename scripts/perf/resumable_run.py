#!/usr/bin/env python3
"""Parallel, resumable, crash-tolerant driver for per-question HL backtests —
single config OR a (config × question) tuning sweep.

Why this exists
---------------
The built-in `hl-bt run/tune --workers N` uses a ProcessPoolExecutor that, on a
memory-constrained box, can hang or OOM mid-corpus — and `pkill` on the parent
leaves orphaned pool workers. There is also no resume: a crashed run re-replays
every question from scratch.

This driver runs **each (config, question) as its own isolated subprocess**
(`--skip-markets i --max-markets 1 --workers 1`, i.e. in-process, no nested pool
→ tiny memory footprint), supervised by a bounded **work-queue pool**:

* a flat queue of M configs × N questions = M·N independent jobs; workers pick up
  the next undone job as each finishes (work-stealing, full core utilisation),
* a job whose result already exists is **skipped** (resume — re-running an
  overlapping/larger grid continues where it left off),
* a crashed/timed-out subprocess is **classified and retried** if retryable,
* every attempt is logged to a JSON manifest,
* the driver **reaps all child subprocess groups on exit** (no orphans).

Each job is the live-faithful `--slot` single-question run → **bit-identical to
live**; only the per-config knobs vary. Memory is bounded by the pool size, not
by M·N, so it doesn't OOM like the monolithic ProcessPool.

NOTE (intentional): per-question isolation does NOT enforce the cross-market
`max_total_inventory` / `max_concurrent_positions` caps. Accepted fidelity trade
for HL binary; validate against a shared-ledger run if it matters (buckets).

Usage
-----
    # single config (1D):
    HLBT_HL_DATA_ROOT=../../../data uv run python scripts/perf/resumable_run.py \
        --slot v31 --kind binary --start 2026-05-06 --end 2026-06-11 \
        --out-base /tmp/run_binary --workers 6

    # sweep (2D): configs.json = [{"id":"fav085","slot_config":".../a.yaml"},
    #                             {"id":"fav095","slot_config":".../b.yaml",
    #                              "env":{"HLBT_DEPTH_BACKEND":"roi"},
    #                              "scan_min":1.0,"scan_max":5.0}]
    ... --kind binary --start ... --end ... --out-base /tmp/sweep --configs configs.json --workers 6

    # resume after a crash: re-run the same command.
    # aggregate completed cells without running:
    ... --aggregate-only
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

# --- result classification ------------------------------------------------

RETRYABLE = "retryable"
DETERMINISTIC = "deterministic"
SUCCESS = "success"


def classify(returncode: int, log_text: str, report_exists: bool) -> str:
    """Decide success / retryable / deterministic from a finished subprocess."""
    if returncode == 0 and report_exists:
        return SUCCESS
    # Killed by signal (negative rc) or OOM/SIGKILL(137)/SIGTERM(143)/timeout(124).
    if returncode < 0 or returncode in (124, 137, 143):
        return RETRYABLE
    low = log_text.lower()
    if "memoryerror" in low or "killed" in low or "out of memory" in low:
        return RETRYABLE
    # Exit 0 but no report → silent failure (the ProcessPool-style death we saw).
    if returncode == 0 and not report_exists:
        return RETRYABLE
    # A real Python exception that will recur → don't loop on it.
    if "traceback (most recent call last)" in low:
        return DETERMINISTIC
    # Unknown non-zero exit: retry a bounded number of times rather than give up.
    return RETRYABLE


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


# --- config + job state ---------------------------------------------------


@dataclass
class Config:
    """One tuning cell: a variant slot config and/or per-config env + cadence."""

    id: str
    slot_config: str | None = None  # variant strategy.yaml; None → use --slot-config / base
    env: dict[str, str] = field(default_factory=dict)  # e.g. {"HLBT_DEPTH_BACKEND": "roi"}
    scan_min: float | None = None  # event-mode min interval; None → slot default
    scan_max: float | None = None


@dataclass
class ChunkState:
    chunk_idx: int
    status: str = "pending"  # pending | running | done | failed
    attempts: int = 0
    wall_s: float = 0.0
    last_class: str = ""
    last_error: str = ""
    n_cells: int = 0  # configs × questions-in-chunk
    n_done: int = 0  # cells with a .done marker


# --- per-job command + result parsing -------------------------------------


def build_run_argv(args, cfg: Config, q_global: int, out_dir: Path) -> list[str]:
    """The ``hl-bt`` argv (sans the ``uv run hl-bt`` prefix) for ONE cell —
    a single question (``--skip-markets q_global --max-markets 1``) under one
    config. Worker mode passes this straight to ``cli.main``."""
    argv = [
        "run",
        "--data-source",
        "hl_hip4",
        "--kind",
        args.kind,
        "--start",
        args.start,
        "--end",
        args.end,
        "--skip-markets",
        str(q_global),
        "--max-markets",
        "1",
        "--workers",
        "1",
        "--out-dir",
        str(out_dir),
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
            "--scan-mode",
            "event",
            "--scan-min-interval-seconds",
            str(scan_min),
            "--scan-max-interval-seconds",
            str(scan_max if scan_max is not None else 2.0),
        ]
    return argv


def report_path(out_dir: Path) -> Path:
    return out_dir / "report.md"


def done_marker(out_dir: Path) -> Path:
    return out_dir / ".done"


def qdir(out_base: Path, config_id: str, q_global: int) -> Path:
    return out_base / config_id / f"q{q_global:04d}"


def parse_pnl(out_dir: Path) -> tuple[float | None, int | None]:
    rp = report_path(out_dir)
    if not rp.exists():
        return None, None
    pnl = ntr = None
    for line in rp.read_text().splitlines():
        s = line.strip()
        if s.startswith("- total PnL:"):
            try:
                pnl = float(s.split("$", 1)[1].replace(",", ""))
            except (ValueError, IndexError):
                pass
        elif s.startswith("- trades:"):
            try:
                ntr = int(s.split(":", 1)[1])
            except (ValueError, IndexError):
                pass
    return pnl, ntr


# --- discovery ------------------------------------------------------------


def discover_count(args) -> int:
    """Number of questions in [start,end) for the kind — config-independent."""
    from hlanalysis.backtest.data.hl_hip4 import HLHip4DataSource

    data_root = os.environ.get("HLBT_HL_DATA_ROOT", "data")
    ds = HLHip4DataSource(data_root)
    klass = "priceBucket" if args.kind == "bucket" else "priceBinary"
    return len(ds.discover(start=args.start, end=args.end, kinds=(klass,)))


# --- supervisor -----------------------------------------------------------


class Driver:
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
        # state keyed by chunk index; one job = one chunk = all configs over K questions
        self.states: dict[int, ChunkState] = {c: ChunkState(chunk_idx=c) for c in range(n_chunks)}
        self._load_manifest()
        for c, st in self.states.items():
            cells = self.cells_for_chunk(c)
            st.n_cells = len(cells)
            st.n_done = sum(1 for (cid, q) in cells if done_marker(qdir(self.out_base, cid, q)).exists())
            if st.n_cells > 0 and st.n_done == st.n_cells:
                st.status = "done"
        self.queue: list[int] = [c for c, s in self.states.items() if s.status not in ("done", "failed")]
        self.running: dict[int, tuple[subprocess.Popen, float, Path]] = {}
        self._stop = False

    def qdir(self, config_id: str, q_global: int) -> Path:
        return qdir(self.out_base, config_id, q_global)

    def cells_for_chunk(self, chunk_idx: int) -> list[tuple[str, int]]:
        start, length = chunk_bounds(chunk_idx, self.n_questions, self.chunk_size)
        return [(c.id, q) for q in range(start, start + length) for c in self.configs]

    def chunk_done(self, chunk_idx: int) -> bool:
        cells = self.cells_for_chunk(chunk_idx)
        return bool(cells) and all(done_marker(qdir(self.out_base, cid, q)).exists() for cid, q in cells)

    def _load_manifest(self) -> None:
        if not self.manifest_path.exists():
            return
        try:
            data = json.loads(self.manifest_path.read_text())
            for d in data.get("chunks", []):
                key = d["chunk_idx"]
                if key in self.states:
                    self.states[key] = ChunkState(**d)
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    def save_manifest(self) -> None:
        done = sum(1 for s in self.states.values() if s.status == "done")
        failed = sum(1 for s in self.states.values() if s.status == "failed")
        payload = {
            "n_chunks": len(self.states),
            "chunk_size": self.chunk_size,
            "done": done,
            "failed": failed,
            "pending": len(self.states) - done - failed,
            "configs": [c.id for c in self.configs],
            "chunks": [asdict(s) for s in self.states.values()],
        }
        tmp = self.manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(self.manifest_path)

    def _launch(self, chunk_idx: int) -> None:
        log_path = self.out_base / f"_chunk{chunk_idx:04d}.log"
        logf = open(log_path, "w")  # noqa: SIM115
        env = dict(os.environ)
        env.setdefault("LOGURU_LEVEL", "ERROR")
        env["HLBT_INPROC_BUNDLE_MEMO"] = "1"
        env["HLBT_INPROC_BUNDLE_MEMO_WORKERS"] = str(max(1, int(self.args.workers)))
        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--_worker-chunk",
            str(chunk_idx),
            "--configs",
            str(self.configs_path),
            "--out-base",
            str(self.out_base),
            "--kind",
            self.args.kind,
            "--start",
            self.args.start,
            "--end",
            self.args.end,
            "--chunk-size",
            str(self.chunk_size),
            "--n-questions",
            str(self.n_questions),
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
        # start_new_session=True → own process group, so we can kill the whole
        # subtree (incl. python children) on timeout/shutdown. No orphans.
        p = subprocess.Popen(
            cmd, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True, cwd=os.getcwd(), env=env
        )
        self.running[chunk_idx] = (p, time.time(), log_path)
        st = self.states[chunk_idx]
        st.status = "running"
        st.attempts += 1
        print(f"[launch] chunk{chunk_idx:04d} (attempt {st.attempts}) pid={p.pid}", flush=True)

    def _kill(self, p: subprocess.Popen) -> None:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            try:
                p.kill()
            except ProcessLookupError:
                pass

    def _finish(self, chunk_idx: int, returncode: int) -> None:
        _p, t0, log_path = self.running.pop(chunk_idx)
        st = self.states[chunk_idx]
        st.wall_s = round(time.time() - t0, 1)
        log_text = ""
        try:
            log_text = log_path.read_text(errors="replace")[-4000:]
        except OSError:
            pass
        cells = self.cells_for_chunk(chunk_idx)
        st.n_done = sum(1 for (cid, q) in cells if done_marker(qdir(self.out_base, cid, q)).exists())
        all_done = self.chunk_done(chunk_idx)
        cls = classify(returncode, log_text, report_exists=all_done)
        st.last_class = cls
        if cls == SUCCESS:
            st.status = "done"
            print(
                f"[done]   chunk{chunk_idx:04d} rc={returncode} wall={st.wall_s}s cells={st.n_done}/{st.n_cells}",
                flush=True,
            )
        else:
            last_line = log_text.strip().splitlines()[-1] if log_text.strip() else ""
            st.last_error = f"rc={returncode} class={cls}; {last_line}"
            if cls == RETRYABLE and st.attempts <= self.args.max_retries:
                st.status = "pending"
                self.queue.append(chunk_idx)
                print(
                    f"[retry]  chunk{chunk_idx:04d} rc={returncode} class={cls} "
                    f"attempt={st.attempts}/{self.args.max_retries + 1}",
                    flush=True,
                )
            else:
                st.status = "failed"
                print(
                    f"[FAILED] chunk{chunk_idx:04d} rc={returncode} class={cls} "
                    f"attempts={st.attempts} :: {st.last_error}",
                    flush=True,
                )
        self.save_manifest()

    def run(self) -> int:
        def _handler(signum, _frame):
            self._stop = True
            print(f"\n[signal {signum}] draining + killing children…", flush=True)

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)
        try:
            while not self._stop and (self.queue or self.running):
                while not self._stop and self.queue and len(self.running) < self.args.workers:
                    self._launch(self.queue.pop(0))
                time.sleep(2.0)
                now = time.time()
                for key in list(self.running.keys()):
                    p, t0, _out = self.running[key]
                    rc = p.poll()
                    if rc is not None:
                        self._finish(key, rc)
                    elif now - t0 > self.args.timeout:
                        print(f"[timeout] chunk{key:04d} > {self.args.timeout}s — killing", flush=True)
                        self._kill(p)
                        try:
                            p.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            pass
                        self._finish(key, 124)
        finally:
            for key in list(self.running.keys()):
                self._kill(self.running[key][0])
            self.save_manifest()

        done = sum(1 for s in self.states.values() if s.status == "done")
        failed = [k for k, s in self.states.items() if s.status == "failed"]
        print(
            f"\n=== SUMMARY === done={done}/{len(self.states)} failed={len(failed)} {failed if failed else ''}",
            flush=True,
        )
        return 1 if failed else 0


# --- config loading + aggregation -----------------------------------------


def load_configs(args) -> list[Config]:
    if args.configs:
        raw = json.loads(Path(args.configs).read_text())
        return [Config(**c) for c in raw]
    # single-config (1D) back-compat
    return [Config(id="base", slot_config=args.slot_config)]


def write_configs_file(out_base: Path, configs: list[Config]) -> Path:
    """Persist the resolved config list so every worker subprocess reads the
    SAME cells (single-config and sweep share one path)."""
    path = out_base / "_configs.json"
    path.write_text(json.dumps([asdict(c) for c in configs], indent=2))
    return path


def load_configs_file(path: Path) -> list[Config]:
    return [Config(**c) for c in json.loads(Path(path).read_text())]


def aggregate(out_base: Path) -> None:
    """Per-config totals from completed cells, read straight from report dirs.

    Jobs are chunk-keyed now, so per-config PnL no longer lives in the manifest —
    the per-(config, question) report dirs are the source of truth.
    """
    cfg_file = out_base / "_configs.json"
    if cfg_file.exists():
        config_ids = [c.id for c in load_configs_file(cfg_file)]
    else:  # fall back to top-level dirs that contain q* cells
        config_ids = sorted(p.name for p in out_base.iterdir() if p.is_dir() and not p.name.startswith("_"))
    print("=== AGGREGATE (completed cells per config) ===")
    for cid in config_ids:
        cdir = out_base / cid
        cells = sorted(cdir.glob("q*/report.md")) if cdir.is_dir() else []
        rows = [parse_pnl(rp.parent) for rp in cells]
        rows = [(p, t) for (p, t) in rows if p is not None]
        tot = sum(p for p, _ in rows)
        tr = sum(t or 0 for _, t in rows)
        print(f"  {cid:16s}: n={len(rows):>3} totalPnL=${tot:>9.2f} trades={tr:>5}")


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
    ap.add_argument(
        "--chunk-size",
        type=int,
        default=25,
        help="questions per warm subprocess (amortizes startup + shares the bundle memo across configs)",
    )
    ap.add_argument("--max-retries", type=int, default=2, help="retries AFTER the first attempt")
    ap.add_argument("--timeout", type=float, default=3600.0, help="per-chunk wall timeout (s)")
    ap.add_argument(
        "--scan-min", type=float, default=None, help="event-mode min interval (s); omit to use slot default"
    )
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
    print(
        f"discovered {n} {args.kind} questions in [{args.start},{args.end}); "
        f"{len(configs)} config(s); chunk_size={args.chunk_size} → {num_chunks(n, args.chunk_size)} chunks",
        flush=True,
    )
    if n == 0:
        print("no questions — nothing to do", file=sys.stderr)
        return 2
    rc = Driver(args, configs, n).run()
    aggregate(out_base)
    return rc


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


if __name__ == "__main__":
    raise SystemExit(main())
