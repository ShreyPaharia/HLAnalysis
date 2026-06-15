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
class JobState:
    config_id: str
    idx: int
    status: str = "pending"  # pending | running | done | failed
    attempts: int = 0
    wall_s: float = 0.0
    last_class: str = ""
    last_error: str = ""
    pnl: float | None = None
    n_trades: int | None = None


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
        self.configs = {c.id: c for c in configs}
        self.out_base = Path(args.out_base)
        self.out_base.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.out_base / "manifest.json"
        # state keyed by (config_id, idx)
        self.states: dict[tuple[str, int], JobState] = {
            (c.id, i): JobState(config_id=c.id, idx=i) for c in configs for i in range(n_questions)
        }
        self._load_manifest()
        for key, st in self.states.items():
            if done_marker(self.qdir(*key)).exists():
                st.status = "done"
                st.pnl, st.n_trades = parse_pnl(self.qdir(*key))
        self.queue: list[tuple[str, int]] = [k for k, s in self.states.items() if s.status not in ("done", "failed")]
        self.running: dict[tuple[str, int], tuple[subprocess.Popen, float, Path]] = {}
        self._stop = False

    def qdir(self, config_id: str, idx: int) -> Path:
        return self.out_base / config_id / f"q{idx:04d}"

    def _load_manifest(self) -> None:
        if not self.manifest_path.exists():
            return
        try:
            data = json.loads(self.manifest_path.read_text())
            for d in data.get("jobs", []):
                key = (d["config_id"], d["idx"])
                if key in self.states:
                    self.states[key] = JobState(**d)
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    def save_manifest(self) -> None:
        done = sum(1 for s in self.states.values() if s.status == "done")
        failed = sum(1 for s in self.states.values() if s.status == "failed")
        payload = {
            "n_jobs": len(self.states),
            "done": done,
            "failed": failed,
            "pending": len(self.states) - done - failed,
            "configs": list(self.configs),
            "jobs": [asdict(s) for s in self.states.values()],
        }
        tmp = self.manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(self.manifest_path)

    def _launch(self, key: tuple[str, int]) -> None:
        config_id, idx = key
        cfg = self.configs[config_id]
        out_dir = self.qdir(config_id, idx)
        out_dir.mkdir(parents=True, exist_ok=True)
        logf = open(out_dir / "run.log", "w")  # noqa: SIM115
        env = dict(os.environ)
        env.setdefault("LOGURU_LEVEL", "ERROR")
        env.update(cfg.env)  # per-config env (e.g. HLBT_DEPTH_BACKEND)
        cmd = ["uv", "run", "hl-bt"] + build_run_argv(self.args, cfg, q_global=idx, out_dir=out_dir)
        # start_new_session=True → own process group, so we can kill the whole
        # subtree (incl. uv/python children) on timeout/shutdown. No orphans.
        p = subprocess.Popen(
            cmd, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True, cwd=os.getcwd(), env=env
        )
        self.running[key] = (p, time.time(), out_dir)
        st = self.states[key]
        st.status = "running"
        st.attempts += 1
        print(f"[launch] {config_id}/q{idx:04d} (attempt {st.attempts}) pid={p.pid}", flush=True)

    def _kill(self, p: subprocess.Popen) -> None:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            try:
                p.kill()
            except ProcessLookupError:
                pass

    def _finish(self, key: tuple[str, int], returncode: int) -> None:
        config_id, idx = key
        _p, t0, out_dir = self.running.pop(key)
        st = self.states[key]
        st.wall_s = round(time.time() - t0, 1)
        log_text = ""
        try:
            log_text = (out_dir / "run.log").read_text(errors="replace")[-4000:]
        except OSError:
            pass
        cls = classify(returncode, log_text, report_path(out_dir).exists())
        st.last_class = cls
        if cls == SUCCESS:
            done_marker(out_dir).write_text(str(int(time.time())))
            st.status = "done"
            st.pnl, st.n_trades = parse_pnl(out_dir)
            print(
                f"[done]   {config_id}/q{idx:04d} rc={returncode} wall={st.wall_s}s pnl={st.pnl} trades={st.n_trades}",
                flush=True,
            )
        else:
            last_line = log_text.strip().splitlines()[-1] if log_text.strip() else ""
            st.last_error = f"rc={returncode} class={cls}; {last_line}"
            if cls == RETRYABLE and st.attempts <= self.args.max_retries:
                st.status = "pending"
                self.queue.append(key)
                print(
                    f"[retry]  {config_id}/q{idx:04d} rc={returncode} class={cls} "
                    f"attempt={st.attempts}/{self.args.max_retries + 1}",
                    flush=True,
                )
            else:
                st.status = "failed"
                print(
                    f"[FAILED] {config_id}/q{idx:04d} rc={returncode} class={cls} "
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
                        print(f"[timeout] {key[0]}/q{key[1]:04d} > {self.args.timeout}s — killing", flush=True)
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
    """Per-config totals from completed cells (reads the manifest)."""
    mp = out_base / "manifest.json"
    if not mp.exists():
        print("no manifest", file=sys.stderr)
        return
    m = json.loads(mp.read_text())
    by_cfg: dict[str, list[dict]] = {}
    for j in m["jobs"]:
        if j["status"] == "done":
            by_cfg.setdefault(j["config_id"], []).append(j)
    print("=== AGGREGATE (completed cells per config) ===")
    for cid in sorted(by_cfg):
        js = by_cfg[cid]
        tot = sum(j["pnl"] or 0 for j in js)
        tr = sum(j["n_trades"] or 0 for j in js)
        walls = [j["wall_s"] for j in js]
        mean_w = sum(walls) / len(walls) if walls else 0
        print(f"  {cid:16s}: n={len(js):>3} totalPnL=${tot:>9.2f} trades={tr:>5} mean_wall={mean_w:.0f}s")


def main() -> int:
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
    ap.add_argument("--max-retries", type=int, default=2, help="retries AFTER the first attempt")
    ap.add_argument("--timeout", type=float, default=3600.0, help="per-job wall timeout (s)")
    ap.add_argument(
        "--scan-min", type=float, default=None, help="event-mode min interval (s); omit to use slot default"
    )
    ap.add_argument("--scan-max", type=float, default=None)
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()

    out_base = Path(args.out_base)
    if args.aggregate_only:
        aggregate(out_base)
        return 0

    configs = load_configs(args)
    n = discover_count(args)
    print(
        f"discovered {n} {args.kind} questions in [{args.start},{args.end}); "
        f"{len(configs)} config(s) → {n * len(configs)} jobs",
        flush=True,
    )
    if n == 0:
        print("no questions — nothing to do", file=sys.stderr)
        return 2
    rc = Driver(args, configs, n).run()
    aggregate(out_base)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
