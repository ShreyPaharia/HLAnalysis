#!/usr/bin/env python3
"""Parallel, resumable, crash-tolerant driver for HL backtests — single config
OR a (config × question) tuning sweep, executed in **warm chunks**.

Why this exists
---------------
The built-in `hl-bt run/tune --workers N` uses a ProcessPoolExecutor that, on a
memory-constrained box, can hang or OOM mid-corpus — and `pkill` on the parent
leaves orphaned pool workers. There is also no resume: a crashed run re-replays
every question from scratch.

The warm-chunk model
--------------------
The unit of parallel work is a **chunk** = K consecutive questions run against
the **whole config list** inside ONE warm subprocess (worker mode, below). Within
that process the loop is question-outer / config-inner with the in-process bundle
memo on (`HLBT_INPROC_BUNDLE_MEMO=1`): the first config to touch a question
decodes its event bundle, the other M-1 configs reuse it from the memo. So for an
M-config × N-question sweep the expensive data-decode + settlement work drops from
M·N → ~N, and the `uv`/import cold-start from M·N → ~⌈N/K⌉ — typically an ~M×
speedup on the data path, with no change to results (each cell is the same
live-faithful `--slot` run → bit-identical to live).

The supervisor runs each chunk as its OWN process group (own subprocess, so an
OOM/crash kills one chunk, not the run), supervised by a bounded **work-queue
pool**:

* a queue of ⌈N/K⌉ chunk jobs; workers pick up the next undone chunk as each
  finishes (work-stealing, full core utilisation),
* `--workers N` runs N chunks concurrently, each with memo budget total/N
  (`HLBT_INPROC_BUNDLE_MEMO_WORKERS`), so aggregate memo RAM stays bounded,
* a chunk is **done** when every per-(config, question) `.done` marker exists;
  cells already done are **skipped** (per-chunk resume, per-cell skip),
* a crashed/timed-out chunk is **classified and retried** if retryable,
* every attempt is logged to a JSON manifest; the driver **reaps all child
  subprocess groups on exit** (no orphans).

Per-(config, question) report dirs (`out_base/<config_id>/qNNNN/`) are unchanged —
the chunk only governs which subprocess computes them.

NOTE (intentional): a chunk's per-question runs do NOT enforce the cross-market
`max_total_inventory` / `max_concurrent_positions` caps. Accepted fidelity trade
for HL binary; validate against a shared-ledger run if it matters (buckets).

Usage
-----
    # single config (warm chunks of 25 questions/subprocess by default):
    HLBT_HL_DATA_ROOT=../../data uv run python scripts/perf/resumable_run.py \
        --slot v31 --kind binary --start 2026-05-06 --end 2026-06-11 \
        --out-base /tmp/run_binary --workers 6 --chunk-size 25 --scan-min 0.5 --scan-max 2.0

    # sweep: configs.json = [{"id":"fav085","slot_config":".../a.yaml"},
    #                        {"id":"fav095","slot_config":".../b.yaml",
    #                         "scan_min":1.0,"scan_max":5.0}]
    ... --kind binary --start ... --end ... --out-base /tmp/sweep --configs configs.json --workers 6

    # resume after a crash: re-run the same command (done chunks are skipped).
    # aggregate completed cells without running:
    ... --aggregate-only

    # --chunk-size 1 reproduces the historical per-question resume granularity
    # (still warm — imports paid once per question-subprocess).

    # Polymarket: same driver, --data-source polymarket. Recorded (live-faithful)
    # path:
    HLBT_HL_DATA_ROOT=../../data HLBT_PM_CACHE_ROOT=../../data/sim \
        uv run python scripts/perf/resumable_run.py \
        --data-source polymarket --slot v31_pm --kind binary \
        --start 2026-05-27 --end 2026-06-17 --out-base /tmp/pm_sweep \
        --pm-book-source recorded --pm-reference-source binance_bbo \
        --pm-binance-bbo-product-type spot --fee-model pm_binary --fee-rate 0.07 \
        --workers 6 --chunk-size 25
    # …or the klines/pulled path: --pm-reference-source klines (synthetic book).
    # PM/fee flags are per-config overridable in configs.json just like
    # scan_min/scan_max (e.g. {"id":"rec","pm_book_source":"recorded"}).
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
    """One tuning cell: a variant slot config and/or per-config env + cadence.

    The PM / fee knobs (``pm_*``, ``fee_*``, ``cache_root``) are per-config
    overrides of the corresponding driver-level flags: ``None`` falls back to
    the ``args`` value, mirroring how ``scan_min``/``scan_max``/``slot_config``
    already work. They only affect the inner ``hl-bt run`` argv when the run is
    ``--data-source polymarket`` (``cache_root`` aside, which is emitted for PM).
    ``--data-source`` itself is driver-global (it governs question discovery,
    which is config-independent), so it is NOT per-config.
    """

    id: str
    slot_config: str | None = None  # variant strategy.yaml; None → use --slot-config / base
    env: dict[str, str] = field(default_factory=dict)  # e.g. {"HLBT_DEPTH_BACKEND": "roi"}
    scan_min: float | None = None  # event-mode min interval; None → slot default
    scan_max: float | None = None
    # PM / fee per-config overrides (None → fall back to the driver-level arg).
    pm_flavor: str | None = None
    pm_book_source: str | None = None
    pm_reference_source: str | None = None
    pm_binance_bbo_product_type: str | None = None
    fee_model: str | None = None
    fee_rate: float | None = None
    cache_root: str | None = None


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
    config. Worker mode passes this straight to ``cli.main``.

    HL (``--data-source hl_hip4``) argv is byte-identical to the legacy driver;
    the PM (``--data-source polymarket``) branch additionally emits the PM
    source flags (``--pm-flavor`` / ``--pm-book-source`` /
    ``--pm-reference-source`` / ``--pm-binance-bbo-product-type``) and the fee
    flags (``--fee-model`` / ``--fee-rate``), so PM backtests run through the
    same warm-chunk/resume/no-orphan driver as HL.
    """
    data_source = getattr(args, "data_source", None) or "hl_hip4"
    argv = [
        "run",
        "--data-source",
        data_source,
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
    # Underlying (hl_hip4 only). BTC is hl-bt's default, so emit the flag ONLY for
    # non-BTC underlyings — keeps the BTC argv byte-identical to the legacy driver.
    underlying = getattr(args, "underlying", None)
    if data_source == "hl_hip4" and underlying and underlying != "BTC":
        argv += ["--underlying", underlying]
    # Cadence: `--inner-scan-mode fixed` forces the inner run onto a fixed grid
    # (--scanner-interval-seconds), overriding the slot's event-mode default —
    # used for churn-fidelity bucket sweeps. Otherwise fall through to event mode
    # when a scan_min is supplied (legacy behaviour).
    inner_scan_mode = getattr(args, "inner_scan_mode", None)
    if inner_scan_mode == "fixed":
        argv += [
            "--scan-mode",
            "fixed",
            "--scanner-interval-seconds",
            str(int(getattr(args, "inner_scan_interval", 1) or 1)),
        ]
    else:
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
    if data_source == "polymarket":
        argv += _pm_fee_argv(args, cfg)
    return argv


def _resolve(cfg: Config, args, name: str):
    """Per-config override of a driver-level arg: ``cfg.<name>`` when set, else
    ``args.<name>`` (mirrors the scan_min/scan_max fallback)."""
    cfg_val = getattr(cfg, name, None)
    if cfg_val is not None:
        return cfg_val
    return getattr(args, name, None)


def _pm_fee_argv(args, cfg: Config) -> list[str]:
    """PM source + fee flags for the inner ``hl-bt run`` argv (polymarket only).

    Each knob is a per-config override of the driver-level arg. ``--cache-root``
    is emitted only when set (PM cache root is otherwise read from
    ``HLBT_PM_CACHE_ROOT`` in the inherited environment); HL never gets a
    ``--cache-root`` here so its argv stays byte-identical.
    """
    out: list[str] = []
    pm_flavor = _resolve(cfg, args, "pm_flavor") or "btc_updown"
    pm_book_source = _resolve(cfg, args, "pm_book_source") or "synthetic"
    pm_reference_source = _resolve(cfg, args, "pm_reference_source") or "klines"
    pm_bbo_product = _resolve(cfg, args, "pm_binance_bbo_product_type") or "perp"
    fee_model = _resolve(cfg, args, "fee_model") or "flat"
    fee_rate = _resolve(cfg, args, "fee_rate")
    out += ["--pm-flavor", str(pm_flavor)]
    out += ["--pm-book-source", str(pm_book_source)]
    out += ["--pm-reference-source", str(pm_reference_source)]
    out += ["--pm-binance-bbo-product-type", str(pm_bbo_product)]
    out += ["--fee-model", str(fee_model)]
    if fee_rate is not None:
        out += ["--fee-rate", str(fee_rate)]
    cache_root = _resolve(cfg, args, "cache_root")
    if cache_root:
        out += ["--cache-root", str(cache_root)]
    return out


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
    """Number of questions in [start,end) for the kind — config-independent.

    Branches on ``--data-source``: PM enumerates via ``PolymarketDataSource``
    (manifest-backed), HL via ``HLHip4DataSource`` (recorded parquet). The HL
    path is unchanged from the legacy driver.
    """
    data_source = getattr(args, "data_source", None) or "hl_hip4"
    if data_source == "polymarket":
        import hlanalysis.backtest.data.polymarket as pm_mod
        from hlanalysis.backtest.core.source_config import PM_FLAVORS

        cache_root = getattr(args, "cache_root", None) or os.environ.get("HLBT_PM_CACHE_ROOT", "data/sim")
        flavor = getattr(args, "pm_flavor", None) or "btc_updown"
        ds = pm_mod.PolymarketDataSource(cache_root=Path(cache_root), **PM_FLAVORS[flavor])
        return len(ds.discover(start=args.start, end=args.end, kind=args.kind))

    import hlanalysis.backtest.data.hl_hip4 as hl_mod

    data_root = os.environ.get("HLBT_HL_DATA_ROOT", "data")
    ds = hl_mod.HLHip4DataSource(data_root)
    klass = "priceBucket" if args.kind == "bucket" else "priceBinary"
    underlying = getattr(args, "underlying", None) or "BTC"
    return len(ds.discover(start=args.start, end=args.end, kinds=(klass,), underlying=underlying))


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
            # The on-disk .done markers are authoritative for completion. A chunk
            # is done iff every cell is present; otherwise it must run — even if a
            # stale manifest marked it "done" but a cell was since deleted.
            if st.n_cells > 0 and st.n_done == st.n_cells:
                st.status = "done"
            elif st.status == "done":
                st.status = "pending"
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
            # Track + PM/fee defaults: the worker reconstructs the inner argv via
            # build_run_argv, which reads these off its own parsed args (per-config
            # overrides still come from the persisted _configs.json).
            "--data-source",
            self.args.data_source,
            "--pm-flavor",
            self.args.pm_flavor,
            "--pm-book-source",
            self.args.pm_book_source,
            "--pm-reference-source",
            self.args.pm_reference_source,
            "--pm-binance-bbo-product-type",
            self.args.pm_binance_bbo_product_type,
            "--fee-model",
            self.args.fee_model,
            "--fee-rate",
            str(self.args.fee_rate),
        ]
        if self.args.cache_root:
            cmd += ["--cache-root", self.args.cache_root]
        if self.args.slot:
            cmd += ["--slot", self.args.slot]
        if self.args.slot_config:
            cmd += ["--slot-config", self.args.slot_config]
        if self.args.slot_class:
            cmd += ["--slot-class", self.args.slot_class]
        if self.args.strategy:
            cmd += ["--strategy", self.args.strategy]
        if getattr(self.args, "underlying", None):
            cmd += ["--underlying", self.args.underlying]
        if self.args.scan_min is not None:
            cmd += ["--scan-min", str(self.args.scan_min)]
        if self.args.scan_max is not None:
            cmd += ["--scan-max", str(self.args.scan_max)]
        if getattr(self.args, "inner_scan_mode", None) is not None:
            cmd += [
                "--inner-scan-mode",
                self.args.inner_scan_mode,
                "--inner-scan-interval",
                str(self.args.inner_scan_interval),
            ]
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
    ap.add_argument(
        "--data-source",
        choices=["hl_hip4", "polymarket"],
        default="hl_hip4",
        help="Question track. hl_hip4 (default) = HL HIP-4 recorded binaries/buckets; "
        "polymarket = PM L2 (mirrors `hl-bt run --data-source polymarket`).",
    )
    ap.add_argument("--slot", default=None)
    ap.add_argument("--slot-config", default=None)
    ap.add_argument("--slot-class", default=None)
    ap.add_argument("--strategy", default=None)
    ap.add_argument(
        "--underlying",
        default="BTC",
        help="(hl_hip4 only) Underlying to discover questions for (BTC/ETH/SOL/HYPE). "
        "Reference feed reads the matching HL perp. Default BTC.",
    )
    # PM source flags (polymarket only) — mirror `hl-bt run`. Per-config
    # overridable via configs.json (same mechanism as scan_min/scan_max).
    ap.add_argument(
        "--pm-flavor",
        default="btc_updown",
        help="(polymarket only) Which PM series + reference asset to load.",
    )
    ap.add_argument(
        "--pm-book-source",
        choices=["synthetic", "recorded"],
        default="synthetic",
        help="(polymarket only) Fill-book source. `recorded` feeds the real L2 book.",
    )
    ap.add_argument(
        "--pm-reference-source",
        choices=["klines", "binance_bbo", "klines_1s"],
        default="klines",
        help="(polymarket only) Reference-feed source. `klines` = cached 1m Binance "
        "klines (pulled path); `binance_bbo` = recorded Binance BBO ticks (live-faithful).",
    )
    ap.add_argument(
        "--pm-binance-bbo-product-type",
        choices=["perp", "spot"],
        default="perp",
        dest="pm_binance_bbo_product_type",
        help="(polymarket binance_bbo only) Binance product type. `spot` matches PM's "
        "settlement instrument (Binance SPOT 1m close).",
    )
    # Fee model (mirrors `hl-bt run`). PM uses pm_binary; HL uses flat.
    ap.add_argument(
        "--fee-model",
        choices=["flat", "pm_binary"],
        default="flat",
        help="Binary-leg fee model. `flat` (HL) vs `pm_binary` (Polymarket curve).",
    )
    ap.add_argument(
        "--fee-rate",
        type=float,
        default=0.07,
        help="feeRate for --fee-model pm_binary (PM crypto = 0.07).",
    )
    ap.add_argument(
        "--cache-root",
        default=None,
        help="(polymarket only) Override the PM cache/data root (env: HLBT_PM_CACHE_ROOT).",
    )
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
    ap.add_argument(
        "--inner-scan-mode",
        choices=["event", "fixed"],
        default=None,
        dest="inner_scan_mode",
        help="force the inner hl-bt run cadence. 'fixed' → --scan-mode fixed --scanner-interval-seconds "
        "(--inner-scan-interval), overriding the slot's event-mode default (churn-fidelity sweeps).",
    )
    ap.add_argument(
        "--inner-scan-interval",
        type=int,
        default=1,
        dest="inner_scan_interval",
        help="fixed-mode scanner interval seconds (with --inner-scan-mode fixed).",
    )
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
