"""Disk cache for built FastPathBundles.

Key = sha256(question_id + sorted[(path,size,mtime_ns)] + BUILD_VERSION). Any
re-record (size/mtime change) or assembly-logic change (BUILD_VERSION bump in
_fastpath_core) misses → rebuild. Stat-only keying (never reads file bytes) so
it preserves the speedup.

The cache *management* (concurrency-safe atomic writes, LRU/size-cap eviction,
key→file indexing) is delegated to :mod:`diskcache`; this module keeps only the
domain-specific serializer — the column-split, delta-encoded npz, a ~5.7x
compression no generic cache offers — plugged in as a custom ``Disk`` subclass.
"""

from __future__ import annotations

import hashlib
import logging
import os
import uuid
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path

import diskcache
import numpy as np
from diskcache.core import MODE_BINARY, UNKNOWN

from ..core.events import ReferenceEvent, SettlementEvent, TradeEvent

# Bound as a module global so it's read reflectively via `_self._BUILD_VERSION`
# (cache key + version-shard pruning) and stays monkeypatchable in tests. ruff
# can't see the reflective use, so do not let F401 strip this import.
from ._fastpath_core import BUILD_VERSION as _BUILD_VERSION  # noqa: F401
from ._fastpath_core import FastPathBundle, LegArrays, event_dtype

log = logging.getLogger(__name__)

# Integer struct columns that are monotone-increasing (nanosecond timestamps).
# Delta-encoding them before deflate turns large near-constant values into small
# deltas that compress to almost nothing. Floats are NOT delta'd (lossy via
# float accumulation); they rely on column-split + dictionary-like redundancy.
_DELTA_FIELDS = ("exch_ts", "local_ts")


def _delta(a: np.ndarray) -> np.ndarray:
    """Successive-difference encode an integer array (first element kept raw).

    ``_undelta(_delta(a)) == a`` exactly for int arrays. No-op for size < 2.
    """
    a = np.ascontiguousarray(a)
    if a.size < 2:
        return a
    out = a.copy()
    out[1:] = a[1:] - a[:-1]
    return out


def _undelta(d: np.ndarray) -> np.ndarray:
    """Inverse of ``_delta`` — cumulative sum recovers the original int array."""
    if d.size < 2:
        return d
    return np.cumsum(d, dtype=d.dtype)


def cache_key(question_id: str, source_files: list[Path], config_sig: str = "") -> str:
    """Compute a cache key from question_id, source file metadata, BUILD_VERSION,
    and ``config_sig``.

    ``config_sig`` MUST capture every non-source-file input that changes the
    built bundle — notably the reference resample period (coupled to
    vol_sampling_dt_seconds) and reference/book source mode. Omitting it lets a
    bundle built at one dt alias to a request at another dt (dt=5 vs dt=60),
    silently serving the wrong reference events.

    Uses the module-level ``_BUILD_VERSION`` so ``monkeypatch.setattr`` in tests
    can override it and verify version changes produce different keys.
    """
    import hlanalysis.backtest.data._event_array_cache as _self

    h = hashlib.sha256()
    h.update(question_id.encode())
    h.update(str(_self._BUILD_VERSION).encode())
    h.update(config_sig.encode())
    for p in sorted(source_files, key=str):
        try:
            st = Path(p).stat()
            h.update(f"{p}:{st.st_size}:{st.st_mtime_ns}".encode())
        except OSError:
            h.update(f"{p}:absent".encode())
    return h.hexdigest()


def _save(path: Path, b: FastPathBundle) -> None:
    """Serialize a FastPathBundle to a .npz file at ``path``.

    ReferenceEvent fields: ts_ns, symbol, high, low, close, open (default 0.0)
    SettlementEvent fields: ts_ns, question_idx, outcome, symbol (default "")
    """
    legs = sorted(b.leg_arrays)
    payload: dict[str, np.ndarray] = {
        "__legs__": np.array(legs, dtype=object),
        "__ref__": np.array(
            [(e.ts_ns, e.symbol, e.high, e.low, e.close, e.open) for e in b.reference_events],
            dtype=object,
        ),
        "__settle__": np.array(
            [(e.ts_ns, e.question_idx, e.outcome, e.symbol) for e in b.settlement_events],
            dtype=object,
        ),
        # SHR-97: persist the reference_events_are_raw_ticks flag so a cache hit
        # on the raw-tick path ("--reference-ticks raw") restores True, not False.
        # Pre-v7 npz lacked this key → the runner called apply_reference (bar path)
        # instead of apply_reference_tick (raw-tick path) on every cache hit,
        # inflating σ and collapsing 74 trades → 3 on v31 binary.
        "__ref_raw_ticks__": np.array([bool(b.reference_events_are_raw_ticks)], dtype=bool),
    }
    for i, sym in enumerate(legs):
        la = b.leg_arrays[sym]
        # Column-split: store each struct field as its own homogeneous array so
        # deflate sees contiguous like-typed bytes (zeros, repeats, monotone
        # runs) instead of a 64-byte interleaved record. Timestamp columns are
        # delta-encoded. ``_load`` reassembles the event_dtype struct.
        ev = la.events
        for name in ev.dtype.names:
            col = np.ascontiguousarray(ev[name])
            payload[f"ev_{i}__{name}"] = _delta(col) if name in _DELTA_FIELDS else col
        payload[f"bts_{i}"] = _delta(np.ascontiguousarray(la.book_ts))
        # SHR-94: per-snapshot best ask/bid for the IOC marketability re-check.
        payload[f"sba_{i}"] = np.ascontiguousarray(la.snap_best_ask)
        payload[f"sbb_{i}"] = np.ascontiguousarray(la.snap_best_bid)
        # Per-leg trade events (SHR-78): persist (ts, px, sz, side) so a cache
        # HIT restores the recent_volume_usd inputs. Omitting these silently
        # zeroed the volume gate on every cached run → 0 trades for any strategy
        # with min_recent_volume_usd > 0. Column-split + delta-encoded ts, same
        # as the book columns; side encoded buy=0/sell=1 to stay homogeneous.
        trades = b.trade_events_per_leg.get(sym, []) if b.trade_events_per_leg else []
        tr_ts = np.array([t.ts_ns for t in trades], dtype=np.int64)
        payload[f"tr_{i}__ts"] = _delta(tr_ts)
        payload[f"tr_{i}__px"] = np.array([t.price for t in trades], dtype=np.float64)
        payload[f"tr_{i}__sz"] = np.array([t.size for t in trades], dtype=np.float64)
        payload[f"tr_{i}__side"] = np.array([1 if t.side == "sell" else 0 for t in trades], dtype=np.int8)
    # np.savez_compressed appends ".npz" to the path if not already present.
    # Use a tmp file that already ends in ".npz" so the written file
    # matches the tmp variable name, then atomically rename to final path.
    # Compressed: assembled book bundles are large but highly repetitive
    # (zero-padded fields, monotone timestamps), so deflate cuts the on-disk
    # corpus by ~an order of magnitude — the disk-blowup mitigation.
    #
    # The tmp name MUST be unique per writer (pid + uuid), not derived from the
    # final key alone: under `tune --workers N` several spawn workers rebuild the
    # SAME bundle concurrently, and a shared tmp name made them truncate each
    # other's bytes (corrupt npz) and ENOENT on the second rename (SHR-71 rebuild
    # storm). With a private tmp each writer's bytes are intact and the atomic
    # rename is simply last-writer-wins.
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.{uuid.uuid4().hex}.tmp.npz")
    try:
        np.savez_compressed(tmp, **payload)
        tmp.replace(path)
    finally:
        # If savez or replace failed, don't leave a private orphan behind.
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _load(path: Path) -> FastPathBundle:
    """Deserialize a FastPathBundle from a .npz file."""
    z = np.load(path, allow_pickle=True)
    legs = list(z["__legs__"])
    leg_arrays = {}
    for i, sym in enumerate(legs):
        # Reassemble the event_dtype struct from its column-split fields,
        # un-delta'ing the timestamp columns.
        first = z[f"ev_{i}__{event_dtype.names[0]}"]
        ev = np.zeros(len(first), dtype=event_dtype)
        for name in event_dtype.names:
            col = z[f"ev_{i}__{name}"]
            ev[name] = _undelta(col) if name in _DELTA_FIELDS else col
        # SHR-94: restore snap_best arrays; tolerate pre-v6 npz (empty → recheck skipped).
        sba = z[f"sba_{i}"] if f"sba_{i}" in z.files else np.zeros(0, dtype=np.float64)
        sbb = z[f"sbb_{i}"] if f"sbb_{i}" in z.files else np.zeros(0, dtype=np.float64)
        leg_arrays[sym] = LegArrays(
            events=ev,
            book_ts=_undelta(z[f"bts_{i}"]),
            snap_best_ask=sba,
            snap_best_bid=sbb,
        )
    # Per-leg trade events (SHR-78). Tolerate pre-v5 npz that predate trade
    # persistence: a missing ``tr_*`` key → empty list (the BUILD_VERSION bump
    # evicts those, but stay defensive in case of a hand-rolled/partial file).
    trade_events_per_leg: dict[str, list[TradeEvent]] = {}
    for i, sym in enumerate(legs):
        if f"tr_{i}__ts" not in z.files:
            trade_events_per_leg[sym] = []
            continue
        tr_ts = _undelta(z[f"tr_{i}__ts"])
        tr_px = z[f"tr_{i}__px"]
        tr_sz = z[f"tr_{i}__sz"]
        tr_side = z[f"tr_{i}__side"]
        trade_events_per_leg[sym] = [
            TradeEvent(
                ts_ns=int(tr_ts[j]),
                symbol=sym,
                side="sell" if int(tr_side[j]) == 1 else "buy",
                price=float(tr_px[j]),
                size=float(tr_sz[j]),
            )
            for j in range(len(tr_ts))
        ]
    ref = [
        ReferenceEvent(
            ts_ns=int(r[0]),
            symbol=str(r[1]),
            high=float(r[2]),
            low=float(r[3]),
            close=float(r[4]),
            open=float(r[5]),
        )
        for r in z["__ref__"]
    ]
    settle = [
        SettlementEvent(
            ts_ns=int(r[0]),
            question_idx=int(r[1]),
            outcome=r[2],
            symbol=str(r[3]) if r[3] is not None else "",
        )
        for r in z["__settle__"]
    ]
    # SHR-97: restore reference_events_are_raw_ticks. Tolerate pre-v7 npz that
    # predate this key (BUILD_VERSION bump evicts those, but stay defensive).
    ref_raw_ticks = bool(z["__ref_raw_ticks__"][0]) if "__ref_raw_ticks__" in z.files else False
    return FastPathBundle(
        leg_arrays=leg_arrays,
        reference_events=ref,
        settlement_events=settle,
        trade_events_per_leg=trade_events_per_leg,
        reference_events_are_raw_ticks=ref_raw_ticks,
    )


class _NpzDisk(diskcache.Disk):
    """diskcache ``Disk`` that (de)serializes ``FastPathBundle`` values via the
    column-split delta-encoded npz serializer (``_save`` / ``_load``).

    A bundle is written as one npz file by ``_save`` (its private-tmp + atomic
    rename retained, harmless under diskcache's per-write random filenames) and
    read back by ``_load`` straight from the stored path — no extra in-memory
    copy. Non-bundle values (diskcache stores none here) fall back to default
    pickle handling.
    """

    def store(self, value, read, key=UNKNOWN):
        if isinstance(value, FastPathBundle):
            filename, full_path = self.filename(key, value)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            _save(Path(full_path), value)
            return os.path.getsize(full_path), MODE_BINARY, filename, None
        return super().store(value, read, key)

    def fetch(self, mode, filename, value, read):
        if mode == MODE_BINARY and not read and filename is not None:
            return _load(Path(os.path.join(self._directory, filename)))
        return super().fetch(mode, filename, value, read)


# Per-(root, BUILD_VERSION) open Cache objects, reused across calls — each holds
# a SQLite connection; reuse is diskcache's recommended pattern and avoids
# re-opening on every question.
_CACHES: dict[tuple[str, int], diskcache.Cache] = {}
_MISS = object()


def _prune_stale_versions(root: Path, version: int) -> None:
    """Remove cache shards from a superseded BUILD_VERSION.

    Each BUILD_VERSION lives in its own ``v{N}`` subdir; entries from an old
    version are unreachable (the key folds in BUILD_VERSION) so the whole stale
    subdir is removed, preserving the old version-orphan eviction. Any open
    Cache handle for a pruned shard is closed first. Best-effort.
    """
    import shutil

    for child in root.glob("v*"):
        if not child.is_dir() or child.name == f"v{version}":
            continue
        try:
            other = int(child.name[1:])
        except ValueError:
            continue
        cached = _CACHES.pop((str(root), other), None)
        if cached is not None:
            try:
                cached.close()
            except Exception:
                pass
        shutil.rmtree(child, ignore_errors=True)


def _get_cache(cache_dir: Path) -> diskcache.Cache:
    """Return the (reused) ``diskcache.Cache`` for ``cache_dir`` at the current
    BUILD_VERSION, pruning stale-version shards and re-applying the size cap."""
    import hlanalysis.backtest.data._event_array_cache as _self

    version = _self._BUILD_VERSION
    root = Path(cache_dir)
    ckey = (str(root), version)
    cache = _CACHES.get(ckey)
    if cache is None:
        root.mkdir(parents=True, exist_ok=True)
        _prune_stale_versions(root, version)
        cache = diskcache.Cache(
            str(root / f"v{version}"),
            disk=_NpzDisk,
            size_limit=_cache_max_bytes(),
            # Match the old write-time LRU (least-recently-WRITTEN) eviction and
            # avoid mutating the SQLite index on the hot read path.
            eviction_policy="least-recently-stored",
        )
        _CACHES[ckey] = cache
    else:
        # Honour a size cap changed (e.g. via env) since the cache was opened.
        cache.reset("size_limit", _cache_max_bytes())
    return cache


def caching_enabled() -> bool:
    """Event-array disk caching is default-ON.

    The two reasons it was once opt-in are both closed: poisoning (config_sig is
    now in the key) and disk runaway (version-prefixed eviction + a size cap +
    column-split compression bound the footprint). So normal `run`/`tune` cache
    by default for the sweep speedup.

    Escape hatches (both inherited by spawn workers via the environment):
      * ``HLBT_NO_CACHE`` (set by ``--fresh``/``--no-cache``) — hard off, for
        when you suspect a stale/poisoned entry and want a guaranteed fresh
        build.
      * ``HLBT_CACHE_EVENT_ARRAYS=0`` — also off (explicit disable; the test
        suite sets this for hermeticity). Any other value, or unset, is ON.
    """
    import os

    if os.environ.get("HLBT_NO_CACHE"):
        return False
    v = os.environ.get("HLBT_CACHE_EVENT_ARRAYS")
    if v is None:
        return True
    return v.strip().lower() not in ("0", "", "false", "no", "off")


def _cache_max_bytes() -> int:
    """Cache size cap in bytes. Override via ``HLBT_CACHE_MAX_BYTES`` (exact) or
    ``HLBT_CACHE_MAX_GB``; default 20 GiB — a backstop, not a tight budget."""
    import os

    b = os.environ.get("HLBT_CACHE_MAX_BYTES")
    if b is not None:
        return int(b)
    gb = float(os.environ.get("HLBT_CACHE_MAX_GB", "20"))
    return int(gb * 1024**3)


# --- Opt-in process-level bundle memo --------------------------------------
# A tuning sweep replays the *same* question across many param cells; each cell
# reconstructs the data source but the built event-array bundle is
# param-independent, so re-running cache_key (file stat) + npz inflate per cell
# is pure waste. When HLBT_INPROC_BUNDLE_MEMO=1 (set by `tune`), cached_bundle
# keeps an in-process LRU of bundles keyed on (question_id, config_sig) and
# returns the memoized object directly — skipping disk entirely on repeat.
#
# Default-OFF: a single `run` processes each question once (no repeat → no
# benefit) and the memo assumes source files are immutable for the process
# lifetime, which a `run`/`tune` over historical parquet satisfies but the
# mtime-invalidation contract test deliberately violates. Off by default keeps
# both honest.
#
# RAM bound (SHR-71): each bundle is ~100-140 MB of decompressed numpy arrays.
# Bounding the memo by ENTRY COUNT alone (the old default 512) let one worker
# retain ~65 GB, and `tune` spawns N such worker processes EACH with its own
# module-global memo → ~TB aggregate at N=12 → OOM. So the memo is bounded
# primarily by retained BYTES (worker-aware), with the count cap kept only as a
# secondary backstop. See ``_inproc_max_bytes`` for the worker-aware budget.
_INPROC_MEMO: OrderedDict[tuple[str, str], FastPathBundle] = OrderedDict()
# Parallel size index (bytes per memo key) so eviction is O(1) per pop without
# re-walking arrays; kept in lockstep with _INPROC_MEMO on every insert/evict.
_INPROC_SIZES: OrderedDict[tuple[str, str], int] = OrderedDict()


def _inproc_enabled() -> bool:
    return os.environ.get("HLBT_INPROC_BUNDLE_MEMO", "0") not in ("0", "", "false", "False")


def _inproc_max() -> int:
    """Secondary entry-count backstop. Lowered from 512 (meaningless for ~130 MB
    objects) to 32 — at ~130 MB/bundle that is ~4 GiB, matching the default byte
    budget; the BYTE bound (``_inproc_max_bytes``) is the real limit."""
    return int(os.environ.get("HLBT_INPROC_BUNDLE_MEMO_MAX", "32"))


def _inproc_max_bytes() -> int:
    """Primary per-process byte budget for the in-proc bundle memo.

    Mirrors the disk cache's ``_cache_max_bytes`` style:
      * ``HLBT_INPROC_BUNDLE_MEMO_MAX_BYTES`` — exact per-process budget (used
        verbatim, NOT divided by the worker count).
      * else ``HLBT_INPROC_BUNDLE_MEMO_MAX_GB`` (default 4 GiB) is the TOTAL
        budget across all spawn workers, divided by
        ``HLBT_INPROC_BUNDLE_MEMO_WORKERS`` (set by ``tune`` before the
        ProcessPoolExecutor spawns; default 1) so the AGGREGATE stays under the
        total regardless of ``--workers``. At 4 GiB / 12 workers that is
        ~340 MiB/worker ≈ 2-3 bundles — enough to win the sweep memo (the same
        question is replayed across many param cells) without the SHR-71 OOM.
    """
    b = os.environ.get("HLBT_INPROC_BUNDLE_MEMO_MAX_BYTES")
    if b is not None:
        return int(b)
    gb = float(os.environ.get("HLBT_INPROC_BUNDLE_MEMO_MAX_GB", "4"))
    total = int(gb * 1024**3)
    workers = max(1, int(os.environ.get("HLBT_INPROC_BUNDLE_MEMO_WORKERS", "1")))
    return total // workers


def _bundle_nbytes(b: FastPathBundle) -> int:
    """Estimate a bundle's retained RAM: the numpy arrays dominate, so sum each
    leg's ``events.nbytes + book_ts.nbytes`` plus a small constant per ref/settle
    event for the Python object overhead."""
    total = 0
    for la in b.leg_arrays.values():
        total += (
            int(la.events.nbytes) + int(la.book_ts.nbytes) + int(la.snap_best_ask.nbytes) + int(la.snap_best_bid.nbytes)
        )
    total += (len(b.reference_events) + len(b.settlement_events)) * 64
    return total


def _inproc_clear() -> None:
    _INPROC_MEMO.clear()
    _INPROC_SIZES.clear()


def _inproc_store(key: tuple[str, str], bundle: FastPathBundle) -> None:
    """Insert ``bundle`` under ``key`` (most-recent) and LRU-evict until under
    BOTH the byte budget (primary) and the count cap (secondary). The
    just-inserted key is never evicted, so a bundle larger than the whole budget
    still hits once before the next insert reclaims it."""
    _INPROC_MEMO[key] = bundle
    _INPROC_MEMO.move_to_end(key)
    _INPROC_SIZES[key] = _bundle_nbytes(bundle)
    _INPROC_SIZES.move_to_end(key)
    max_bytes = _inproc_max_bytes()
    max_count = _inproc_max()
    retained = sum(_INPROC_SIZES.values())
    while _INPROC_MEMO and (retained > max_bytes or len(_INPROC_MEMO) > max_count):
        oldest = next(iter(_INPROC_MEMO))
        if oldest == key:
            break  # never evict the entry we just inserted
        _INPROC_MEMO.pop(oldest, None)
        retained -= _INPROC_SIZES.pop(oldest, 0)


def inproc_lookup(question_id: str, config_sig: str = "", *, force_rebuild: bool = False) -> FastPathBundle | None:
    """Peek the process memo for (question_id, config_sig) without touching disk.

    Returns the memoized bundle if the memo is enabled and populated, else None.
    Data sources call this *before* computing source_files so a memo hit skips
    the source-file glob, not just the npz inflate inside ``cached_bundle``.
    """
    if force_rebuild or not _inproc_enabled() or os.environ.get("HLBT_REBUILD_CACHE"):
        return None
    key = (question_id, config_sig)
    bundle = _INPROC_MEMO.get(key)
    if bundle is not None:
        _INPROC_MEMO.move_to_end(key)
    return bundle


def cached_bundle(
    cache_dir: Path,
    question_id: str,
    source_files: list[Path],
    build_fn: Callable[[], FastPathBundle],
    *,
    force_rebuild: bool = False,
    config_sig: str = "",
) -> FastPathBundle:
    """Return a FastPathBundle, loading from disk cache if valid.

    When caching is disabled (the default — see ``caching_enabled``), this just
    calls ``build_fn()`` and returns, touching no disk. Otherwise: on cache miss
    (no file, mtime/size change, corrupt file, or force_rebuild), calls
    ``build_fn()`` and attempts to persist the result. Write failures are logged
    and silently skipped — the returned bundle is always correct.

    When ``HLBT_INPROC_BUNDLE_MEMO`` is set, an in-process LRU keyed on
    (question_id, config_sig) short-circuits both the disk cache and build on a
    repeat call within the process (the tune-sweep fast path).
    """
    memo_on = _inproc_enabled()
    rebuild = force_rebuild or bool(os.environ.get("HLBT_REBUILD_CACHE"))
    memo_key = (question_id, config_sig)
    if memo_on and not rebuild and memo_key in _INPROC_MEMO:
        _INPROC_MEMO.move_to_end(memo_key)
        return _INPROC_MEMO[memo_key]

    bundle = _cached_bundle_disk(
        cache_dir,
        question_id,
        source_files,
        build_fn,
        force_rebuild=force_rebuild,
        config_sig=config_sig,
    )

    if memo_on:
        _inproc_store(memo_key, bundle)
    return bundle


def _cached_bundle_disk(
    cache_dir: Path,
    question_id: str,
    source_files: list[Path],
    build_fn: Callable[[], FastPathBundle],
    *,
    force_rebuild: bool = False,
    config_sig: str = "",
) -> FastPathBundle:
    if not caching_enabled():
        return build_fn()
    # HLBT_REBUILD_CACHE=1 forces a rebuild process-wide; set by the CLI's
    # --rebuild-cache flag and inherited by spawn workers via the environment.
    force_rebuild = force_rebuild or bool(os.environ.get("HLBT_REBUILD_CACHE"))
    cache = _get_cache(cache_dir)
    key = cache_key(question_id, source_files, config_sig)
    if not force_rebuild:
        try:
            cached = cache.get(key, default=_MISS)
        except Exception as e:  # corruption / unreadable value → rebuild
            log.warning("event-array cache load failed (%s); rebuilding", e)
            cached = _MISS
        if cached is not _MISS:
            return cached
    bundle = build_fn()
    try:
        cache.set(key, bundle)  # diskcache: atomic write + LRU/size-cap eviction
    except Exception as e:
        log.warning("event-array cache write failed (%s); continuing uncached", e)
    return bundle


__all__ = [
    "cache_key",
    "cached_bundle",
]
