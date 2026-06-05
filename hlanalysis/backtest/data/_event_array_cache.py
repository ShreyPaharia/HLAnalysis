"""Disk cache for built FastPathBundles.

Key = sha256(question_id + sorted[(path,size,mtime_ns)] + BUILD_VERSION). Any
re-record (size/mtime change) or assembly-logic change (BUILD_VERSION bump in
_fastpath_core) misses → rebuild. Stat-only keying (never reads file bytes) so
it preserves the speedup. Stored as one .npz per key; load failure = miss.
"""
from __future__ import annotations

import hashlib
import logging
import os
from collections import OrderedDict
from pathlib import Path
from typing import Callable

import numpy as np

from ._fastpath_core import BUILD_VERSION as _BUILD_VERSION
from ._fastpath_core import FastPathBundle, LegArrays, event_dtype
from ..core.events import ReferenceEvent, SettlementEvent

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
            [
                (e.ts_ns, e.symbol, e.high, e.low, e.close, e.open)
                for e in b.reference_events
            ],
            dtype=object,
        ),
        "__settle__": np.array(
            [
                (e.ts_ns, e.question_idx, e.outcome, e.symbol)
                for e in b.settlement_events
            ],
            dtype=object,
        ),
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
    # np.savez_compressed appends ".npz" to the path if not already present.
    # Use a tmp file that already ends in ".npz" so the written file
    # matches the tmp variable name, then atomically rename to final path.
    # Compressed: assembled book bundles are large but highly repetitive
    # (zero-padded fields, monotone timestamps), so deflate cuts the on-disk
    # corpus by ~an order of magnitude — the disk-blowup mitigation.
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **payload)
    tmp.replace(path)


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
        leg_arrays[sym] = LegArrays(events=ev, book_ts=_undelta(z[f"bts_{i}"]))
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
    return FastPathBundle(
        leg_arrays=leg_arrays,
        reference_events=ref,
        settlement_events=settle,
    )


_pruned: set[tuple[str, int]] = set()


def _version_prefix() -> str:
    """Filename prefix encoding the current BUILD_VERSION (e.g. ``v2_``).

    Folding the version into the *filename* (not just the key hash) lets eviction
    identify stale-version orphans with a cheap glob — no need to open each .npz.
    A config_sig change keeps the same prefix (different hash), so legitimate
    config variants (dt=5 vs dt=60) are NOT mistaken for orphans.
    """
    import hlanalysis.backtest.data._event_array_cache as _self
    return f"v{_self._BUILD_VERSION}_"


def _prune_stale_versions(cache_dir: Path) -> None:
    """Delete cached .npz files from a superseded BUILD_VERSION.

    Such entries are unreachable (the key hash and filename prefix both fold in
    BUILD_VERSION) but were never removed, so the dir grew without bound across
    assembly-logic changes. Runs at most once per (dir, version) per process.
    Best-effort: a concurrent spawn worker may unlink the same file first.
    """
    import hlanalysis.backtest.data._event_array_cache as _self
    guard = (str(cache_dir), _self._BUILD_VERSION)
    if guard in _pruned:
        return
    _pruned.add(guard)
    prefix = _version_prefix()
    for f in cache_dir.glob("*.npz"):
        if not f.name.startswith(prefix):
            try:
                f.unlink()
            except OSError:
                pass


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
    return int(gb * 1024 ** 3)


def _enforce_size_cap(cache_dir: Path) -> None:
    """Evict least-recently-written entries until the dir is under the cap.

    LRU by mtime. Best-effort (a concurrent worker may unlink first). Keeps the
    cache from growing without bound now that it is default-on.
    """
    max_bytes = _cache_max_bytes()
    files = list(cache_dir.glob("*.npz"))
    sized = []
    total = 0
    for f in files:
        try:
            st = f.stat()
        except OSError:
            continue
        sized.append((st.st_mtime_ns, st.st_size, f))
        total += st.st_size
    sized.sort()  # oldest first
    for _mtime, size, f in sized:
        if total <= max_bytes:
            break
        try:
            f.unlink()
            total -= size
        except OSError:
            pass


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
_INPROC_MEMO: "OrderedDict[tuple[str, str], FastPathBundle]" = OrderedDict()
# Parallel size index (bytes per memo key) so eviction is O(1) per pop without
# re-walking arrays; kept in lockstep with _INPROC_MEMO on every insert/evict.
_INPROC_SIZES: "OrderedDict[tuple[str, str], int]" = OrderedDict()


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
    total = int(gb * 1024 ** 3)
    workers = max(1, int(os.environ.get("HLBT_INPROC_BUNDLE_MEMO_WORKERS", "1")))
    return total // workers


def _bundle_nbytes(b: FastPathBundle) -> int:
    """Estimate a bundle's retained RAM: the numpy arrays dominate, so sum each
    leg's ``events.nbytes + book_ts.nbytes`` plus a small constant per ref/settle
    event for the Python object overhead."""
    total = 0
    for la in b.leg_arrays.values():
        total += int(la.events.nbytes) + int(la.book_ts.nbytes)
    total += (len(b.reference_events) + len(b.settlement_events)) * 64
    return total


def _inproc_clear() -> None:
    _INPROC_MEMO.clear()
    _INPROC_SIZES.clear()


def _inproc_store(key: "tuple[str, str]", bundle: FastPathBundle) -> None:
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


def inproc_lookup(
    question_id: str, config_sig: str = "", *, force_rebuild: bool = False
) -> "FastPathBundle | None":
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
        cache_dir, question_id, source_files, build_fn,
        force_rebuild=force_rebuild, config_sig=config_sig,
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
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    _prune_stale_versions(cache_dir)
    key = cache_key(question_id, source_files, config_sig)
    path = cache_dir / f"{_version_prefix()}{key}.npz"
    if path.exists() and not force_rebuild:
        try:
            return _load(path)
        except Exception as e:  # corruption / version skew → rebuild
            log.warning("event-array cache load failed (%s); rebuilding", e)
    bundle = build_fn()
    try:
        _save(path, bundle)
        _enforce_size_cap(cache_dir)
    except Exception as e:
        log.warning("event-array cache write failed (%s); continuing uncached", e)
    return bundle


__all__ = [
    "cache_key",
    "cached_bundle",
]
