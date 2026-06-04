"""Disk cache for built FastPathBundles.

Key = sha256(question_id + sorted[(path,size,mtime_ns)] + BUILD_VERSION). Any
re-record (size/mtime change) or assembly-logic change (BUILD_VERSION bump in
_fastpath_core) misses → rebuild. Stat-only keying (never reads file bytes) so
it preserves the speedup. Stored as one .npz per key; load failure = miss.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Callable

import numpy as np

from ._fastpath_core import BUILD_VERSION as _BUILD_VERSION
from ._fastpath_core import FastPathBundle, LegArrays
from ..core.events import ReferenceEvent, SettlementEvent

log = logging.getLogger(__name__)


def cache_key(question_id: str, source_files: list[Path]) -> str:
    """Compute a cache key from question_id, source file metadata, and BUILD_VERSION.

    Uses the module-level ``_BUILD_VERSION`` so ``monkeypatch.setattr`` in tests
    can override it and verify version changes produce different keys.
    """
    import hlanalysis.backtest.data._event_array_cache as _self
    h = hashlib.sha256()
    h.update(question_id.encode())
    h.update(str(_self._BUILD_VERSION).encode())
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
        payload[f"ev_{i}"] = la.events
        payload[f"bts_{i}"] = la.book_ts
    # np.savez appends ".npz" to the path if not already present.
    # Use a tmp file that already ends in ".npz" so the written file
    # matches the tmp variable name, then atomically rename to final path.
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **payload)
    tmp.replace(path)


def _load(path: Path) -> FastPathBundle:
    """Deserialize a FastPathBundle from a .npz file."""
    z = np.load(path, allow_pickle=True)
    legs = list(z["__legs__"])
    leg_arrays = {
        sym: LegArrays(events=z[f"ev_{i}"], book_ts=z[f"bts_{i}"])
        for i, sym in enumerate(legs)
    }
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


def cached_bundle(
    cache_dir: Path,
    question_id: str,
    source_files: list[Path],
    build_fn: Callable[[], FastPathBundle],
    *,
    force_rebuild: bool = False,
) -> FastPathBundle:
    """Return a FastPathBundle, loading from disk cache if valid.

    On cache miss (no file, mtime/size change, corrupt file, or force_rebuild),
    calls ``build_fn()`` and attempts to persist the result. Write failures are
    logged and silently skipped — the returned bundle is always correct.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{cache_key(question_id, source_files)}.npz"
    if path.exists() and not force_rebuild:
        try:
            return _load(path)
        except Exception as e:  # corruption / version skew → rebuild
            log.warning("event-array cache load failed (%s); rebuilding", e)
    bundle = build_fn()
    try:
        _save(path, bundle)
    except Exception as e:
        log.warning("event-array cache write failed (%s); continuing uncached", e)
    return bundle


__all__ = [
    "cache_key",
    "cached_bundle",
]
