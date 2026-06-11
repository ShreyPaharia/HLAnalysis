# tests/unit/backtest/test_event_array_cache.py
from __future__ import annotations
import numpy as np
import pytest
import hlanalysis.backtest.data._event_array_cache as _cache_mod
from hlanalysis.backtest.data._event_array_cache import (
    cached_bundle, cache_key, caching_enabled,
)
from hlanalysis.backtest.data._fastpath_core import FastPathBundle, LegArrays, event_dtype
from hlanalysis.backtest.core.events import ReferenceEvent, SettlementEvent, TradeEvent


@pytest.fixture(autouse=True)
def _enable_caching(monkeypatch):
    # Caching is default-ON in prod, but the global conftest fixture forces it
    # OFF for hermeticity; these tests exercise the cache itself so re-enable.
    monkeypatch.setenv("HLBT_CACHE_EVENT_ARRAYS", "1")
    monkeypatch.delenv("HLBT_NO_CACHE", raising=False)
    # The process-level bundle memo is a module global; clear it around every
    # test so leftover entries can't leak across tests.
    _cache_mod._inproc_clear()
    yield
    _cache_mod._inproc_clear()
    # diskcache.Cache objects are reused via a module-global registry; close and
    # clear them so SQLite handles don't leak across tests (each test uses a
    # fresh tmp_path, so there is never legitimate cross-test reuse).
    for _c in _cache_mod._CACHES.values():
        try:
            _c.close()
        except Exception:
            pass
    _cache_mod._CACHES.clear()


def _val_files(cache_dir):
    """diskcache stores each value as a ``*.val`` file under a ``v{N}`` shard."""
    return list(cache_dir.rglob("*.val"))


def _bundle():
    arr = np.zeros(2, dtype=event_dtype)
    return FastPathBundle(
        leg_arrays={"#0": LegArrays(events=arr, book_ts=np.array([1, 2], dtype=np.int64))},
        reference_events=[], settlement_events=[],
    )


def _realistic_bundle(n: int = 40_000) -> FastPathBundle:
    """A leg resembling real book depth: monotone ns timestamps, ~40 price
    levels, low-card event flags, mostly-zero id columns; fewer book_ts than
    events. Exercises the delta-encode/round-trip path (zeros would not)."""
    rng = np.random.default_rng(0)
    ev = np.zeros(n, dtype=event_dtype)
    ts0 = 1_700_000_000_000_000_000
    ev["exch_ts"] = ts0 + np.cumsum(rng.integers(1, 5_000_000, n))
    ev["local_ts"] = ev["exch_ts"] + rng.integers(0, 1_000_000, n)
    ev["px"] = rng.choice(40000.0 + np.arange(40) * 5.0, n)
    ev["qty"] = rng.choice(np.arange(1, 50) * 0.01, n)
    ev["ev"] = rng.choice([1, 2, 4], n).astype(ev["ev"].dtype)
    bts = np.ascontiguousarray(ev["exch_ts"][::3])  # monotone, shorter than events
    return FastPathBundle(
        leg_arrays={"#0": LegArrays(events=ev, book_ts=bts)},
        reference_events=[
            ReferenceEvent(ts_ns=ts0 + i, symbol="BTC", high=1.0, low=0.5, close=0.7, open=0.6)
            for i in range(5)
        ],
        settlement_events=[
            SettlementEvent(ts_ns=ts0 + 9, question_idx=0, outcome="up", symbol="@30"),
        ],
    )


def test_miss_then_hit(tmp_path):
    src = tmp_path / "a.parquet"; src.write_bytes(b"x" * 10)
    calls = {"n": 0}
    def build():
        calls["n"] += 1
        return _bundle()
    b1 = cached_bundle(tmp_path / "cache", "q1", [src], build)
    b2 = cached_bundle(tmp_path / "cache", "q1", [src], build)
    assert calls["n"] == 1  # second call is a hit
    assert np.array_equal(b1.leg_arrays["#0"].events, b2.leg_arrays["#0"].events)
    assert np.array_equal(b1.leg_arrays["#0"].book_ts, b2.leg_arrays["#0"].book_ts)


def test_config_sig_differentiates_key():
    """A different config_sig (e.g. dt=5 vs dt=60) MUST produce a different key,
    so a bundle built at one dt can never be served for a request at another."""
    src = []
    assert cache_key("q1", src, "rrs=5000000000") != cache_key("q1", src, "rrs=60000000000")
    assert cache_key("q1", src, "") != cache_key("q1", src, "rrs=5000000000")


def test_config_sig_change_forces_rebuild(tmp_path):
    """Two requests for the same question + files but different config_sig must
    NOT share a cached bundle (the dt=60-bundle-for-dt=5-request footgun)."""
    src = tmp_path / "a.parquet"; src.write_bytes(b"x" * 10)
    calls = {"n": 0}
    def build():
        calls["n"] += 1
        return _bundle()
    cached_bundle(tmp_path / "cache", "q1", [src], build, config_sig="rrs=60000000000")
    cached_bundle(tmp_path / "cache", "q1", [src], build, config_sig="rrs=5000000000")
    assert calls["n"] == 2  # different dt -> rebuilt, not aliased


def test_mtime_change_invalidates(tmp_path):
    src = tmp_path / "a.parquet"; src.write_bytes(b"x" * 10)
    calls = {"n": 0}
    def build():
        calls["n"] += 1; return _bundle()
    cached_bundle(tmp_path / "cache", "q1", [src], build)
    src.write_bytes(b"y" * 20)  # size + mtime change
    cached_bundle(tmp_path / "cache", "q1", [src], build)
    assert calls["n"] == 2  # rebuilt


# --- opt-in process-level bundle memo (the tune-sweep win) ------------------


def test_inproc_bundle_memo_skips_rebuild_when_enabled(tmp_path, monkeypatch):
    """With HLBT_INPROC_BUNDLE_MEMO=1, a repeat call for the same
    (question_id, config_sig) within the process returns the *same* bundle
    object without re-stat (cache_key) or npz re-load — the tune-sweep fast
    path where one question is replayed across many param cells."""
    monkeypatch.setenv("HLBT_INPROC_BUNDLE_MEMO", "1")
    qid = f"q-{tmp_path.name}"
    src = tmp_path / "a.parquet"; src.write_bytes(b"x" * 10)
    calls = {"n": 0}
    def build():
        calls["n"] += 1; return _bundle()
    cdir = tmp_path / "cache"
    b1 = cached_bundle(cdir, qid, [src], build, config_sig="c")
    b2 = cached_bundle(cdir, qid, [src], build, config_sig="c")
    assert calls["n"] == 1
    assert b1 is b2  # served from the process memo, not a fresh disk load


def test_inproc_bundle_memo_off_by_default_loads_fresh(tmp_path):
    """Default-off: behaviour is unchanged — each call returns a freshly loaded
    bundle (distinct objects), preserving the mtime-invalidation contract."""
    qid = f"q-{tmp_path.name}"
    src = tmp_path / "a.parquet"; src.write_bytes(b"x" * 10)
    cdir = tmp_path / "cache"
    b1 = cached_bundle(cdir, qid, [src], _bundle, config_sig="c")
    b2 = cached_bundle(cdir, qid, [src], _bundle, config_sig="c")
    assert b1 is not b2


def test_inproc_bundle_memo_keys_on_config_sig(tmp_path, monkeypatch):
    """Different config_sig (e.g. dt=5 vs dt=60) must NOT share a memo entry."""
    monkeypatch.setenv("HLBT_INPROC_BUNDLE_MEMO", "1")
    qid = f"q-{tmp_path.name}"
    src = tmp_path / "a.parquet"; src.write_bytes(b"x" * 10)
    cdir = tmp_path / "cache"
    b1 = cached_bundle(cdir, qid, [src], _bundle, config_sig="dt5")
    b2 = cached_bundle(cdir, qid, [src], _bundle, config_sig="dt60")
    assert b1 is not b2


def test_build_version_in_key(tmp_path, monkeypatch):
    src = tmp_path / "a.parquet"; src.write_bytes(b"x" * 10)
    import hlanalysis.backtest.data._event_array_cache as m
    k1 = cache_key("q1", [src])
    monkeypatch.setattr(m, "_BUILD_VERSION", 999)
    k2 = cache_key("q1", [src])
    assert k1 != k2


def test_corrupt_cache_file_rebuilds(tmp_path):
    src = tmp_path / "a.parquet"; src.write_bytes(b"x" * 10)
    cdir = tmp_path / "cache"
    calls = {"n": 0}
    def build():
        calls["n"] += 1; return _bundle()
    cached_bundle(cdir, "q1", [src], build)
    # Corrupt every stored value file.
    for f in _val_files(cdir):
        f.write_bytes(b"not-an-npz")
    cached_bundle(cdir, "q1", [src], build)
    assert calls["n"] == 2  # corruption treated as miss


def test_saved_bundle_is_compressed(tmp_path):
    """A bundle dominated by a large zero array must serialize far below its raw
    nbytes — i.e. the .npz is compressed, not stored raw (the disk-blowup fix)."""
    n = 50_000
    big = np.zeros(n, dtype=event_dtype)
    bundle = FastPathBundle(
        leg_arrays={"#0": LegArrays(events=big, book_ts=np.zeros(n, dtype=np.int64))},
        reference_events=[], settlement_events=[],
    )
    src = tmp_path / "a.parquet"; src.write_bytes(b"x")
    cdir = tmp_path / "cache"
    cached_bundle(cdir, "q1", [src], lambda: bundle)
    npz = next(iter(_val_files(cdir)))
    raw = big.nbytes + n * 8
    assert npz.stat().st_size < raw * 0.1  # compresses to well under 10% of raw


def test_stale_build_version_entries_evicted(tmp_path, monkeypatch):
    """Entries from a superseded BUILD_VERSION are orphaned (unreachable) — a
    later cache op must delete them so the dir doesn't grow without bound. Each
    version lives in its own ``v{N}`` shard; opening a new version prunes the
    stale shard wholesale."""
    import hlanalysis.backtest.data._event_array_cache as m
    cdir = tmp_path / "cache"
    src = tmp_path / "a.parquet"; src.write_bytes(b"x" * 10)
    monkeypatch.setattr(m, "_BUILD_VERSION", 1)
    cached_bundle(cdir, "q1", [src], _bundle)
    assert (cdir / "v1").is_dir() and _val_files(cdir / "v1")  # written under v1
    monkeypatch.setattr(m, "_BUILD_VERSION", 2)
    cached_bundle(cdir, "q2", [src], _bundle)
    assert not (cdir / "v1").exists()  # stale v1 shard evicted
    assert (cdir / "v2").is_dir() and _val_files(cdir / "v2")


def test_config_variants_under_same_version_kept(tmp_path):
    """Different config_sig (e.g. dt=5 vs dt=60) under the SAME BUILD_VERSION are
    both valid — eviction must NOT touch them."""
    cdir = tmp_path / "cache"
    src = tmp_path / "a.parquet"; src.write_bytes(b"x" * 10)
    cached_bundle(cdir, "q1", [src], _bundle, config_sig="rrs=5")
    cached_bundle(cdir, "q1", [src], _bundle, config_sig="rrs=60")
    assert len(_val_files(cdir)) == 2  # both kept


def test_roundtrip_preserves_realistic_data(tmp_path):
    """Column-split + delta-encoded storage must reconstruct the bundle exactly:
    every struct field, book_ts, and the ref/settle events."""
    b = _realistic_bundle()
    cdir = tmp_path / "cache"; src = tmp_path / "a"; src.write_bytes(b"x")
    cached_bundle(cdir, "q", [src], lambda: b)  # cold: writes
    def _no_build():
        raise AssertionError("should have hit the cache, not rebuilt")
    got = cached_bundle(cdir, "q", [src], _no_build)  # warm: loads from disk
    la0, la1 = b.leg_arrays["#0"], got.leg_arrays["#0"]
    assert la1.events.dtype == event_dtype
    assert la0.events.tobytes() == la1.events.tobytes()  # every field exact
    assert np.array_equal(la0.book_ts, la1.book_ts)
    assert [(r.ts_ns, r.symbol, r.high, r.low, r.close, r.open) for r in got.reference_events] \
        == [(r.ts_ns, r.symbol, r.high, r.low, r.close, r.open) for r in b.reference_events]
    assert [(s.ts_ns, s.question_idx, s.outcome, s.symbol) for s in got.settlement_events] \
        == [(s.ts_ns, s.question_idx, s.outcome, s.symbol) for s in b.settlement_events]


def test_npz_disk_serializer_roundtrip(tmp_path):
    """The custom diskcache ``Disk`` (the npz column-split serializer plugged
    into diskcache) must round-trip a bundle exactly: ``fetch(store(x)) == x``
    for every event-array field, book_ts, and the ref/settle event lists."""
    import diskcache
    from hlanalysis.backtest.data._event_array_cache import _NpzDisk

    b = _realistic_bundle()
    with diskcache.Cache(str(tmp_path / "dc"), disk=_NpzDisk) as cache:
        cache["k"] = b  # store -> npz file
        got = cache["k"]  # fetch -> _load
    la0, la1 = b.leg_arrays["#0"], got.leg_arrays["#0"]
    assert la1.events.dtype == event_dtype
    assert la0.events.tobytes() == la1.events.tobytes()  # every field exact
    assert np.array_equal(la0.book_ts, la1.book_ts)
    assert [(r.ts_ns, r.symbol, r.high, r.low, r.close, r.open) for r in got.reference_events] \
        == [(r.ts_ns, r.symbol, r.high, r.low, r.close, r.open) for r in b.reference_events]
    assert [(s.ts_ns, s.question_idx, s.outcome, s.symbol) for s in got.settlement_events] \
        == [(s.ts_ns, s.question_idx, s.outcome, s.symbol) for s in b.settlement_events]


def _bundle_with_trades():
    """A bundle whose legs carry trade events — the recent_volume_usd inputs."""
    arr = np.zeros(2, dtype=event_dtype)
    return FastPathBundle(
        leg_arrays={
            "#0": LegArrays(events=arr, book_ts=np.array([1, 2], dtype=np.int64)),
            "#1": LegArrays(events=arr.copy(), book_ts=np.array([1, 2], dtype=np.int64)),
        },
        reference_events=[],
        settlement_events=[],
        trade_events_per_leg={
            "#0": [
                TradeEvent(ts_ns=1_000, symbol="#0", side="buy", price=0.97, size=12.0),
                TradeEvent(ts_ns=2_000, symbol="#0", side="sell", price=0.98, size=3.5),
            ],
            "#1": [],
        },
    )


def _trade_tuples(b):
    return {
        sym: [(t.ts_ns, t.symbol, t.side, t.price, t.size) for t in trades]
        for sym, trades in b.trade_events_per_leg.items()
    }


def test_roundtrip_preserves_trade_events(tmp_path):
    """SHR-78 regression: a cache HIT must restore ``trade_events_per_leg`` so
    the recent_volume_usd gate sees the same volume as a fresh build. Pre-fix the
    npz dropped trades → cached runs read 0 volume → 0 trades for any strategy
    with min_recent_volume_usd > 0."""
    b = _bundle_with_trades()
    cdir = tmp_path / "cache"; src = tmp_path / "a"; src.write_bytes(b"x")
    cached_bundle(cdir, "q", [src], lambda: b)  # cold: writes

    def _no_build():
        raise AssertionError("should have hit the cache, not rebuilt")

    got = cached_bundle(cdir, "q", [src], _no_build)  # warm: loads from disk
    assert _trade_tuples(got) == _trade_tuples(b)


def test_npz_disk_serializer_roundtrips_trades(tmp_path):
    """The custom diskcache ``Disk`` must round-trip trade events too."""
    import diskcache
    from hlanalysis.backtest.data._event_array_cache import _NpzDisk

    b = _bundle_with_trades()
    with diskcache.Cache(str(tmp_path / "dc"), disk=_NpzDisk) as cache:
        cache["k"] = b
        got = cache["k"]
    assert _trade_tuples(got) == _trade_tuples(b)


def test_column_split_smaller_than_struct_layout(tmp_path):
    """The cache must store the event arrays column-split (not as the 64-byte
    interleaved struct), so deflate compresses each homogeneous column far
    better than the row-major record layout."""
    import io
    b = _realistic_bundle()
    la = b.leg_arrays["#0"]
    buf = io.BytesIO()
    np.savez_compressed(buf, ev=la.events, bts=la.book_ts)  # old struct layout
    struct_size = buf.getbuffer().nbytes
    cdir = tmp_path / "cache"; src = tmp_path / "a"; src.write_bytes(b"x")
    cached_bundle(cdir, "q", [src], lambda: b)
    got = next(iter(_val_files(cdir))).stat().st_size
    assert got < struct_size * 0.92  # meaningfully smaller than struct layout


def test_caching_default_on_when_unset(monkeypatch):
    """Default is ON: with no cache env vars set, caching is enabled."""
    monkeypatch.delenv("HLBT_CACHE_EVENT_ARRAYS", raising=False)
    monkeypatch.delenv("HLBT_NO_CACHE", raising=False)
    assert caching_enabled() is True


def test_no_cache_env_disables(monkeypatch):
    """HLBT_NO_CACHE (set by --fresh/--no-cache) is a hard off, even if unset
    otherwise-default-on."""
    monkeypatch.delenv("HLBT_CACHE_EVENT_ARRAYS", raising=False)
    monkeypatch.setenv("HLBT_NO_CACHE", "1")
    assert caching_enabled() is False


def test_explicit_zero_disables(monkeypatch):
    monkeypatch.delenv("HLBT_NO_CACHE", raising=False)
    monkeypatch.setenv("HLBT_CACHE_EVENT_ARRAYS", "0")
    assert caching_enabled() is False


def test_concurrent_saves_same_key_dont_corrupt(tmp_path, monkeypatch):
    """Several workers rebuilding the SAME bundle concurrently (the --workers
    cold-cache case) must not corrupt the cache or crash: each writer needs its
    OWN tmp file so they can't truncate each other's bytes, and the atomic rename
    is last-writer-wins rather than ENOENT'ing when another already renamed
    (SHR-71 rebuild storm)."""
    import threading
    import time
    b = _realistic_bundle()
    path = tmp_path / "v3_samekey.npz"

    real_savez = np.savez_compressed

    def slow_savez(file, **kw):
        real_savez(file, **kw)
        time.sleep(0.05)  # widen the write→rename window so writers overlap

    monkeypatch.setattr(np, "savez_compressed", slow_savez)

    errors: list[Exception] = []

    def writer():
        try:
            _cache_mod._save(path, b)
        except Exception as e:  # ENOENT on a collided rename == the bug
            errors.append(e)

    threads = [threading.Thread(target=writer) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent _save raised: {errors}"
    got = _cache_mod._load(path)  # must be a complete, loadable bundle
    assert got.leg_arrays["#0"].events.tobytes() == b.leg_arrays["#0"].events.tobytes()


def test_bundle_nbytes_counts_arrays():
    """The byte estimate sums each leg's events.nbytes + book_ts.nbytes (the RAM
    the memo actually retains), plus a small per-event constant for ref/settle."""
    b = _realistic_bundle()
    la = b.leg_arrays["#0"]
    est = _cache_mod._bundle_nbytes(b)
    assert est >= la.events.nbytes + la.book_ts.nbytes
    # ref/settle lists add a small constant, never less than the array floor.
    assert est >= la.events.nbytes + la.book_ts.nbytes


def test_inproc_memo_evicts_by_bytes(tmp_path, monkeypatch):
    """With a tiny BYTE budget, inserting several realistic-sized bundles keeps
    total retained bytes under the budget (LRU-evicts), while the just-inserted
    key still hits the memo (no rebuild)."""
    monkeypatch.setenv("HLBT_INPROC_BUNDLE_MEMO", "1")
    one = _cache_mod._bundle_nbytes(_realistic_bundle())
    monkeypatch.setenv("HLBT_INPROC_BUNDLE_MEMO_MAX_BYTES", str(int(one * 1.5)))
    cdir = tmp_path / "cache"
    src = tmp_path / "a.parquet"; src.write_bytes(b"x")
    for i in range(3):
        cached_bundle(cdir, f"q{i}", [src], _realistic_bundle, config_sig="c")
    retained = sum(_cache_mod._bundle_nbytes(b) for b in _cache_mod._INPROC_MEMO.values())
    assert retained <= int(one * 1.5)  # byte budget held via LRU eviction
    # The most-recently inserted key is still memoized -> hit without rebuild.
    def _no_build():
        raise AssertionError("just-inserted key should hit the memo, not rebuild")
    got = cached_bundle(cdir, "q2", [src], _no_build, config_sig="c")
    assert got is not None


def test_inproc_budget_is_worker_aware(monkeypatch):
    """The default per-process budget is divided by the worker count so the
    AGGREGATE across N spawn workers stays under the total budget (SHR-71)."""
    monkeypatch.delenv("HLBT_INPROC_BUNDLE_MEMO_MAX_BYTES", raising=False)
    monkeypatch.setenv("HLBT_INPROC_BUNDLE_MEMO_MAX_GB", "4")
    monkeypatch.setenv("HLBT_INPROC_BUNDLE_MEMO_WORKERS", "1")
    solo = _cache_mod._inproc_max_bytes()
    monkeypatch.setenv("HLBT_INPROC_BUNDLE_MEMO_WORKERS", "8")
    par = _cache_mod._inproc_max_bytes()
    assert par < solo  # parallelism shrinks each worker's slice
    assert par == solo // 8


def test_inproc_explicit_max_bytes_overrides_worker_division(monkeypatch):
    """An explicit per-process HLBT_INPROC_BUNDLE_MEMO_MAX_BYTES is exact — it is
    NOT divided by the worker count (the knob the byte-budget tests pin)."""
    monkeypatch.setenv("HLBT_INPROC_BUNDLE_MEMO_MAX_BYTES", "12345")
    monkeypatch.setenv("HLBT_INPROC_BUNDLE_MEMO_WORKERS", "8")
    assert _cache_mod._inproc_max_bytes() == 12345


def test_size_cap_evicts_oldest(tmp_path, monkeypatch):
    """A configured byte cap keeps the cache bounded and evicts the
    least-recently-stored entry first, so it cannot grow without bound under
    default-on. (Exact retained count is left to diskcache's soft cap + batch
    culling; the contract is bounded growth + oldest-first eviction.)"""
    cdir = tmp_path / "cache"; src = tmp_path / "a"; src.write_bytes(b"x")
    cached_bundle(cdir, "q0", [src], _realistic_bundle, config_sig="s0")
    one = next(iter(_val_files(cdir))).stat().st_size
    monkeypatch.setenv("HLBT_CACHE_MAX_BYTES", str(int(one * 2)))  # ~2 entries
    for i in range(1, 8):
        cached_bundle(cdir, f"q{i}", [src], _realistic_bundle, config_sig=f"s{i}")
    assert len(_val_files(cdir)) < 8  # bounded — not all 8 retained
    # The oldest entry (q0) was evicted first → re-requesting it rebuilds.
    calls = {"n": 0}
    def _rebuild():
        calls["n"] += 1
        return _realistic_bundle()
    cached_bundle(cdir, "q0", [src], _rebuild, config_sig="s0")
    assert calls["n"] == 1


# --- SHR-97: reference_events_are_raw_ticks round-trip --------------------


def _bundle_raw_ticks() -> FastPathBundle:
    """A bundle with ``reference_events_are_raw_ticks=True`` (the raw/event path).

    Pre-v7 serialization dropped this flag → cache hit always loaded False →
    runner called apply_reference (bar path) instead of apply_reference_tick →
    inflated σ → 74 trades collapsed to 3 on v31 binary.
    """
    arr = np.zeros(2, dtype=event_dtype)
    ts0 = 1_700_000_000_000_000_000
    return FastPathBundle(
        leg_arrays={"#0": LegArrays(events=arr, book_ts=np.array([ts0], dtype=np.int64))},
        reference_events=[
            ReferenceEvent(ts_ns=ts0 + i * 5_000_000_000, symbol="BTC",
                           high=100.0, low=100.0, close=100.0)
            for i in range(3)
        ],
        settlement_events=[],
        reference_events_are_raw_ticks=True,
    )


def test_roundtrip_preserves_reference_events_are_raw_ticks_true(tmp_path):
    """SHR-97 regression: a cache HIT on the raw-tick path must restore
    ``reference_events_are_raw_ticks=True``, not silently downgrade to False.

    Pre-v7 _save() omitted the ``__ref_raw_ticks__`` scalar → _load() always
    returned False → the runner called apply_reference (bar OHLC path) on
    raw per-tick events, computing a much higher σ → far fewer qualifying
    trades (74 → 3 on v31 binary 2026-06-10)."""
    b = _bundle_raw_ticks()
    assert b.reference_events_are_raw_ticks is True
    cdir = tmp_path / "cache"
    src = tmp_path / "a"
    src.write_bytes(b"x")
    cached_bundle(cdir, "q", [src], lambda: b)  # cold: writes

    def _no_build():
        raise AssertionError("should have hit the cache, not rebuilt")

    got = cached_bundle(cdir, "q", [src], _no_build)  # warm: loads from disk
    assert got.reference_events_are_raw_ticks is True, (
        "cache hit must restore reference_events_are_raw_ticks=True "
        "(pre-v7 bug: flag silently dropped → apply_reference called instead of "
        "apply_reference_tick → σ inflated → 74 trades became 3)"
    )


def test_roundtrip_preserves_reference_events_are_raw_ticks_false(tmp_path):
    """Complementary: the default False value also round-trips correctly."""
    b = _bundle()  # reference_events_are_raw_ticks defaults to False
    assert b.reference_events_are_raw_ticks is False
    cdir = tmp_path / "cache"
    src = tmp_path / "a"
    src.write_bytes(b"x")
    cached_bundle(cdir, "q", [src], lambda: b)

    def _no_build():
        raise AssertionError("should have hit the cache, not rebuilt")

    got = cached_bundle(cdir, "q", [src], _no_build)
    assert got.reference_events_are_raw_ticks is False


def test_npz_disk_serializer_roundtrips_raw_ticks_flag(tmp_path):
    """The custom diskcache ``Disk`` must round-trip reference_events_are_raw_ticks
    via ``_NpzDisk`` (the direct ``_save`` / ``_load`` path used by diskcache)."""
    import diskcache
    from hlanalysis.backtest.data._event_array_cache import _NpzDisk

    b = _bundle_raw_ticks()
    with diskcache.Cache(str(tmp_path / "dc"), disk=_NpzDisk) as cache:
        cache["k"] = b
        got = cache["k"]
    assert got.reference_events_are_raw_ticks is True
