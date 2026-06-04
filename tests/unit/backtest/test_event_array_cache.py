# tests/unit/backtest/test_event_array_cache.py
from __future__ import annotations
import numpy as np
import pytest
from hlanalysis.backtest.data._event_array_cache import cached_bundle, cache_key
from hlanalysis.backtest.data._fastpath_core import FastPathBundle, LegArrays, event_dtype


@pytest.fixture(autouse=True)
def _enable_caching(monkeypatch):
    # Caching is opt-in (default OFF); these tests exercise the cache itself.
    monkeypatch.setenv("HLBT_CACHE_EVENT_ARRAYS", "1")


def _bundle():
    arr = np.zeros(2, dtype=event_dtype)
    return FastPathBundle(
        leg_arrays={"#0": LegArrays(events=arr, book_ts=np.array([1, 2], dtype=np.int64))},
        reference_events=[], settlement_events=[],
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
    # Corrupt every cached npz.
    for f in cdir.glob("*.npz"):
        f.write_bytes(b"not-an-npz")
    cached_bundle(cdir, "q1", [src], build)
    assert calls["n"] == 2  # corruption treated as miss
