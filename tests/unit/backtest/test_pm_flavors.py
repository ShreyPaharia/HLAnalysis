from pathlib import Path
from hlanalysis.backtest.data.polymarket import PolymarketDataSource
from hlanalysis.backtest.core.source_config import PM_FLAVORS, SourceConfig


# ---- klines_1s_subdir correctness (ETH vs BTC reference, config-sig isolation) ---


def test_eth_flavors_use_eth_1s_subdir():
    """ETH flavors must carry eth_klines_1s; BTC flavors must carry btc_klines_1s.

    Without this guard a flavor built with klines_1s reference_source reads BTC
    1s bars as the intraday σ input for an ETH market — pinning p_model at 1.000.
    """
    assert PM_FLAVORS["eth_updown"]["klines_1s_subdir"] == "eth_klines_1s"
    assert PM_FLAVORS["eth_multistrike"]["klines_1s_subdir"] == "eth_klines_1s"
    assert PM_FLAVORS["btc_updown"]["klines_1s_subdir"] == "btc_klines_1s"
    assert PM_FLAVORS["btc_multistrike"]["klines_1s_subdir"] == "btc_klines_1s"


def test_datasource_built_from_eth_flavor_reads_eth_1s(tmp_path):
    """SourceConfig.build() for eth_multistrike must wire the ETH 1s subdir and
    reference symbol through to the constructed PolymarketDataSource."""
    cfg = SourceConfig(kind="polymarket", cache_root=str(tmp_path), pm_flavor="eth_multistrike")
    ds = cfg.build()
    assert ds._klines_1s_subdir == "eth_klines_1s"
    assert ds._reference_symbol == "ETH"


def test_bundle_config_sig_includes_kline_subdirs(tmp_path):
    """Two PolymarketDataSource instances that differ only in klines_1s_subdir
    must produce distinct _bundle_config_sig() values so the event-array cache
    never serves a BTC-reference bundle for an ETH-reference request."""
    btc_ds = PolymarketDataSource(
        cache_root=tmp_path,
        klines_subdir="btc_klines",
        klines_1s_subdir="btc_klines_1s",
    )
    eth_ds = PolymarketDataSource(
        cache_root=tmp_path,
        klines_subdir="eth_klines",
        klines_1s_subdir="eth_klines_1s",
    )
    assert btc_ds._bundle_config_sig() != eth_ds._bundle_config_sig()


# ---- original tests below ---------------------------------------------------


def test_bucket_series_slug_is_configurable(tmp_path: Path):
    ds = PolymarketDataSource(
        cache_root=tmp_path,
        reference_symbol="ETH",
        bucket_series_slug="ethereum-multi-strikes-weekly",
    )
    assert ds._bucket_series_slug == "ethereum-multi-strikes-weekly"


def test_bucket_series_slug_defaults_to_btc_weekly(tmp_path: Path):
    ds = PolymarketDataSource(cache_root=tmp_path)
    assert ds._bucket_series_slug == "btc-multi-strikes-weekly"


def test_new_flavors_registered():
    assert PM_FLAVORS["eth_updown"]["reference_symbol"] == "ETH"
    assert PM_FLAVORS["eth_updown"]["series_slug"] == "eth-up-or-down-daily"
    assert PM_FLAVORS["btc_multistrike"]["bucket_series_slug"] == "btc-multi-strikes-weekly"
    assert PM_FLAVORS["eth_multistrike"]["reference_symbol"] == "ETH"
    assert PM_FLAVORS["eth_multistrike"]["bucket_series_slug"] == "ethereum-multi-strikes-weekly"


def test_each_flavor_builds(tmp_path):
    for flavor in ("eth_updown", "btc_multistrike", "eth_multistrike"):
        cfg = SourceConfig(kind="polymarket", cache_root=str(tmp_path), pm_flavor=flavor)
        ds = cfg.build()
        assert ds is not None


import subprocess, sys


def test_cli_run_help_lists_new_flavors():
    out = subprocess.run(
        [sys.executable, "-m", "hlanalysis.backtest.cli", "run", "--help"],
        capture_output=True, text=True,
    )
    assert "eth_updown" in out.stdout
    assert "eth_multistrike" in out.stdout


def test_kline_coverage_includes_bucket_windows(tmp_path, monkeypatch):
    ds = PolymarketDataSource(cache_root=tmp_path, reference_symbol="ETH",
                              klines_subdir="eth_klines",
                              bucket_series_slug="ethereum-multi-strikes-weekly")
    fake_manifest = {
        "evt-slug": {"kind": "bucket",
                     "bucket": {"start_ts_ns": 1_700_000_000_000_000_000,
                                "end_ts_ns": 1_700_600_000_000_000_000}},
    }
    monkeypatch.setattr(ds, "_load_manifest", lambda: fake_manifest)
    monkeypatch.setattr(ds, "_fetch_and_cache_binary", lambda *a, **k: None)
    monkeypatch.setattr(ds, "_fetch_and_cache_bucket", lambda *a, **k: None)
    monkeypatch.setattr(ds, "_write_manifest", lambda m: None)
    monkeypatch.setattr(ds, "discover", lambda **k: [])
    seen = {}
    monkeypatch.setattr(ds, "_ensure_kline_coverage",
                        lambda s, e: seen.update(start=s, end=e))
    ds.fetch_and_cache(start="2023-11-01", end="2023-12-01", kind="bucket")
    assert seen == {"start": 1_700_000_000_000_000_000, "end": 1_700_600_000_000_000_000}
