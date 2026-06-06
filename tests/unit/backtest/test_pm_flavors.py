from pathlib import Path
from hlanalysis.backtest.data.polymarket import PolymarketDataSource


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


from hlanalysis.backtest.core.source_config import PM_FLAVORS, SourceConfig


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
