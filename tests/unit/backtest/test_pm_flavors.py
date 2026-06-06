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
