# tests/unit/backtest/test_bundle_config_sig.py
"""Coverage guard for the event-array cache's config_sig.

Because caching is default-ON, a bundle-affecting constructor param that is NOT
reflected in ``_bundle_config_sig`` would let a bundle built under one config be
served for a request under another (silent poisoning — the dt=60-for-dt=5 class
that bit twice). These tests assert that varying each such param changes the
signature, so dropping a field from the sig fails loudly.
"""

from __future__ import annotations

from hlanalysis.backtest.data.hl_hip4 import HLHip4DataSource
from hlanalysis.backtest.data.polymarket import PolymarketDataSource


def test_hl_config_sig_changes_with_each_bundle_param(tmp_path):
    base = dict(data_root=tmp_path, ref_event="bbo", ref_source="hl_perp", reference_resample_seconds=60)
    sig = HLHip4DataSource(**base)._bundle_config_sig()
    variants = [
        {**base, "reference_resample_seconds": 5},  # dt
        {**base, "ref_event": "mark"},  # feed kind
        {**base, "ref_source": "binance_perp"},  # reference venue
        {**base, "reference_ticks": "raw"},  # SHR-93: bars vs raw ticks
        {**base, "leg_prune_favorite_threshold": 0.85},  # bucket leg pruning
    ]
    sigs = {HLHip4DataSource(**v)._bundle_config_sig() for v in variants}
    assert sig not in sigs  # each variant differs from baseline
    assert len(sigs) == len(variants)  # ...and from each other


def test_pm_config_sig_changes_with_each_bundle_param(tmp_path):
    base = dict(
        cache_root=tmp_path,
        reference_source="klines",
        reference_resample_seconds=60,
        binance_bbo_product_type="perp",
        book_source="synthetic",
    )
    sig = PolymarketDataSource(**base)._bundle_config_sig()
    variants = [
        {**base, "reference_resample_seconds": 5},  # dt
        {**base, "reference_source": "binance_bbo"},  # reference feed mode
        {**base, "binance_bbo_product_type": "spot"},  # bbo instrument
        {**base, "book_source": "recorded"},  # book fill source
    ]
    sigs = {PolymarketDataSource(**v)._bundle_config_sig() for v in variants}
    assert sig not in sigs
    assert len(sigs) == len(variants)
