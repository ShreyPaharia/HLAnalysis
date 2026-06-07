# tests/unit/engine/test_config_pm_bucket_slots.py
"""Config tests for the PM multi-strike bucket routing.

Routing (operator decision 2026-06-08):
  - BTC multi-strike is FOLDED onto the existing live `v31_pm` slot/wallet
    (same BTCUSDT_SPOT reference) as a priceBucket allowlist entry + a per-class
    theta_override carrying PR #12's tuned cell (fav0.75/eb0.02/vlb1800/esd0.5).
  - ETH multi-strike is a dedicated paper slot `v31_pm_eth_ms` (needs the new
    ETHUSDT_SPOT reference, so it must be its own slot/wallet). Bucket-only, so
    the base theta block IS its effective config.

esd=0.5 is below the binary baseline (1.0); allowed via the documented
(v31_pm, priceBucket) exemption in test_bucket_override_risk_gates_not_below_binary.
"""
from __future__ import annotations

from pathlib import Path

from hlanalysis.engine.config import load_strategies_config
from hlanalysis.engine.config_builders import build_theta_harvester_configs_by_class
from hlanalysis.engine.runtime import build_theta_harvester_config

_TUNED = {"favorite_threshold": 0.75, "edge_buffer": 0.02,
          "vol_lookback_seconds": 1800, "exit_safety_d": 0.5}


def _by_alias():
    cfg = load_strategies_config(Path("config/strategy.yaml"))
    return {s.account_alias: s for s in cfg.strategies}


def test_btc_multistrike_folded_onto_v31_pm():
    by_alias = _by_alias()
    assert "v31_pm" in by_alias
    # No dedicated BTC slot — BTC rides v31_pm.
    assert "v31_pm_btc_ms" not in by_alias
    v31_pm = by_alias["v31_pm"]
    assert v31_pm.reference_symbol == "BTCUSDT_SPOT"
    # v31_pm stays LIVE; the bucket fold trades live on deploy.
    assert v31_pm.paper_mode is False
    # The slot now allowlists the multistrike bucket series alongside up/down.
    series = {
        (e.match or {}).get("series_slug")
        for e in v31_pm.allowlist
    }
    assert "btc-up-or-down-daily" in series
    assert "btc-multi-strikes-weekly" in series
    # priceBucket override carries the tuned cell; binary base stays protective.
    by_class = build_theta_harvester_configs_by_class(v31_pm)
    assert set(by_class) == {"priceBucket"}
    bucket = by_class["priceBucket"]
    for k, v in _TUNED.items():
        assert getattr(bucket, k) == v, f"v31_pm bucket {k}={getattr(bucket, k)}"
    base = build_theta_harvester_config(v31_pm)
    assert base.favorite_threshold == 0.85   # binary baseline unchanged
    assert base.exit_safety_d == 1.0


def test_eth_multistrike_dedicated_paper_slot():
    by_alias = _by_alias()
    assert "v31_pm_eth_ms" in by_alias
    eth = by_alias["v31_pm_eth_ms"]
    assert eth.paper_mode is True
    assert eth.reference_symbol == "ETHUSDT_SPOT"
    # Bucket-only slot: no per-class override; base theta is the config.
    assert build_theta_harvester_configs_by_class(eth) == {}
    base = build_theta_harvester_config(eth)
    for k, v in _TUNED.items():
        assert getattr(base, k) == v, f"eth base {k}={getattr(base, k)}"
    assert base.vol_estimator == "bipower"
