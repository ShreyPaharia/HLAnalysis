from __future__ import annotations

from hlanalysis.engine.config import GlobalRiskConfig


def _base_global(**over):
    base = dict(
        max_total_inventory_usd=1000, max_concurrent_positions=5,
        daily_loss_cap_usd=100, max_strike_distance_pct=5,
        min_recent_volume_usd=0, stale_data_halt_seconds=30,
        reconcile_interval_seconds=15,
    )
    base.update(over)
    return GlobalRiskConfig(**base)


def test_scan_cadence_defaults_preserve_1hz():
    g = _base_global()
    assert g.scan_min_interval_seconds == 1.0
    assert g.scan_max_interval_seconds == 1.0
    assert g.stop_loss_loop_enabled is False


def test_scan_cadence_overridable():
    g = _base_global(scan_min_interval_seconds=0.05, scan_max_interval_seconds=1.0,
                     stop_loss_loop_enabled=True)
    assert g.scan_min_interval_seconds == 0.05
    assert g.stop_loss_loop_enabled is True
