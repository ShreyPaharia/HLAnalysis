from scripts.calibrate_pm_liquidity import calibrate_from_snapshots


def test_calibrate_buckets_median():
    rows = [
        {"p": 0.88, "half_spread": 0.005, "depth": 100.0},
        {"p": 0.88, "half_spread": 0.007, "depth": 200.0},
        {"p": 0.30, "half_spread": 0.02, "depth": 50.0},
    ]
    prof = calibrate_from_snapshots(rows, bucket_width=0.1)
    assert abs(prof["half_spread"][8] - 0.006) < 1e-9
    assert abs(prof["depth"][8] - 150.0) < 1e-9
    assert prof["half_spread"][0] is None
    assert prof["global_depth"] > 0
