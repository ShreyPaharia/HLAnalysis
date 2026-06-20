"""SHR-148: λ-adaptive (Poisson) reference-gap threshold.

The flat 60 s inter-tick threshold over-flags on slow/bursty feeds (43 mostly
benign gaps on #1000465). Modelling inter-update gaps as a Poisson process and
flagging only when P(wait > Δ) = e^(−λΔ) is implausibly small auto-adapts to the
feed cadence. The flat path must remain available and unchanged (back-compat).
"""

from __future__ import annotations

from hlanalysis.research.reconcile.reconcile import check_reference_coverage

_T0 = 1_718_000_000_000_000_000
_1S_NS = 1_000_000_000


def _ts_reader(ts: list[int]):
    def reader(symbol: str, start_ns: int, end_ns: int, data_root) -> list[int]:
        return ts

    return reader


class TestPoissonThreshold:
    def test_slow_feed_no_false_flag(self) -> None:
        """A ~90 s-cadence feed has no improbable gap — Poisson flags none.

        The flat 60 s threshold would flag *every* inter-tick as a gap; the
        Poisson method adapts λ to the cadence and stays quiet.
        """
        # 40 ticks at ~90 s with jitter (75-105 s), no outage.
        ts = [_T0]
        for i in range(40):
            step = 75 + (i % 7) * 5  # 75..105 s, all "normal" for this feed
            ts.append(ts[-1] + step * _1S_NS)

        gaps = check_reference_coverage(
            start_ns=ts[0],
            end_ns=ts[-1],
            data_root=None,
            method="poisson",
            poisson_p=1e-6,
            ts_reader=_ts_reader(ts),
        )
        assert gaps == [], f"slow but regular feed should not flag: {gaps}"

    def test_true_outage_flagged(self) -> None:
        """An implausibly long gap on the same ~90 s feed IS flagged."""
        ts = [_T0]
        for i in range(40):
            step = 75 + (i % 7) * 5
            ts.append(ts[-1] + step * _1S_NS)
        # Inject a 2000 s outage (e^(−2000/90) ≈ 3e-10 ≪ 1e-6).
        outage_start = ts[-1]
        ts.append(outage_start + 2000 * _1S_NS)
        for i in range(5):
            ts.append(ts[-1] + 90 * _1S_NS)

        gaps = check_reference_coverage(
            start_ns=ts[0],
            end_ns=ts[-1],
            data_root=None,
            method="poisson",
            poisson_p=1e-6,
            ts_reader=_ts_reader(ts),
        )
        assert len(gaps) == 1, gaps
        assert abs(gaps[0].gap_seconds - 2000.0) < 1.0
        assert gaps[0].start_ns == outage_start
        # The improbability is carried for the report.
        assert gaps[0].p_value is not None
        assert gaps[0].p_value < 1e-6

    def test_bursty_ticks_not_flagged(self) -> None:
        """Fast bursty ticks with occasional moderate gaps stay quiet."""
        # ~1 s cadence with bursts and occasional 8 s lulls — none improbable.
        ts = [_T0]
        for i in range(60):
            step = 1 if i % 4 else 8  # mostly 1 s, every 4th is 8 s
            ts.append(ts[-1] + step * _1S_NS)
        gaps = check_reference_coverage(
            start_ns=ts[0],
            end_ns=ts[-1],
            data_root=None,
            method="poisson",
            poisson_p=1e-6,
            ts_reader=_ts_reader(ts),
        )
        assert gaps == [], f"bursty-but-live feed should not flag: {gaps}"


class TestFlatBackCompat:
    def test_flat_is_default(self) -> None:
        """The default method stays flat 60 s (unchanged behaviour)."""
        ts = [_T0 + i * 2 * _1S_NS for i in range(5)]
        ts.append(ts[-1] + 120 * _1S_NS)
        ts += [ts[-1] + i * 2 * _1S_NS for i in range(1, 5)]
        gaps = check_reference_coverage(
            start_ns=ts[0],
            end_ns=ts[-1],
            data_root=None,
            ts_reader=_ts_reader(ts),
        )
        assert len(gaps) == 1
        assert abs(gaps[0].gap_seconds - 120.0) < 1.0

    def test_flat_explicit_unchanged(self) -> None:
        ts = [_T0 + i * 2 * _1S_NS for i in range(5)]
        ts.append(ts[-1] + 90 * _1S_NS)
        ts += [ts[-1] + i * 2 * _1S_NS for i in range(1, 5)]
        gaps = check_reference_coverage(
            start_ns=ts[0],
            end_ns=ts[-1],
            data_root=None,
            method="flat",
            gap_threshold_seconds=60.0,
            ts_reader=_ts_reader(ts),
        )
        assert len(gaps) == 1
