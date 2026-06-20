"""SHR-149: global-silence gate for reference-gap attribution.

Ports engine commit 9bea25f. A reference gap on one symbol while *other* feeds
keep ticking is a calm/illiquid market, not a recording outage — so the gap must
NOT be attributed to data. A gap coinciding with global ingest silence (every
feed quiet) IS a real outage and must be attributed.
"""

from __future__ import annotations

from hlanalysis.research.reconcile.reconcile import (
    attributable_gaps,
    check_reference_coverage,
)

_T0 = 1_718_000_000_000_000_000
_1S_NS = 1_000_000_000


def _ts_reader(ts: list[int]):
    def reader(symbol: str, start_ns: int, end_ns: int, data_root) -> list[int]:
        return ts

    return reader


def _cross_reader(ts: list[int]):
    def reader(start_ns: int, end_ns: int, data_root) -> list[int]:
        return [t for t in ts if start_ns <= t <= end_ns]

    return reader


def _ref_ts_with_gap() -> tuple[list[int], int, int]:
    """~2 s-cadence BTC feed with a single 1200 s hole; return (ts, gap_lo, gap_hi)."""
    ts = [_T0 + i * 2 * _1S_NS for i in range(5)]
    gap_lo = ts[-1]
    gap_hi = gap_lo + 1200 * _1S_NS
    ts.append(gap_hi)
    ts += [ts[-1] + i * 2 * _1S_NS for i in range(1, 5)]
    return ts, gap_lo, gap_hi


class TestGlobalSilenceGate:
    def test_one_feed_gap_other_feeds_ticking_not_attributed(self) -> None:
        """Other feeds tick through the BTC gap → benign, not attributed."""
        ts, gap_lo, gap_hi = _ref_ts_with_gap()
        # ETH/SOL feeds tick every ~3 s right through the BTC hole.
        cross = [gap_lo + i * 3 * _1S_NS for i in range(1, 400) if gap_lo + i * 3 * _1S_NS < gap_hi]

        gaps = check_reference_coverage(
            start_ns=ts[0],
            end_ns=ts[-1],
            data_root=None,
            ts_reader=_ts_reader(ts),
            cross_feed_reader=_cross_reader(cross),
        )
        assert len(gaps) == 1
        assert gaps[0].global_silence is False
        assert attributable_gaps(gaps) == []

    def test_coincident_global_gap_is_attributed(self) -> None:
        """No feed ticks through the BTC gap → real outage, attributed."""
        ts, gap_lo, gap_hi = _ref_ts_with_gap()
        # Cross feeds only tick OUTSIDE the gap (same global outage).
        cross = [t for t in ts if t != gap_hi]

        gaps = check_reference_coverage(
            start_ns=ts[0],
            end_ns=ts[-1],
            data_root=None,
            ts_reader=_ts_reader(ts),
            cross_feed_reader=_cross_reader(cross),
        )
        assert len(gaps) == 1
        assert gaps[0].global_silence is True
        assert attributable_gaps(gaps) == gaps

    def test_no_cross_reader_defaults_to_attributed(self) -> None:
        """Back-compat: without a cross-feed reader every gap is treated as outage."""
        ts, _, _ = _ref_ts_with_gap()
        gaps = check_reference_coverage(
            start_ns=ts[0],
            end_ns=ts[-1],
            data_root=None,
            ts_reader=_ts_reader(ts),
        )
        assert len(gaps) == 1
        assert gaps[0].global_silence is True
        assert attributable_gaps(gaps) == gaps
