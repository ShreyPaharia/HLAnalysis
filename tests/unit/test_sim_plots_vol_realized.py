"""Unit tests for hlanalysis.sim.plots.vol_realized (Task C4).

Test strategy:
- Empty / missing diagnostics_dir → returns None, no file written.
- No markets with valid σ → returns None, no file written.
- 5 synthetic markets: asserts plotted (σ, |Δln S|) values within 1e-9 tolerance.
- Reference line: y = σ·√(1/365.25) at every plotted σ — exact equality.
- Report integration: when plot_vol_realized returns a path, report.md contains
  a '## Vol vs realized' section with a 'vol_realized.html' link.
- v1 path (all-null σ): no point, no crash.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from hlanalysis.sim.plots.vol_realized import (
    _compute_vol_realized_points,
    plot_vol_realized,
    _TAU_1D,
    _TTE_MID_YR,
)


# ---------------------------------------------------------------------------
# Fake market + helpers
# ---------------------------------------------------------------------------

@dataclass
class _FakeMarket:
    condition_id: str
    start_ts_ns: int
    end_ts_ns: int
    resolved_outcome: str = "yes"


_SECS_PER_YEAR = 365.25 * 24.0 * 3600.0


def _make_diagnostics_parquet(
    path: Path,
    rows: list[dict],
) -> None:
    """Write a diagnostics.parquet at *path* matching DIAGNOSTICS_SCHEMA."""
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([
        pa.field("ts_ns",        pa.int64()),
        pa.field("condition_id", pa.string()),
        pa.field("question_idx", pa.int64()),
        pa.field("action",       pa.string()),
        pa.field("reason",       pa.string()),
        pa.field("p_model",      pa.float64()),
        pa.field("edge_yes",     pa.float64()),
        pa.field("edge_no",      pa.float64()),
        pa.field("sigma",        pa.float64()),
        pa.field("tau_yr",       pa.float64()),
        pa.field("ln_sk",        pa.float64()),
        pa.field("ref_price",    pa.float64()),
        pa.field("yes_bid",      pa.float64()),
        pa.field("yes_ask",      pa.float64()),
        pa.field("no_bid",       pa.float64()),
        pa.field("no_ask",       pa.float64()),
    ])
    if not rows:
        table = pa.table(
            {f.name: pa.array([], type=f.type) for f in schema},
            schema=schema,
        )
    else:
        cols: dict = {f.name: [] for f in schema}
        for row in rows:
            for f in schema:
                cols[f.name].append(row.get(f.name))
        arrays = {
            name: pa.array(vals, type=schema.field(name).type)
            for name, vals in cols.items()
        }
        table = pa.table(arrays, schema=schema)
    pq.write_table(table, path)


def _make_klines_json(path: Path, klines: list[dict]) -> None:
    """Write a klines JSON file (list of dicts with ts_ns, open, high, low, close, volume)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(klines))


def _diag_row(
    *,
    condition_id: str,
    ts_ns: int,
    action: str = "hold",
    sigma: Optional[float] = 0.8,
    tau_yr: Optional[float] = None,
) -> dict:
    """Build a minimal diagnostics row dict."""
    return dict(
        ts_ns=ts_ns,
        condition_id=condition_id,
        question_idx=0,
        action=action,
        reason="edge",
        p_model=0.5,
        edge_yes=0.05,
        edge_no=0.05,
        sigma=sigma,
        tau_yr=tau_yr,
        ln_sk=None,
        ref_price=50000.0,
        yes_bid=None,
        yes_ask=None,
        no_bid=None,
        no_ask=None,
    )


def _kline_dict(ts_ns: int, open_: float, close: float) -> dict:
    """Build a minimal kline dict matching Kline fields."""
    return dict(ts_ns=ts_ns, open=open_, high=close, low=open_, close=close, volume=1.0)


# ---------------------------------------------------------------------------
# TC1: Empty / missing diagnostics_dir → returns None, no file
# ---------------------------------------------------------------------------

class TestEmptyOrMissing:
    def test_missing_diagnostics_dir_returns_none(self, tmp_path: Path):
        """diagnostics_dir does not exist → returns None, no file written."""
        diag_dir = tmp_path / "diagnostics"
        klines_dir = tmp_path / "klines"
        klines_dir.mkdir()
        out_path = tmp_path / "vol_realized.html"
        market = _FakeMarket("cid_a", int(1e18), int(2e18))
        result = plot_vol_realized(diag_dir, klines_dir, [market], out_path)
        assert result is None
        assert not out_path.exists()

    def test_empty_diagnostics_dir_returns_none(self, tmp_path: Path):
        """diagnostics_dir exists but has no parquets → returns None."""
        diag_dir = tmp_path / "diagnostics"
        diag_dir.mkdir()
        klines_dir = tmp_path / "klines"
        klines_dir.mkdir()
        out_path = tmp_path / "vol_realized.html"
        market = _FakeMarket("cid_a", int(1e18), int(2e18))
        result = plot_vol_realized(diag_dir, klines_dir, [market], out_path)
        assert result is None
        assert not out_path.exists()

    def test_no_markets_returns_none(self, tmp_path: Path):
        """Empty market list → returns None."""
        diag_dir = tmp_path / "diagnostics"
        diag_dir.mkdir()
        klines_dir = tmp_path / "klines"
        klines_dir.mkdir()
        out_path = tmp_path / "vol_realized.html"
        result = plot_vol_realized(diag_dir, klines_dir, [], out_path)
        assert result is None
        assert not out_path.exists()


# ---------------------------------------------------------------------------
# TC2: No markets with valid σ → returns None, no file
# ---------------------------------------------------------------------------

class TestNoValidSigma:
    def test_v1_fills_all_null_sigma_returns_none(self, tmp_path: Path):
        """v1 path: all sigma values are None → no point, no file."""
        diag_dir = tmp_path / "diagnostics"
        klines_dir = tmp_path / "klines"

        start_ns = int(1_700_000_000 * 1e9)
        end_ns = start_ns + int(24 * 3600 * 1e9)
        market = _FakeMarket("cid_v1", start_ns, end_ns)

        # Diagnostics with null sigma (v1 style)
        _make_diagnostics_parquet(diag_dir / "cid_v1.parquet", [
            _diag_row(condition_id="cid_v1", ts_ns=start_ns + int(1e9), sigma=None, tau_yr=None),
            _diag_row(condition_id="cid_v1", ts_ns=start_ns + int(2e9), sigma=None, tau_yr=None),
        ])

        # Write klines covering the window
        klines = [
            _kline_dict(start_ns, 50000.0, 50000.0),
            _kline_dict(end_ns, 51000.0, 51000.0),
        ]
        _make_klines_json(klines_dir / "klines.json", klines)

        out_path = tmp_path / "vol_realized.html"
        result = plot_vol_realized(diag_dir, klines_dir, [market], out_path)
        assert result is None
        assert not out_path.exists()

    def test_empty_parquet_sigma_returns_none(self, tmp_path: Path):
        """Diagnostics parquet exists but is empty (0 rows) → no file."""
        diag_dir = tmp_path / "diagnostics"
        klines_dir = tmp_path / "klines"

        market = _FakeMarket("cid_empty", int(1e18), int(2e18))
        _make_diagnostics_parquet(diag_dir / "cid_empty.parquet", [])
        klines = [_kline_dict(int(1e18), 50000.0, 51000.0)]
        _make_klines_json(klines_dir / "klines.json", klines)

        out_path = tmp_path / "vol_realized.html"
        result = plot_vol_realized(diag_dir, klines_dir, [market], out_path)
        assert result is None
        assert not out_path.exists()


# ---------------------------------------------------------------------------
# TC3: 5 synthetic markets — plotted (σ, |Δln S|) within 1e-9
# ---------------------------------------------------------------------------

class TestSyntheticMarkets:
    """With 5 synthetic markets + klines + diagnostics, assert correct (σ, |Δln S|)."""

    def _setup_5_markets(self, tmp_path: Path):
        """Build 5 markets with known σ and BTC price paths. Returns (markets, expected).

        Each market's klines are offset so start_ns and end_ns don't collide across
        markets (end of market i != start of market i+1).
        """
        diag_dir = tmp_path / "diagnostics"
        klines_dir = tmp_path / "klines"

        # Market timestamps: 5 windows separated by a 1h gap to avoid ts_ns collisions
        day_ns = int(24 * 3600 * 1e9)
        gap_ns = int(1 * 3600 * 1e9)  # 1h gap between markets
        base_ns = int(1_700_000_000 * 1e9)  # arbitrary epoch

        # Known σ values and price moves
        test_cases = [
            # (sigma, open_price, close_price)
            (0.50, 40000.0, 42000.0),
            (0.80, 45000.0, 44000.0),
            (0.60, 50000.0, 52500.0),
            (0.70, 55000.0, 55000.0),  # zero return
            (1.20, 60000.0, 57000.0),
        ]

        markets = []
        expected: list[tuple[float, float]] = []
        all_klines: list[dict] = []

        for i, (sigma, open_p, close_p) in enumerate(test_cases):
            cid = f"cid_{i}"
            start_ns = base_ns + i * (day_ns + gap_ns)
            end_ns = start_ns + day_ns
            market = _FakeMarket(cid, start_ns, end_ns)
            markets.append(market)

            # Write diagnostics: one ENTER row with the known sigma
            tau_enter = 6.0 * 3600 / _SECS_PER_YEAR  # 6h TTE at entry
            _make_diagnostics_parquet(diag_dir / f"{cid}.parquet", [
                _diag_row(
                    condition_id=cid,
                    ts_ns=start_ns + int(6 * 3600 * 1e9),
                    action="enter",
                    sigma=sigma,
                    tau_yr=tau_enter,
                ),
            ])

            # Build klines: one kline exactly at start (for open) and one exactly at end (for close)
            # Use distinct ts_ns values so no two klines collide
            all_klines.append(_kline_dict(start_ns, open_p, open_p))
            all_klines.append(_kline_dict(end_ns, close_p, close_p))

            # Expected realized |log-return|
            # open_val = open from kline at start_ns; close_val = close from kline at end_ns
            expected.append((sigma, abs(math.log(close_p / open_p))))

        # Write all klines to one JSON file
        _make_klines_json(klines_dir / "all_klines.json", all_klines)

        return markets, expected, diag_dir, klines_dir

    def test_points_match_expected_within_tolerance(self, tmp_path: Path):
        """_compute_vol_realized_points returns exactly the expected (σ, |Δln S|)."""
        markets, expected, diag_dir, klines_dir = self._setup_5_markets(tmp_path)
        points = _compute_vol_realized_points(diag_dir, klines_dir, markets)

        assert len(points) == 5
        for (got_sigma, got_rlr), (exp_sigma, exp_rlr) in zip(points, expected):
            assert abs(got_sigma - exp_sigma) < 1e-9, (
                f"sigma mismatch: got {got_sigma}, expected {exp_sigma}"
            )
            assert abs(got_rlr - exp_rlr) < 1e-9, (
                f"|Δln S| mismatch: got {got_rlr}, expected {exp_rlr}"
            )

    def test_one_point_per_evaluated_market(self, tmp_path: Path):
        """Returns exactly one point per market that has valid σ and klines."""
        markets, _, diag_dir, klines_dir = self._setup_5_markets(tmp_path)
        points = _compute_vol_realized_points(diag_dir, klines_dir, markets)
        assert len(points) == len(markets)

    def test_plot_written_and_nonempty(self, tmp_path: Path):
        """plot_vol_realized writes a non-empty HTML file and returns out_path."""
        markets, _, diag_dir, klines_dir = self._setup_5_markets(tmp_path)
        out_path = tmp_path / "vol_realized.html"
        result = plot_vol_realized(diag_dir, klines_dir, markets, out_path)
        assert result == out_path
        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_plot_is_html(self, tmp_path: Path):
        """Output file contains HTML markers."""
        markets, _, diag_dir, klines_dir = self._setup_5_markets(tmp_path)
        out_path = tmp_path / "vol_realized.html"
        plot_vol_realized(diag_dir, klines_dir, markets, out_path)
        content = out_path.read_text()
        assert "<html" in content.lower() or "plotly" in content.lower()


# ---------------------------------------------------------------------------
# TC4: Reference line exact equality — y = σ·√(1/365.25)
# ---------------------------------------------------------------------------

class TestReferenceLine:
    """The reference line passes through (σ, σ·√(1/365.25)) at every plotted σ."""

    def test_reference_line_exact(self, tmp_path: Path):
        """For each plotted σ, verify σ·√τ with τ=1/365.25 exactly."""
        diag_dir = tmp_path / "diagnostics"
        klines_dir = tmp_path / "klines"

        base_ns = int(1_700_000_000 * 1e9)
        day_ns = int(24 * 3600 * 1e9)
        sigmas = [0.4, 0.6, 0.8, 1.0, 1.2]
        open_price = 50000.0
        close_price = 51000.0

        all_klines: list[dict] = []
        markets = []
        for i, sigma in enumerate(sigmas):
            cid = f"cid_{i}"
            start_ns = base_ns + i * day_ns
            end_ns = start_ns + day_ns
            market = _FakeMarket(cid, start_ns, end_ns)
            markets.append(market)

            _make_diagnostics_parquet(diag_dir / f"{cid}.parquet", [
                _diag_row(
                    condition_id=cid,
                    ts_ns=start_ns + int(1e9),
                    action="enter",
                    sigma=sigma,
                    tau_yr=6.0 * 3600 / _SECS_PER_YEAR,
                ),
            ])
            all_klines.append(_kline_dict(start_ns, open_price, open_price))
            all_klines.append(_kline_dict(end_ns, close_price, close_price))

        _make_klines_json(klines_dir / "klines.json", all_klines)

        points = _compute_vol_realized_points(diag_dir, klines_dir, markets)
        assert len(points) == len(sigmas)

        for (sigma, _), expected_sigma in zip(points, sigmas):
            # The reference line value at this σ must be exactly σ·√(1/365.25)
            ref_y = sigma * math.sqrt(_TAU_1D)
            expected_ref_y = expected_sigma * math.sqrt(1.0 / 365.25)
            assert abs(ref_y - expected_ref_y) < 1e-12, (
                f"Reference line mismatch at σ={sigma}: "
                f"got {ref_y}, expected {expected_ref_y}"
            )


# ---------------------------------------------------------------------------
# TC5: σ selection fallback (no ENTER row → use midpoint-TTE row)
# ---------------------------------------------------------------------------

class TestSigmaFallback:
    def test_no_enter_uses_midpoint_tte_row(self, tmp_path: Path):
        """When no ENTER row exists, use the row closest to midpoint TTE σ."""
        diag_dir = tmp_path / "diagnostics"
        klines_dir = tmp_path / "klines"

        base_ns = int(1_700_000_000 * 1e9)
        day_ns = int(24 * 3600 * 1e9)
        start_ns = base_ns
        end_ns = start_ns + day_ns

        # Two HOLD rows: one far from midpoint TTE, one close to it
        mid_tau = _TTE_MID_YR
        far_tau = mid_tau * 2.0
        expected_sigma = 0.55  # the row closest to midpoint TTE

        _make_diagnostics_parquet(diag_dir / "cid_fallback.parquet", [
            _diag_row(
                condition_id="cid_fallback",
                ts_ns=start_ns + int(1e9),
                action="hold",
                sigma=0.99,  # far from midpoint
                tau_yr=far_tau,
            ),
            _diag_row(
                condition_id="cid_fallback",
                ts_ns=start_ns + int(2e9),
                action="hold",
                sigma=expected_sigma,  # closest to midpoint TTE
                tau_yr=mid_tau,
            ),
        ])

        all_klines = [
            _kline_dict(start_ns, 50000.0, 50000.0),
            _kline_dict(end_ns, 51000.0, 51000.0),
        ]
        _make_klines_json(klines_dir / "klines.json", all_klines)

        market = _FakeMarket("cid_fallback", start_ns, end_ns)
        points = _compute_vol_realized_points(diag_dir, klines_dir, [market])
        assert len(points) == 1
        assert abs(points[0][0] - expected_sigma) < 1e-9

    def test_enter_sigma_takes_priority_over_midpoint(self, tmp_path: Path):
        """ENTER row σ is used even if a closer-midpoint-TTE HOLD row exists."""
        diag_dir = tmp_path / "diagnostics"
        klines_dir = tmp_path / "klines"

        base_ns = int(1_700_000_000 * 1e9)
        day_ns = int(24 * 3600 * 1e9)
        start_ns = base_ns
        end_ns = start_ns + day_ns

        enter_sigma = 0.75
        hold_sigma = 0.55
        mid_tau = _TTE_MID_YR

        _make_diagnostics_parquet(diag_dir / "cid_prio.parquet", [
            # HOLD row exactly at midpoint TTE
            _diag_row(
                condition_id="cid_prio",
                ts_ns=start_ns + int(1e9),
                action="hold",
                sigma=hold_sigma,
                tau_yr=mid_tau,
            ),
            # ENTER row with a different sigma and farther TTE
            _diag_row(
                condition_id="cid_prio",
                ts_ns=start_ns + int(3e9),
                action="enter",
                sigma=enter_sigma,
                tau_yr=mid_tau * 0.1,
            ),
        ])

        all_klines = [
            _kline_dict(start_ns, 50000.0, 50000.0),
            _kline_dict(end_ns, 51000.0, 51000.0),
        ]
        _make_klines_json(klines_dir / "klines.json", all_klines)

        market = _FakeMarket("cid_prio", start_ns, end_ns)
        points = _compute_vol_realized_points(diag_dir, klines_dir, [market])
        assert len(points) == 1
        assert abs(points[0][0] - enter_sigma) < 1e-9


# ---------------------------------------------------------------------------
# TC6: No klines for a market → market skipped
# ---------------------------------------------------------------------------

class TestMissingKlines:
    def test_market_skipped_when_no_klines_at_all(self, tmp_path: Path):
        """Market with valid σ but empty klines_dir → skipped (no crash)."""
        diag_dir = tmp_path / "diagnostics"
        klines_dir = tmp_path / "klines"
        klines_dir.mkdir(parents=True, exist_ok=True)

        start_ns = int(1_700_000_000 * 1e9)
        end_ns = start_ns + int(24 * 3600 * 1e9)
        market = _FakeMarket("cid_noklines", start_ns, end_ns)

        _make_diagnostics_parquet(diag_dir / "cid_noklines.parquet", [
            _diag_row(
                condition_id="cid_noklines",
                ts_ns=start_ns + int(1e9),
                action="enter",
                sigma=0.8,
                tau_yr=0.001,
            ),
        ])
        # No klines written — klines_dir is empty

        points = _compute_vol_realized_points(diag_dir, klines_dir, [market])
        assert points == []

        out_path = tmp_path / "vol_realized.html"
        result = plot_vol_realized(diag_dir, klines_dir, [market], out_path)
        assert result is None
        assert not out_path.exists()

    def test_market_skipped_when_no_kline_before_start(self, tmp_path: Path):
        """No kline at or before start_ts_ns → open_val is None → market skipped."""
        diag_dir = tmp_path / "diagnostics"
        klines_dir = tmp_path / "klines"

        start_ns = int(1_700_000_000 * 1e9)
        end_ns = start_ns + int(24 * 3600 * 1e9)
        market = _FakeMarket("cid_nostartk", start_ns, end_ns)

        _make_diagnostics_parquet(diag_dir / "cid_nostartk.parquet", [
            _diag_row(
                condition_id="cid_nostartk",
                ts_ns=start_ns + int(1e9),
                action="enter",
                sigma=0.8,
                tau_yr=0.001,
            ),
        ])

        # Only a kline AFTER start_ns (but before end_ns) — no open_at_start
        after_start_ts = start_ns + int(1 * 3600 * 1e9)  # 1h after start
        _make_klines_json(klines_dir / "klines.json", [
            _kline_dict(after_start_ts, 50000.0, 51000.0),
        ])

        points = _compute_vol_realized_points(diag_dir, klines_dir, [market])
        assert points == []


# ---------------------------------------------------------------------------
# TC6b: Bisect regression — klines that extend past end_ts_ns are ignored
# ---------------------------------------------------------------------------

class TestBisectLookup:
    """_realized_abs_logret must handle a non-pre-filtered klines list correctly.

    This is the regression test that proves the bisect fix isn't fragile: even
    when the full klines list includes rows whose ts_ns > end_ts_ns, the result
    is identical to passing only the window-relevant klines.
    """

    def test_klines_past_end_do_not_affect_result(self, tmp_path: Path):
        """Extra klines after end_ts_ns are invisible to bisect lookup."""
        from hlanalysis.sim.plots.vol_realized import _realized_abs_logret

        start_ns = int(1_700_000_000 * 1e9)
        end_ns = start_ns + int(24 * 3600 * 1e9)

        open_price = 48_000.0
        close_price = 50_400.0
        expected_rlr = abs(math.log(close_price / open_price))

        # klines list containing:
        # - one at exactly start_ns  (open lookup target)
        # - one at exactly end_ns    (close lookup target)
        # - two AFTER end_ns         (should be ignored)
        klines = [
            _kline_dict(start_ns,                          open_price,  open_price),
            _kline_dict(end_ns,                            close_price, close_price),
            _kline_dict(end_ns + int(1 * 3600 * 1e9),     55_000.0,    55_000.0),
            _kline_dict(end_ns + int(25 * 3600 * 1e9),    60_000.0,    60_000.0),
        ]
        # List is already sorted; pass it unsorted to prove sort-stability
        import random as _random
        shuffled = list(klines)
        _random.shuffle(shuffled)
        # _realized_abs_logret requires a sorted list — sort it as _load_all_klines would
        shuffled.sort(key=lambda k: k["ts_ns"])

        result = _realized_abs_logret(shuffled, start_ns, end_ns)
        assert result is not None
        assert abs(result - expected_rlr) < 1e-12, (
            f"bisect result {result} != expected {expected_rlr}"
        )

    def test_compute_points_with_unfiltered_klines(self, tmp_path: Path):
        """_compute_vol_realized_points gives correct result when klines extend
        well beyond the market window (regression: old forward-scan was fragile)."""
        diag_dir = tmp_path / "diagnostics"
        klines_dir = tmp_path / "klines"

        start_ns = int(1_700_000_000 * 1e9)
        end_ns = start_ns + int(24 * 3600 * 1e9)
        market = _FakeMarket("cid_bisect", start_ns, end_ns)

        open_price = 52_000.0
        close_price = 49_400.0
        expected_rlr = abs(math.log(close_price / open_price))

        _make_diagnostics_parquet(diag_dir / "cid_bisect.parquet", [
            _diag_row(
                condition_id="cid_bisect",
                ts_ns=start_ns + int(1e9),
                action="enter",
                sigma=0.75,
                tau_yr=6.0 * 3600 / _SECS_PER_YEAR,
            ),
        ])

        # klines extend 3 days past end_ns — these extra rows must not pollute
        # the close_val bisect result.
        extra_klines = [
            _kline_dict(end_ns + int(d * 24 * 3600 * 1e9), 99_000.0 + d * 100, 99_000.0 + d * 100)
            for d in range(1, 4)
        ]
        all_klines = [
            _kline_dict(start_ns, open_price,  open_price),
            _kline_dict(end_ns,   close_price, close_price),
            *extra_klines,
        ]
        _make_klines_json(klines_dir / "klines.json", all_klines)

        points = _compute_vol_realized_points(diag_dir, klines_dir, [market])
        assert len(points) == 1
        assert abs(points[0][0] - 0.75) < 1e-9
        assert abs(points[0][1] - expected_rlr) < 1e-12, (
            f"expected {expected_rlr}, got {points[0][1]}"
        )


# ---------------------------------------------------------------------------
# TC7: Report integration — vol_realized.html link in report.md
# ---------------------------------------------------------------------------

@dataclass
class _FakeMarketWithOutcome:
    condition_id: str
    resolved_outcome: str
    start_ts_ns: int
    end_ts_ns: int


class TestReportIntegration:
    """Verify write_single_run_report links vol_realized.html when data present."""

    def _make_summary(self):
        from hlanalysis.sim.metrics import RunSummary
        return RunSummary(
            n_markets=1, n_trades=1, total_pnl_usd=5.0,
            sharpe=1.0, hit_rate=1.0, max_drawdown_usd=0.0,
        )

    def test_report_links_vol_realized_html(self, tmp_path: Path):
        """report.md contains link to vol_realized.html when v2 data present."""
        from hlanalysis.sim.report import write_single_run_report

        diag_dir = tmp_path / "diagnostics"
        klines_dir = tmp_path / "klines"

        base_ns = int(1_700_000_000 * 1e9)
        day_ns = int(24 * 3600 * 1e9)

        # Create 3 markets with diagnostics + klines
        markets = []
        all_klines: list[dict] = []
        for i in range(3):
            cid = f"cid_{i}"
            start_ns = base_ns + i * day_ns
            end_ns = start_ns + day_ns
            markets.append(_FakeMarketWithOutcome(
                condition_id=cid,
                resolved_outcome="yes",
                start_ts_ns=start_ns,
                end_ts_ns=end_ns,
            ))
            _make_diagnostics_parquet(diag_dir / f"{cid}.parquet", [
                _diag_row(
                    condition_id=cid,
                    ts_ns=start_ns + int(1e9),
                    action="enter",
                    sigma=0.6 + i * 0.1,
                    tau_yr=4.0 * 3600 / _SECS_PER_YEAR,
                ),
            ])
            all_klines.append(_kline_dict(start_ns, 50000.0 + i * 1000, 50000.0 + i * 1000))
            all_klines.append(_kline_dict(end_ns, 51000.0 + i * 1000, 51000.0 + i * 1000))

        _make_klines_json(klines_dir / "klines.json", all_klines)

        from hlanalysis.sim.metrics import RunSummary
        summary = RunSummary(
            n_markets=3, n_trades=3, total_pnl_usd=10.0,
            sharpe=1.0, hit_rate=1.0, max_drawdown_usd=0.0,
        )

        write_single_run_report(
            out_dir=tmp_path,
            strategy_name="v2",
            config_summary={"edge_buffer": 0.02},
            per_market_pnl=[3.0, 4.0, 3.0],
            summary=summary,
            markets=markets,
            diagnostics_dir=diag_dir,
            klines_dir=klines_dir,
        )

        assert (tmp_path / "vol_realized.html").exists()
        text = (tmp_path / "report.md").read_text()
        assert "vol_realized.html" in text
        assert "## Vol vs realized" in text

    def test_report_no_vol_realized_for_v1(self, tmp_path: Path):
        """report.md does NOT link vol_realized.html when only v1 diagnostics present."""
        from hlanalysis.sim.report import write_single_run_report

        diag_dir = tmp_path / "diagnostics"
        klines_dir = tmp_path / "klines"
        klines_dir.mkdir(parents=True)

        base_ns = int(1_700_000_000 * 1e9)
        day_ns = int(24 * 3600 * 1e9)

        market = _FakeMarketWithOutcome(
            condition_id="cid_v1",
            resolved_outcome="yes",
            start_ts_ns=base_ns,
            end_ts_ns=base_ns + day_ns,
        )

        # v1: all-null sigma
        _make_diagnostics_parquet(diag_dir / "cid_v1.parquet", [
            _diag_row(condition_id="cid_v1", ts_ns=base_ns + int(1e9), sigma=None, tau_yr=None),
        ])

        from hlanalysis.sim.metrics import RunSummary
        summary = RunSummary(
            n_markets=1, n_trades=0, total_pnl_usd=0.0,
            sharpe=0.0, hit_rate=0.0, max_drawdown_usd=0.0,
        )

        write_single_run_report(
            out_dir=tmp_path,
            strategy_name="v1",
            config_summary={"edge_buffer": 0.02},
            per_market_pnl=[0.0],
            summary=summary,
            markets=[market],
            diagnostics_dir=diag_dir,
            klines_dir=klines_dir,
        )

        assert not (tmp_path / "vol_realized.html").exists()
        text = (tmp_path / "report.md").read_text()
        assert "vol_realized.html" not in text

    def test_report_no_vol_realized_when_dirs_none(self, tmp_path: Path):
        """report.md does NOT link vol_realized.html when diagnostics_dir/klines_dir not passed."""
        from hlanalysis.sim.report import write_single_run_report

        market = _FakeMarketWithOutcome(
            condition_id="cid_x",
            resolved_outcome="yes",
            start_ts_ns=int(1e18),
            end_ts_ns=int(2e18),
        )

        from hlanalysis.sim.metrics import RunSummary
        summary = RunSummary(
            n_markets=1, n_trades=0, total_pnl_usd=0.0,
            sharpe=0.0, hit_rate=0.0, max_drawdown_usd=0.0,
        )

        write_single_run_report(
            out_dir=tmp_path,
            strategy_name="v2",
            config_summary={"edge_buffer": 0.02},
            per_market_pnl=[0.0],
            summary=summary,
            markets=[market],
            # diagnostics_dir and klines_dir deliberately omitted
        )

        assert not (tmp_path / "vol_realized.html").exists()
        text = (tmp_path / "report.md").read_text()
        assert "vol_realized.html" not in text
