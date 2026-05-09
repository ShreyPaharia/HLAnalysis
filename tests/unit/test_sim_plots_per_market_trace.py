"""Unit tests for hlanalysis.sim.plots.per_market_trace (Task C6).

Test strategy
-------------
1. Returns None and writes no file when diagnostics parquet doesn't exist.
2. Synthetic 1-market fixture with diagnostics + 1 ENTER + 1 settle fill:
   ``plot_market_trace`` returns the path, file exists, content non-empty.
3. HTML contains the entry timestamp string and a settlement annotation.
4. CLI subcommand ``cmd_trace`` invokes the plotter and writes to the correct path.
5. CLI default ``--out`` derivation: ``<run_dir>/traces/<market>.html``.
6. CLI when diagnostics is missing: logs an error, exits cleanly (no exception).
7. Markets with no fills at all still produce a YES/NO mid trace (no fill markers).
8. Markets with no-bid/ask nulls in some rows skip those rows gracefully.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from hlanalysis.sim.plots.per_market_trace import plot_market_trace, _ns_to_dt_str


# ---------------------------------------------------------------------------
# Shared schemas (match diagnostics.py exactly)
# ---------------------------------------------------------------------------

_DIAG_SCHEMA = pa.schema([
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

_FILLS_SCHEMA = pa.schema([
    pa.field("cloid",                   pa.string()),
    pa.field("ts_ns",                   pa.int64()),
    pa.field("side",                    pa.string()),
    pa.field("price",                   pa.float64()),
    pa.field("size",                    pa.float64()),
    pa.field("fee",                     pa.float64()),
    pa.field("condition_id",            pa.string()),
    pa.field("question_idx",            pa.int64()),
    pa.field("symbol",                  pa.string()),
    pa.field("entry_p_model",           pa.float64()),
    pa.field("entry_edge_chosen_side",  pa.float64()),
    pa.field("entry_sigma",             pa.float64()),
    pa.field("entry_tau_yr",            pa.float64()),
    pa.field("realized_pnl_at_settle",  pa.float64()),
    pa.field("resolved_outcome",        pa.string()),
])


# ---------------------------------------------------------------------------
# Helper writers
# ---------------------------------------------------------------------------

def _write_diagnostics(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        table = pa.table(
            {f.name: pa.array([], type=f.type) for f in _DIAG_SCHEMA},
            schema=_DIAG_SCHEMA,
        )
    else:
        cols: dict = {f.name: [] for f in _DIAG_SCHEMA}
        for row in rows:
            for f in _DIAG_SCHEMA:
                cols[f.name].append(row.get(f.name))
        arrays = {
            name: pa.array(vals, type=_DIAG_SCHEMA.field(name).type)
            for name, vals in cols.items()
        }
        table = pa.table(arrays, schema=_DIAG_SCHEMA)
    pq.write_table(table, path)


def _write_fills(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        table = pa.table(
            {f.name: pa.array([], type=f.type) for f in _FILLS_SCHEMA},
            schema=_FILLS_SCHEMA,
        )
    else:
        cols: dict = {f.name: [] for f in _FILLS_SCHEMA}
        for row in rows:
            for f in _FILLS_SCHEMA:
                cols[f.name].append(row.get(f.name))
        arrays = {
            name: pa.array(vals, type=_FILLS_SCHEMA.field(name).type)
            for name, vals in cols.items()
        }
        table = pa.table(arrays, schema=_FILLS_SCHEMA)
    pq.write_table(table, path)


# ---------------------------------------------------------------------------
# Fixture factories
# ---------------------------------------------------------------------------

_BASE_TS_NS = int(1_700_000_000 * 1e9)  # 2023-11-14 ~22:13 UTC — arbitrary fixed epoch
_ENTRY_TS_NS = _BASE_TS_NS + int(3_600 * 1e9)     # 1 hour in
_SETTLE_TS_NS = _BASE_TS_NS + int(86_400 * 1e9)   # 24 hours in (end of market)


def _diag_row(
    *,
    ts_ns: int,
    condition_id: str = "mkt_abc",
    action: str = "hold",
    yes_bid: Optional[float] = 0.45,
    yes_ask: Optional[float] = 0.55,
    no_bid: Optional[float] = 0.44,
    no_ask: Optional[float] = 0.56,
) -> dict:
    return dict(
        ts_ns=ts_ns,
        condition_id=condition_id,
        question_idx=0,
        action=action,
        reason="",
        p_model=None,
        edge_yes=None,
        edge_no=None,
        sigma=None,
        tau_yr=None,
        ln_sk=None,
        ref_price=50000.0,
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        no_bid=no_bid,
        no_ask=no_ask,
    )


def _enter_fill(
    *,
    ts_ns: int = _ENTRY_TS_NS,
    condition_id: str = "mkt_abc",
    side: str = "buy",
    price: float = 0.52,
) -> dict:
    return dict(
        cloid="cloid_enter_001",
        ts_ns=ts_ns,
        side=side,
        price=price,
        size=100.0,
        fee=0.0,
        condition_id=condition_id,
        question_idx=0,
        symbol="@0",
        entry_p_model=0.55,
        entry_edge_chosen_side=0.05,
        entry_sigma=0.4,
        entry_tau_yr=0.003,
        realized_pnl_at_settle=0.0,
        resolved_outcome=None,
    )


def _settle_fill(
    *,
    ts_ns: int = _SETTLE_TS_NS,
    condition_id: str = "mkt_abc",
    outcome_price: float = 1.0,  # 1.0 = YES won, 0.0 = NO won
    resolved_outcome: Optional[str] = None,
) -> dict:
    return dict(
        cloid="settle",
        ts_ns=ts_ns,
        side="sell",
        price=outcome_price,
        size=100.0,
        fee=0.0,
        condition_id=condition_id,
        question_idx=0,
        symbol="@0",
        entry_p_model=None,
        entry_edge_chosen_side=None,
        entry_sigma=None,
        entry_tau_yr=None,
        realized_pnl_at_settle=48.0,
        resolved_outcome=resolved_outcome,
    )


# ---------------------------------------------------------------------------
# Helper: build a run_dir with diagnostics + fills for one market
# ---------------------------------------------------------------------------

def _make_run_dir(
    tmp_path: Path,
    condition_id: str = "mkt_abc",
    diag_rows: Optional[list[dict]] = None,
    fill_rows: Optional[list[dict]] = None,
) -> Path:
    run_dir = tmp_path / "run"
    if diag_rows is not None:
        _write_diagnostics(run_dir / "diagnostics" / f"{condition_id}.parquet", diag_rows)
    if fill_rows is not None:
        _write_fills(run_dir / "fills" / f"{condition_id}.parquet", fill_rows)
    return run_dir


# ---------------------------------------------------------------------------
# _ns_to_dt_str unit test
# ---------------------------------------------------------------------------

class TestNsToDtStr:
    def test_zero_epoch(self):
        s = _ns_to_dt_str(0)
        assert s.startswith("1970-01-01")

    def test_roundtrip(self):
        import datetime
        ts_ns = int(1_700_000_000 * 1e9)
        s = _ns_to_dt_str(ts_ns)
        # Plotly needs ISO format; just verify it parses as datetime
        dt = datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))
        assert dt.year >= 2023


# ---------------------------------------------------------------------------
# Test 1: Returns None when diagnostics parquet missing
# ---------------------------------------------------------------------------

class TestMissingDiagnostics:
    def test_returns_none_when_diag_missing(self, tmp_path: Path):
        """plot_market_trace returns None and writes no file when diagnostics absent."""
        run_dir = tmp_path / "run"
        out_path = tmp_path / "trace.html"
        result = plot_market_trace("mkt_missing", run_dir, out_path)
        assert result is None
        assert not out_path.exists()

    def test_no_crash_on_missing_diag(self, tmp_path: Path):
        """No exception is raised when diagnostics are absent."""
        run_dir = tmp_path / "run"
        out_path = tmp_path / "trace.html"
        # Should not raise
        plot_market_trace("no_such_market", run_dir, out_path)


# ---------------------------------------------------------------------------
# Test 2: Synthetic 1-market fixture: returns path, file exists, non-empty
# ---------------------------------------------------------------------------

class TestSyntheticOneMarket:
    def _build_run(self, tmp_path: Path, condition_id: str = "mkt_abc") -> Path:
        diag_rows = [
            _diag_row(ts_ns=_BASE_TS_NS + i * int(3_600 * 1e9), condition_id=condition_id)
            for i in range(5)
        ]
        fill_rows = [
            _enter_fill(ts_ns=_ENTRY_TS_NS, condition_id=condition_id),
            _settle_fill(ts_ns=_SETTLE_TS_NS, condition_id=condition_id, outcome_price=1.0),
        ]
        return _make_run_dir(tmp_path, condition_id, diag_rows, fill_rows)

    def test_returns_path(self, tmp_path: Path):
        run_dir = self._build_run(tmp_path)
        out_path = tmp_path / "trace.html"
        result = plot_market_trace("mkt_abc", run_dir, out_path)
        assert result == out_path

    def test_file_exists(self, tmp_path: Path):
        run_dir = self._build_run(tmp_path)
        out_path = tmp_path / "trace.html"
        plot_market_trace("mkt_abc", run_dir, out_path)
        assert out_path.exists()

    def test_file_non_empty(self, tmp_path: Path):
        run_dir = self._build_run(tmp_path)
        out_path = tmp_path / "trace.html"
        plot_market_trace("mkt_abc", run_dir, out_path)
        assert out_path.stat().st_size > 0

    def test_output_is_html(self, tmp_path: Path):
        run_dir = self._build_run(tmp_path)
        out_path = tmp_path / "trace.html"
        plot_market_trace("mkt_abc", run_dir, out_path)
        content = out_path.read_text()
        assert "<html" in content.lower() or "plotly" in content.lower()

    def test_creates_parent_dirs(self, tmp_path: Path):
        run_dir = self._build_run(tmp_path)
        out_path = tmp_path / "traces" / "mkt_abc.html"
        result = plot_market_trace("mkt_abc", run_dir, out_path)
        assert result == out_path
        assert out_path.exists()


# ---------------------------------------------------------------------------
# Test 3: HTML contains entry timestamp string and settlement annotation
# (Acceptance Criterion 2)
# ---------------------------------------------------------------------------

class TestHtmlContents:
    def _build_run(self, tmp_path: Path, condition_id: str = "mkt_abc") -> Path:
        diag_rows = [
            _diag_row(ts_ns=_BASE_TS_NS, condition_id=condition_id),
            _diag_row(ts_ns=_ENTRY_TS_NS, condition_id=condition_id, action="enter"),
            _diag_row(ts_ns=_SETTLE_TS_NS, condition_id=condition_id),
        ]
        fill_rows = [
            _enter_fill(ts_ns=_ENTRY_TS_NS, condition_id=condition_id),
            _settle_fill(ts_ns=_SETTLE_TS_NS, condition_id=condition_id,
                         outcome_price=1.0, resolved_outcome="yes"),
        ]
        return _make_run_dir(tmp_path, condition_id, diag_rows, fill_rows)

    def test_html_contains_entry_timestamp(self, tmp_path: Path):
        """HTML must contain the full entry timestamp (HH:MM:SS) as a datetime string."""
        run_dir = self._build_run(tmp_path)
        out_path = tmp_path / "trace.html"
        plot_market_trace("mkt_abc", run_dir, out_path)
        content = out_path.read_text()
        # _ns_to_dt_str converts to ISO datetime with time component; check HH:MM:SS present
        entry_dt_str = _ns_to_dt_str(_ENTRY_TS_NS)
        # Assert the full timestamp (at least up to seconds, e.g. "2023-11-14T23:13:20")
        # is present in the serialised HTML/JSON blob — not just the date portion
        timestamp_prefix = entry_dt_str[:19]  # "YYYY-MM-DDTHH:MM:SS"
        assert timestamp_prefix in content, (
            f"Entry timestamp '{timestamp_prefix}' not found in HTML output"
        )

    def test_html_contains_settlement_annotation(self, tmp_path: Path):
        """HTML must contain the settlement marker text 'settled YES'."""
        run_dir = self._build_run(tmp_path)
        out_path = tmp_path / "trace.html"
        plot_market_trace("mkt_abc", run_dir, out_path)
        content = out_path.read_text()
        assert "settled YES" in content, "Settlement annotation 'settled YES' not found in HTML"

    def test_html_contains_settlement_no_annotation(self, tmp_path: Path):
        """HTML must contain 'settled NO' when resolved_outcome='no'."""
        diag_rows = [_diag_row(ts_ns=_BASE_TS_NS, condition_id="mkt_no")]
        fill_rows = [
            _enter_fill(ts_ns=_ENTRY_TS_NS, condition_id="mkt_no"),
            _settle_fill(ts_ns=_SETTLE_TS_NS, condition_id="mkt_no",
                         outcome_price=0.0, resolved_outcome="no"),
        ]
        run_dir = _make_run_dir(tmp_path, "mkt_no", diag_rows, fill_rows)
        out_path = tmp_path / "trace_no.html"
        plot_market_trace("mkt_no", run_dir, out_path)
        content = out_path.read_text()
        assert "settled NO" in content, "Settlement annotation 'settled NO' not found in HTML"

    def test_html_contains_condition_id_in_title(self, tmp_path: Path):
        """HTML plot title should reference the condition_id."""
        run_dir = self._build_run(tmp_path)
        out_path = tmp_path / "trace.html"
        plot_market_trace("mkt_abc", run_dir, out_path)
        content = out_path.read_text()
        assert "mkt_abc" in content


# ---------------------------------------------------------------------------
# Test 7: Markets with no fills still produce YES/NO mid trace
# ---------------------------------------------------------------------------

class TestNoFills:
    def test_no_fills_produces_trace(self, tmp_path: Path):
        """Market with no fills (no fills parquet) still produces a trace."""
        diag_rows = [
            _diag_row(ts_ns=_BASE_TS_NS + i * int(3_600 * 1e9))
            for i in range(3)
        ]
        run_dir = _make_run_dir(tmp_path, "mkt_nofill", diag_rows, fill_rows=None)
        out_path = tmp_path / "trace_nofill.html"
        result = plot_market_trace("mkt_nofill", run_dir, out_path)
        assert result == out_path
        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_empty_fills_parquet_produces_trace(self, tmp_path: Path):
        """Market with an empty fills parquet still produces a mid trace."""
        diag_rows = [_diag_row(ts_ns=_BASE_TS_NS)]
        fill_rows: list[dict] = []  # empty
        run_dir = _make_run_dir(tmp_path, "mkt_emptyfill", diag_rows, fill_rows)
        out_path = tmp_path / "trace_ef.html"
        result = plot_market_trace("mkt_emptyfill", run_dir, out_path)
        assert result == out_path
        assert out_path.exists()

    def test_no_fills_no_settlement_marker(self, tmp_path: Path):
        """With no fills, there is no settlement annotation in the output."""
        diag_rows = [_diag_row(ts_ns=_BASE_TS_NS)]
        run_dir = _make_run_dir(tmp_path, "mkt_nosettlecheck", diag_rows, fill_rows=None)
        out_path = tmp_path / "trace_nosettlecheck.html"
        plot_market_trace("mkt_nosettlecheck", run_dir, out_path)
        content = out_path.read_text()
        assert "settled YES" not in content
        assert "settled NO" not in content


# ---------------------------------------------------------------------------
# Test 8: Diagnostics rows with null bid/ask are gracefully skipped
# ---------------------------------------------------------------------------

class TestNullBidAskRows:
    def test_null_yes_bid_ask_rows_skipped(self, tmp_path: Path):
        """Rows where yes_bid or yes_ask is None are silently skipped."""
        diag_rows = [
            _diag_row(ts_ns=_BASE_TS_NS, yes_bid=None, yes_ask=None),
            _diag_row(ts_ns=_BASE_TS_NS + int(3_600 * 1e9), yes_bid=0.45, yes_ask=0.55),
        ]
        run_dir = _make_run_dir(tmp_path, "mkt_nullbidask", diag_rows, fill_rows=None)
        out_path = tmp_path / "trace_nullbidask.html"
        result = plot_market_trace("mkt_nullbidask", run_dir, out_path)
        assert result == out_path
        assert out_path.exists()


# ---------------------------------------------------------------------------
# Test 4 & 5: CLI subcommand cmd_trace
# ---------------------------------------------------------------------------

class TestCmdTrace:
    def _build_run(self, tmp_path: Path, condition_id: str = "mkt_cli") -> Path:
        diag_rows = [_diag_row(ts_ns=_BASE_TS_NS, condition_id=condition_id)]
        fill_rows = [
            _enter_fill(ts_ns=_ENTRY_TS_NS, condition_id=condition_id),
            _settle_fill(ts_ns=_SETTLE_TS_NS, condition_id=condition_id),
        ]
        return _make_run_dir(tmp_path, condition_id, diag_rows, fill_rows)

    def test_cmd_trace_explicit_out(self, tmp_path: Path):
        """cmd_trace with --out writes to the specified path."""
        from hlanalysis.sim.cli import cmd_trace
        run_dir = self._build_run(tmp_path)
        out_path = tmp_path / "explicit_out.html"
        args = argparse.Namespace(
            run_dir=str(run_dir),
            market="mkt_cli",
            out=str(out_path),
        )
        cmd_trace(args)
        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_cmd_trace_default_out(self, tmp_path: Path):
        """cmd_trace without --out writes to <run_dir>/traces/<market>.html."""
        from hlanalysis.sim.cli import cmd_trace
        run_dir = self._build_run(tmp_path)
        expected = run_dir / "traces" / "mkt_cli.html"
        args = argparse.Namespace(
            run_dir=str(run_dir),
            market="mkt_cli",
            out=None,
        )
        cmd_trace(args)
        assert expected.exists()
        assert expected.stat().st_size > 0

    def test_cmd_trace_missing_diagnostics_logs_error(self, tmp_path: Path, caplog):
        """cmd_trace with missing diagnostics logs error and doesn't raise."""
        from hlanalysis.sim.cli import cmd_trace
        run_dir = tmp_path / "empty_run"
        out_path = tmp_path / "should_not_exist.html"
        args = argparse.Namespace(
            run_dir=str(run_dir),
            market="no_such_market",
            out=str(out_path),
        )
        # Should not raise
        import logging
        with caplog.at_level(logging.ERROR):
            cmd_trace(args)
        assert not out_path.exists()

    def test_cmd_trace_invokes_plotter(self, tmp_path: Path, monkeypatch):
        """cmd_trace calls plot_market_trace with the correct arguments."""
        from hlanalysis.sim.cli import cmd_trace
        run_dir = self._build_run(tmp_path)
        out_path = tmp_path / "out.html"

        calls: list[tuple] = []

        def _mock_plot(condition_id, run_dir_arg, out_path_arg):
            calls.append((condition_id, run_dir_arg, out_path_arg))
            return out_path_arg

        import hlanalysis.sim.cli as cli_mod
        monkeypatch.setattr(cli_mod, "plot_market_trace", _mock_plot)

        args = argparse.Namespace(
            run_dir=str(run_dir),
            market="mkt_cli",
            out=str(out_path),
        )
        cmd_trace(args)

        assert len(calls) == 1
        assert calls[0][0] == "mkt_cli"
        assert calls[0][1] == run_dir
        assert calls[0][2] == out_path

    def test_cmd_trace_default_out_path_derivation(self, tmp_path: Path):
        """Default out-path is exactly <run_dir>/traces/<market>.html."""
        from hlanalysis.sim.cli import cmd_trace
        import hlanalysis.sim.cli as cli_mod

        run_dir = tmp_path / "my_run"
        captured: list[Path] = []

        def _mock_plot(condition_id, run_dir_arg, out_path_arg):
            captured.append(out_path_arg)
            return None  # simulate no diagnostics

        old = cli_mod.plot_market_trace
        cli_mod.plot_market_trace = _mock_plot
        try:
            args = argparse.Namespace(
                run_dir=str(run_dir),
                market="some_id",
                out=None,
            )
            cmd_trace(args)
        finally:
            cli_mod.plot_market_trace = old

        assert len(captured) == 1
        assert captured[0] == run_dir / "traces" / "some_id.html"


# ---------------------------------------------------------------------------
# Regression: NO-leg position with price=1.0 settle row must label "settled NO"
# Bug: without resolved_outcome, price=1.0 was inferred as "settled YES" even
# when the held position was on NO (which also wins at price=1.0).
# ---------------------------------------------------------------------------

class TestSettleOutcomeFromResolvedOutcome:
    """Regression tests for the settle-label bug.

    When a strategy buys NO and NO wins, runner.py sets settle_px=1.0
    (the held leg won). The old code inferred settle_outcome from price:
    price >= 0.5 → "YES", which is wrong. The fix reads resolved_outcome
    from the fills parquet settle row instead.
    """

    def _build_no_leg_run(
        self,
        tmp_path: Path,
        resolved_outcome: str,
        outcome_price: float,
    ) -> Path:
        """Build a run where strategy entered NO side and market settled."""
        condition_id = "mkt_no_leg"
        diag_rows = [
            _diag_row(ts_ns=_BASE_TS_NS, condition_id=condition_id),
            _diag_row(ts_ns=_ENTRY_TS_NS, condition_id=condition_id, action="enter"),
        ]
        fill_rows = [
            # ENTER on NO side (buy NO token)
            _enter_fill(ts_ns=_ENTRY_TS_NS, condition_id=condition_id,
                        side="buy", price=0.38),
            # Settle: price=1.0 because NO won (held leg wins), resolved_outcome="no"
            _settle_fill(
                ts_ns=_SETTLE_TS_NS,
                condition_id=condition_id,
                outcome_price=outcome_price,
                resolved_outcome=resolved_outcome,
            ),
        ]
        return _make_run_dir(tmp_path, condition_id, diag_rows, fill_rows)

    def test_no_leg_wins_label_is_settled_no(self, tmp_path: Path):
        """When NO won (resolved_outcome='no'), label must be 'settled NO', NOT 'settled YES'.

        Regression: the old price-based inference (price=1.0 → 'YES') was wrong
        when the held position was on NO and NO won.
        """
        run_dir = self._build_no_leg_run(
            tmp_path, resolved_outcome="no", outcome_price=1.0
        )
        out_path = tmp_path / "trace_no_leg.html"
        plot_market_trace("mkt_no_leg", run_dir, out_path)
        content = out_path.read_text()
        assert "settled NO" in content, (
            "Expected 'settled NO' when NO won (resolved_outcome='no'), "
            f"even though settle price=1.0 — old code wrongly labelled it 'settled YES'"
        )
        assert "settled YES" not in content, (
            "Should NOT contain 'settled YES' when resolved_outcome='no'"
        )

    def test_yes_leg_wins_label_is_settled_yes(self, tmp_path: Path):
        """When YES won (resolved_outcome='yes'), label must be 'settled YES'."""
        run_dir = self._build_no_leg_run(
            tmp_path, resolved_outcome="yes", outcome_price=1.0
        )
        out_path = tmp_path / "trace_yes_leg.html"
        plot_market_trace("mkt_no_leg", run_dir, out_path)
        content = out_path.read_text()
        assert "settled YES" in content, (
            "Expected 'settled YES' when YES won (resolved_outcome='yes')"
        )
        assert "settled NO" not in content, (
            "Should NOT contain 'settled NO' when resolved_outcome='yes'"
        )

    def test_unknown_outcome_label_is_settled_no_side(self, tmp_path: Path):
        """When resolved_outcome='unknown', label must be plain 'settled' (no side)."""
        run_dir = self._build_no_leg_run(
            tmp_path, resolved_outcome="unknown", outcome_price=1.0
        )
        out_path = tmp_path / "trace_unknown.html"
        plot_market_trace("mkt_no_leg", run_dir, out_path)
        content = out_path.read_text()
        # The label "settled" should appear, but NOT "settled YES" or "settled NO"
        assert "settled YES" not in content, "Should NOT contain 'settled YES' for unknown outcome"
        assert "settled NO" not in content, "Should NOT contain 'settled NO' for unknown outcome"

    def test_null_resolved_outcome_falls_back_to_plain_settled(self, tmp_path: Path):
        """When resolved_outcome is null (old parquet without column), label is plain 'settled'."""
        # Write fills WITHOUT resolved_outcome column (simulates old parquet schema)
        condition_id = "mkt_null_ro"
        old_fills_schema = pa.schema([
            pa.field("cloid",                   pa.string()),
            pa.field("ts_ns",                   pa.int64()),
            pa.field("side",                    pa.string()),
            pa.field("price",                   pa.float64()),
            pa.field("size",                    pa.float64()),
            pa.field("fee",                     pa.float64()),
            pa.field("condition_id",            pa.string()),
            pa.field("question_idx",            pa.int64()),
            pa.field("symbol",                  pa.string()),
            pa.field("entry_p_model",           pa.float64()),
            pa.field("entry_edge_chosen_side",  pa.float64()),
            pa.field("entry_sigma",             pa.float64()),
            pa.field("entry_tau_yr",            pa.float64()),
            pa.field("realized_pnl_at_settle",  pa.float64()),
        ])
        run_dir = tmp_path / "run_null_ro"
        diag_path = run_dir / "diagnostics" / f"{condition_id}.parquet"
        diag_path.parent.mkdir(parents=True, exist_ok=True)
        _write_diagnostics(diag_path, [_diag_row(ts_ns=_BASE_TS_NS, condition_id=condition_id)])

        # Write fills without resolved_outcome column (old schema)
        fills_path = run_dir / "fills" / f"{condition_id}.parquet"
        fills_path.parent.mkdir(parents=True, exist_ok=True)
        fills_row = {
            "cloid": "settle", "ts_ns": _SETTLE_TS_NS, "side": "sell",
            "price": 1.0, "size": 100.0, "fee": 0.0,
            "condition_id": condition_id, "question_idx": 0, "symbol": "@0",
            "entry_p_model": None, "entry_edge_chosen_side": None,
            "entry_sigma": None, "entry_tau_yr": None, "realized_pnl_at_settle": 48.0,
        }
        cols: dict = {f.name: [] for f in old_fills_schema}
        for f in old_fills_schema:
            cols[f.name].append(fills_row.get(f.name))
        arrays = {
            name: pa.array(vals, type=old_fills_schema.field(name).type)
            for name, vals in cols.items()
        }
        pq.write_table(pa.table(arrays, schema=old_fills_schema), fills_path)

        out_path = tmp_path / "trace_null_ro.html"
        plot_market_trace(condition_id, run_dir, out_path)
        content = out_path.read_text()
        # With no resolved_outcome column, should not label a specific side
        assert "settled YES" not in content, "Should NOT use price heuristic for missing column"
        assert "settled NO" not in content, "Should NOT use price heuristic for missing column"
