"""Tests for Scanner._maybe_write_decision_trace.

Requirements verified:
- Traced question produces one JSONL row per scan (≥2 scans => ≥2 rows).
- All schema keys are present in each row.
- sigma is populated from a diagnostic when the strategy emits it.
- With tracing disabled (empty frozenset), no file is created.
- A non-traced question in the same scan produces no row while a traced one does.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import pytest

from hlanalysis.engine.config import (
    AllowlistEntry,
    GlobalRiskConfig,
    StrategyConfig,
)
from hlanalysis.engine.market_state import MarketState
from hlanalysis.engine.scanner import DECISION_TRACE_HEARTBEAT_NS, Scanner
from hlanalysis.engine.state import StateDAL
from hlanalysis.events import (
    BboEvent,
    MarkEvent,
    Mechanism,
    ProductType,
    QuestionMetaEvent,
)
from hlanalysis.strategy.base import Strategy
from hlanalysis.strategy.types import (
    Action,
    BookState,
    Decision,
    Diagnostic,
    Position,
    QuestionView,
)

# ── Required schema keys for every trace row ──────────────────────────────────
_SCHEMA_KEYS = {
    "ts_ns",
    "question_idx",
    "klass",
    "strategy_id",
    "action",
    "reason",
    "chosen_symbol",
    "chosen_side",
    "reference_price",
    "sigma",
    "p_model",
    "edge",
    "safety_d_entry",
    "safety_d_exit",
    "tte_s",
    "favorite_side",
    "intended_size",
    "intended_price",
    "bid_px",
    "bid_sz",
    "ask_px",
    "ask_sz",
    "position_qty",
    "position_avg_entry",
    "config_hash",
    "diag_fields",
}


# ── Minimal strategy that emits a known sigma diagnostic ─────────────────────


class _FixedHoldStrategy(Strategy):
    """Always returns HOLD with a Diagnostic that carries `sigma` and `p_model`
    fields — enough to populate the well-known trace columns without needing
    the full theta_harvester parameter tree.
    """

    name = "fixed_hold"
    consumes_hl_bars = False

    def __init__(self, sigma: float = 0.25) -> None:
        self._sigma = sigma

    def evaluate(
        self,
        *,
        question: QuestionView,
        books: Mapping[str, BookState],
        reference_price: float,
        recent_returns: tuple[float, ...],
        recent_volume_usd: float,
        position: Position | None,
        now_ns: int,
        recent_hl_bars: tuple[tuple[float, float], ...] = (),
    ) -> Decision:
        diag = Diagnostic(
            "info",
            "edge",
            (
                ("sigma", f"{self._sigma:.4f}"),
                ("p_model", "0.9200"),
                ("edge_yes", "0.0300"),
                ("chosen_leg", question.yes_symbol or ""),
            ),
        )
        return Decision(action=Action.HOLD, diagnostics=(diag,))


class _ControlStrategy(Strategy):
    """Strategy whose action and chosen leg can be mutated between scans, to
    exercise the on-change decision-trace throttle (action change / chosen-leg
    change force a row even inside the heartbeat window)."""

    name = "control"
    consumes_hl_bars = False

    def __init__(self) -> None:
        self.action: Action = Action.HOLD
        self.chosen_leg: str | None = None

    def evaluate(
        self,
        *,
        question: QuestionView,
        books: Mapping[str, BookState],
        reference_price: float,
        recent_returns: tuple[float, ...],
        recent_volume_usd: float,
        position: Position | None,
        now_ns: int,
        recent_hl_bars: tuple[tuple[float, float], ...] = (),
    ) -> Decision:
        leg = self.chosen_leg if self.chosen_leg is not None else (question.yes_symbol or "")
        diag = Diagnostic(
            "info",
            "edge",
            (
                ("sigma", "0.2500"),
                ("p_model", "0.9200"),
                ("edge_yes", "0.0300"),
                ("chosen_leg", leg),
            ),
        )
        return Decision(action=self.action, diagnostics=(diag,))


# ── Shared market / config helpers ────────────────────────────────────────────

_NOW = 1_700_000_000_000_000_000


def _expiry_str(now_ns: int, offset_s: int = 600) -> str:
    return datetime.fromtimestamp((now_ns + offset_s * 1_000_000_000) / 1e9, tz=UTC).strftime("%Y%m%d-%H%M")


def _seed_market(now_ns: int, *, question_idx: int = 42, outcome_idx: int = 3) -> MarketState:
    """Seed a single HL priceBinary question.

    MarketState generates leg symbols as ``f"#{10*outcome_idx+side}"`` for HL
    questions (named_outcome_idxs path).  For outcome_idx=3 that gives YES=#30
    and NO=#31; for outcome_idx=9 that gives YES=#90 and NO=#91.  The BBO
    events must use these generated symbols — the QuestionMetaEvent ``symbol``
    field is just an ingestion alias and is NOT the book key.
    """
    ms = MarketState()
    expiry_str = _expiry_str(now_ns)
    yes_sym = f"#{10 * outcome_idx}"
    ms.apply(
        QuestionMetaEvent(
            venue="hyperliquid",
            product_type=ProductType.PREDICTION_BINARY,
            mechanism=Mechanism.CLOB,
            symbol=yes_sym,
            exchange_ts=now_ns - 60_000_000_000,
            local_recv_ts=now_ns - 60_000_000_000,
            question_idx=question_idx,
            named_outcome_idxs=[outcome_idx],
            keys=["class", "underlying", "period", "expiry", "strike"],
            values=["priceBinary", "BTC", "1h", expiry_str, "80000"],
        )
    )
    for i in range(8):
        ts = now_ns - (8 - i) * 60_000_000_000
        ms.apply(
            MarkEvent(
                venue="hyperliquid",
                product_type=ProductType.PERP,
                mechanism=Mechanism.CLOB,
                symbol="BTC",
                exchange_ts=ts,
                local_recv_ts=ts,
                mark_px=80_300.0 + i * 0.01,
            )
        )
    ms.apply(
        BboEvent(
            venue="hyperliquid",
            product_type=ProductType.PREDICTION_BINARY,
            mechanism=Mechanism.CLOB,
            symbol=yes_sym,
            exchange_ts=now_ns,
            local_recv_ts=now_ns,
            bid_px=0.88,
            bid_sz=10.0,
            ask_px=0.90,
            ask_sz=8.0,
        )
    )
    return ms


def _strategy_cfg() -> StrategyConfig:
    entry = AllowlistEntry(
        match={"class": "priceBinary", "underlying": "BTC", "period": "1h"},
        max_position_usd=100,
        stop_loss_pct=10,
        tte_min_seconds=60,
        tte_max_seconds=1800,
        price_extreme_threshold=0.95,
        distance_from_strike_usd_min=200,
        vol_max=0.5,
    )
    return StrategyConfig(
        name="late_resolution",
        paper_mode=True,
        allowlist=[entry],
        blocklist_question_idxs=[],
        defaults=entry,
        **{
            "global": GlobalRiskConfig(
                max_total_inventory_usd=500,
                max_concurrent_positions=5,
                daily_loss_cap_usd=200,
                max_strike_distance_pct=10,
                min_recent_volume_usd=0,
                stale_data_halt_seconds=5,
                reconcile_interval_seconds=60,
            )
        },
    )


def _make_scanner(
    tmp_path: Path,
    *,
    ms: MarketState,
    trace_question_idxs: frozenset[int] = frozenset(),
    trace_filters: frozenset[tuple[str, str]] = frozenset(),
    decision_trace_path: Path | None = None,
    strategy_id: str | None = "test_slot",
    config_hash: str | None = "abc123",
    strategy: Strategy | None = None,
) -> Scanner:
    dal = StateDAL(tmp_path / "state.db")
    dal.run_migrations()
    cfg = _strategy_cfg()
    return Scanner(
        strategy=strategy if strategy is not None else _FixedHoldStrategy(sigma=0.25),
        cfg=cfg,
        market_state=ms,
        dal=dal,
        kill_switch_path=tmp_path / "halt",
        last_reconcile_ns=_NOW,
        decision_trace_path=decision_trace_path,
        trace_question_idxs=trace_question_idxs,
        trace_filters=trace_filters,
        strategy_id=strategy_id,
        config_hash=config_hash,
    )


def test_trace_coin_class_filter_matches(tmp_path: Path) -> None:
    """A coin/class filter traces a question even when its idx is NOT listed —
    so tomorrow's (unknown-idx) BTC binary is captured from open. Scans are
    spaced beyond the heartbeat so the throttle writes both rows."""
    ms = _seed_market(_NOW, question_idx=999)  # idx NOT in any idx set
    trace_path = tmp_path / "decision_trace.jsonl"
    scanner = _make_scanner(
        tmp_path,
        ms=ms,
        trace_question_idxs=frozenset(),  # no idx listed
        trace_filters=frozenset({("BTC", "priceBinary")}),
        decision_trace_path=trace_path,
    )
    scanner.scan(now_ns=_NOW)
    scanner.scan(now_ns=_NOW + DECISION_TRACE_HEARTBEAT_NS)
    assert trace_path.exists()
    rows = [json.loads(line) for line in trace_path.read_text().strip().splitlines()]
    assert len(rows) == 2
    assert rows[0]["question_idx"] == 999


def test_trace_coin_class_filter_no_match(tmp_path: Path) -> None:
    """A non-matching coin/class filter (different coin) writes nothing."""
    ms = _seed_market(_NOW, question_idx=999)  # underlying BTC
    trace_path = tmp_path / "decision_trace.jsonl"
    scanner = _make_scanner(
        tmp_path,
        ms=ms,
        trace_filters=frozenset({("ETH", "priceBinary")}),  # wrong coin
        decision_trace_path=trace_path,
    )
    scanner.scan(now_ns=_NOW)
    assert not trace_path.exists()


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_trace_disabled_no_file_created(tmp_path: Path) -> None:
    """With tracing disabled (empty frozenset), no trace file is created."""
    ms = _seed_market(_NOW)
    trace_path = tmp_path / "decision_trace.jsonl"
    scanner = _make_scanner(
        tmp_path,
        ms=ms,
        trace_question_idxs=frozenset(),  # disabled
        decision_trace_path=trace_path,
    )
    scanner.scan(now_ns=_NOW)
    scanner.scan(now_ns=_NOW + 1_000_000_000)
    assert not trace_path.exists(), "trace file must not be created when tracing is disabled"


def test_trace_enabled_heartbeat_row_per_scan(tmp_path: Path) -> None:
    """With tracing enabled and scans spaced at/over the heartbeat, one row is
    written per scan for the traced question (heartbeat forces a row even when
    the decision is unchanged)."""
    ms = _seed_market(_NOW, question_idx=42)
    trace_path = tmp_path / "decision_trace.jsonl"
    scanner = _make_scanner(
        tmp_path,
        ms=ms,
        trace_question_idxs=frozenset({42}),
        decision_trace_path=trace_path,
    )

    scanner.scan(now_ns=_NOW)
    scanner.scan(now_ns=_NOW + DECISION_TRACE_HEARTBEAT_NS)
    scanner.scan(now_ns=_NOW + 2 * DECISION_TRACE_HEARTBEAT_NS)

    assert trace_path.exists()
    rows = [json.loads(line) for line in trace_path.read_text().strip().splitlines()]
    assert len(rows) == 3, f"expected 3 rows (one per heartbeat-spaced scan), got {len(rows)}"


def test_trace_throttle_skips_unchanged_holds_within_heartbeat(tmp_path: Path) -> None:
    """Consecutive unchanged HOLD decisions within the heartbeat window are NOT
    written — only the first row appears."""
    ms = _seed_market(_NOW, question_idx=42)
    trace_path = tmp_path / "decision_trace.jsonl"
    scanner = _make_scanner(
        tmp_path,
        ms=ms,
        trace_question_idxs=frozenset({42}),
        decision_trace_path=trace_path,
    )

    # Three scans at 1s spacing — all within the 5s heartbeat, same decision.
    scanner.scan(now_ns=_NOW)
    scanner.scan(now_ns=_NOW + 1_000_000_000)
    scanner.scan(now_ns=_NOW + 2_000_000_000)

    rows = [json.loads(line) for line in trace_path.read_text().strip().splitlines()]
    assert len(rows) == 1, f"unchanged holds within heartbeat must collapse to 1 row, got {len(rows)}"
    assert rows[0]["ts_ns"] == _NOW


def test_trace_throttle_writes_when_heartbeat_elapsed(tmp_path: Path) -> None:
    """An unchanged HOLD is written again once the heartbeat has elapsed."""
    ms = _seed_market(_NOW, question_idx=42)
    trace_path = tmp_path / "decision_trace.jsonl"
    scanner = _make_scanner(
        tmp_path,
        ms=ms,
        trace_question_idxs=frozenset({42}),
        decision_trace_path=trace_path,
    )

    scanner.scan(now_ns=_NOW)
    # Just under the heartbeat → suppressed.
    scanner.scan(now_ns=_NOW + DECISION_TRACE_HEARTBEAT_NS - 1)
    # At/over the heartbeat (measured from the last WRITTEN row) → written.
    scanner.scan(now_ns=_NOW + DECISION_TRACE_HEARTBEAT_NS)

    rows = [json.loads(line) for line in trace_path.read_text().strip().splitlines()]
    assert len(rows) == 2, f"expected 2 rows (first + heartbeat), got {len(rows)}"
    assert rows[0]["ts_ns"] == _NOW
    assert rows[1]["ts_ns"] == _NOW + DECISION_TRACE_HEARTBEAT_NS


def test_trace_throttle_writes_on_action_change(tmp_path: Path) -> None:
    """A change in the decision's action forces a row even inside the heartbeat."""
    ms = _seed_market(_NOW, question_idx=42)
    trace_path = tmp_path / "decision_trace.jsonl"
    strat = _ControlStrategy()
    scanner = _make_scanner(
        tmp_path,
        ms=ms,
        trace_question_idxs=frozenset({42}),
        decision_trace_path=trace_path,
        strategy=strat,
    )

    strat.action = Action.HOLD
    scanner.scan(now_ns=_NOW)
    # Within heartbeat, but the action flips → must write.
    strat.action = Action.EXIT
    scanner.scan(now_ns=_NOW + 1_000_000_000)

    rows = [json.loads(line) for line in trace_path.read_text().strip().splitlines()]
    assert len(rows) == 2, f"action change must force a row, got {len(rows)}"
    assert rows[0]["action"] == "hold"
    assert rows[1]["action"] == "exit"


def test_trace_throttle_writes_on_chosen_leg_change(tmp_path: Path) -> None:
    """A change in the chosen leg/side forces a row even inside the heartbeat,
    while the same action stays HOLD."""
    ms = _seed_market(_NOW, question_idx=42, outcome_idx=3)  # YES=#30, NO=#31
    trace_path = tmp_path / "decision_trace.jsonl"
    strat = _ControlStrategy()
    scanner = _make_scanner(
        tmp_path,
        ms=ms,
        trace_question_idxs=frozenset({42}),
        decision_trace_path=trace_path,
        strategy=strat,
    )

    strat.chosen_leg = "#30"  # YES
    scanner.scan(now_ns=_NOW)
    # Within heartbeat, same action, but chosen leg flips to NO → must write.
    strat.chosen_leg = "#31"  # NO
    scanner.scan(now_ns=_NOW + 1_000_000_000)

    rows = [json.loads(line) for line in trace_path.read_text().strip().splitlines()]
    assert len(rows) == 2, f"chosen-leg change must force a row, got {len(rows)}"
    assert rows[0]["chosen_side"] == "yes"
    assert rows[1]["chosen_side"] == "no"


def test_trace_row_has_all_schema_keys(tmp_path: Path) -> None:
    """Every trace row must contain exactly the required schema keys (no missing keys)."""
    ms = _seed_market(_NOW, question_idx=42)
    trace_path = tmp_path / "decision_trace.jsonl"
    scanner = _make_scanner(
        tmp_path,
        ms=ms,
        trace_question_idxs=frozenset({42}),
        decision_trace_path=trace_path,
    )
    scanner.scan(now_ns=_NOW)

    rows = [json.loads(line) for line in trace_path.read_text().strip().splitlines()]
    assert rows, "expected at least one row"
    row = rows[0]
    missing = _SCHEMA_KEYS - set(row.keys())
    assert not missing, f"trace row missing keys: {missing}"


def test_trace_sigma_populated_from_diagnostic(tmp_path: Path) -> None:
    """sigma is extracted from the diagnostic kv fields and populated in the row."""
    ms = _seed_market(_NOW, question_idx=42)
    trace_path = tmp_path / "decision_trace.jsonl"
    scanner = _make_scanner(
        tmp_path,
        ms=ms,
        trace_question_idxs=frozenset({42}),
        decision_trace_path=trace_path,
        # Use the mock strategy that emits sigma=0.25
    )
    scanner.scan(now_ns=_NOW)

    rows = [json.loads(line) for line in trace_path.read_text().strip().splitlines()]
    assert rows
    row = rows[0]
    assert row["sigma"] == pytest.approx(0.25, abs=1e-4), f"expected sigma ~0.25 from diagnostic, got {row['sigma']!r}"


def test_trace_p_model_and_edge_populated(tmp_path: Path) -> None:
    """p_model and edge are extracted from the diagnostic kv fields."""
    ms = _seed_market(_NOW, question_idx=42)
    trace_path = tmp_path / "decision_trace.jsonl"
    scanner = _make_scanner(
        tmp_path,
        ms=ms,
        trace_question_idxs=frozenset({42}),
        decision_trace_path=trace_path,
    )
    scanner.scan(now_ns=_NOW)

    rows = [json.loads(line) for line in trace_path.read_text().strip().splitlines()]
    row = rows[0]
    assert row["p_model"] == pytest.approx(0.92, abs=1e-4)
    assert row["edge"] == pytest.approx(0.03, abs=1e-4)


def test_trace_metadata_fields(tmp_path: Path) -> None:
    """strategy_id, config_hash, question_idx, klass, action, reference_price are correct."""
    ms = _seed_market(_NOW, question_idx=42)
    trace_path = tmp_path / "decision_trace.jsonl"
    scanner = _make_scanner(
        tmp_path,
        ms=ms,
        trace_question_idxs=frozenset({42}),
        decision_trace_path=trace_path,
        strategy_id="v31",
        config_hash="deadbeef12345678",
    )
    scanner.scan(now_ns=_NOW)

    rows = [json.loads(line) for line in trace_path.read_text().strip().splitlines()]
    row = rows[0]
    assert row["strategy_id"] == "v31"
    assert row["config_hash"] == "deadbeef12345678"
    assert row["question_idx"] == 42
    assert row["klass"] == "priceBinary"
    assert row["action"] == "hold"
    assert row["ts_ns"] == _NOW
    # reference_price comes from the last BTC mark in _seed_market
    assert isinstance(row["reference_price"], float)
    assert row["reference_price"] > 0


def test_trace_book_fields_populated(tmp_path: Path) -> None:
    """bid_px, bid_sz, ask_px, ask_sz are extracted from the chosen leg's book."""
    ms = _seed_market(_NOW, question_idx=42)
    trace_path = tmp_path / "decision_trace.jsonl"
    scanner = _make_scanner(
        tmp_path,
        ms=ms,
        trace_question_idxs=frozenset({42}),
        decision_trace_path=trace_path,
    )
    scanner.scan(now_ns=_NOW)

    rows = [json.loads(line) for line in trace_path.read_text().strip().splitlines()]
    row = rows[0]
    # _seed_market sets bid_px=0.88, ask_px=0.90 on YES_SYM
    assert row["bid_px"] == pytest.approx(0.88)
    assert row["ask_px"] == pytest.approx(0.90)
    assert row["bid_sz"] == pytest.approx(10.0)
    assert row["ask_sz"] == pytest.approx(8.0)


def test_trace_non_traced_question_produces_no_row(tmp_path: Path) -> None:
    """A non-traced question in the same scan does not produce a trace row."""
    # Seed two questions: idx=42 with outcome_idx=3 (#30/#31) and
    # idx=99 with outcome_idx=9 (#90/#91) so the leg symbols don't collide.
    ms = _seed_market(_NOW, question_idx=42, outcome_idx=3)
    expiry_str = _expiry_str(_NOW)
    # Add a second question idx=99 using outcome_idx=9 → YES=#90, NO=#91.
    ms.apply(
        QuestionMetaEvent(
            venue="hyperliquid",
            product_type=ProductType.PREDICTION_BINARY,
            mechanism=Mechanism.CLOB,
            symbol="#90",
            exchange_ts=_NOW - 60_000_000_000,
            local_recv_ts=_NOW - 60_000_000_000,
            question_idx=99,
            named_outcome_idxs=[9],
            keys=["class", "underlying", "period", "expiry", "strike"],
            values=["priceBinary", "BTC", "1h", expiry_str, "80000"],
        )
    )
    ms.apply(
        BboEvent(
            venue="hyperliquid",
            product_type=ProductType.PREDICTION_BINARY,
            mechanism=Mechanism.CLOB,
            symbol="#90",
            exchange_ts=_NOW,
            local_recv_ts=_NOW,
            bid_px=0.85,
            bid_sz=5.0,
            ask_px=0.87,
            ask_sz=5.0,
        )
    )

    trace_path = tmp_path / "decision_trace.jsonl"
    scanner = _make_scanner(
        tmp_path,
        ms=ms,
        trace_question_idxs=frozenset({42}),  # only idx=42 traced
        decision_trace_path=trace_path,
    )
    scanner.scan(now_ns=_NOW)

    assert trace_path.exists()
    rows = [json.loads(line) for line in trace_path.read_text().strip().splitlines()]
    # Should have exactly one row for question_idx=42; idx=99 must not appear.
    assert len(rows) == 1
    assert rows[0]["question_idx"] == 42


def test_trace_diag_fields_present(tmp_path: Path) -> None:
    """diag_fields contains the raw kv pairs from the diagnostic."""
    ms = _seed_market(_NOW, question_idx=42)
    trace_path = tmp_path / "decision_trace.jsonl"
    scanner = _make_scanner(
        tmp_path,
        ms=ms,
        trace_question_idxs=frozenset({42}),
        decision_trace_path=trace_path,
    )
    scanner.scan(now_ns=_NOW)

    rows = [json.loads(line) for line in trace_path.read_text().strip().splitlines()]
    row = rows[0]
    assert row["diag_fields"] is not None
    keys_in_diag = {item["k"] for item in row["diag_fields"]}
    # _FixedHoldStrategy emits sigma, p_model, edge_yes, chosen_leg
    assert "sigma" in keys_in_diag
    assert "p_model" in keys_in_diag
    assert "edge_yes" in keys_in_diag


def test_trace_reason_populated(tmp_path: Path) -> None:
    """reason is the first diagnostic's message."""
    ms = _seed_market(_NOW, question_idx=42)
    trace_path = tmp_path / "decision_trace.jsonl"
    scanner = _make_scanner(
        tmp_path,
        ms=ms,
        trace_question_idxs=frozenset({42}),
        decision_trace_path=trace_path,
    )
    scanner.scan(now_ns=_NOW)

    rows = [json.loads(line) for line in trace_path.read_text().strip().splitlines()]
    row = rows[0]
    # _FixedHoldStrategy emits first diagnostic with message="edge"
    assert row["reason"] == "edge"


def test_trace_path_none_with_nonempty_idxs_is_silent(tmp_path: Path) -> None:
    """If trace_question_idxs is non-empty but decision_trace_path is None,
    no error is raised and nothing is written."""
    ms = _seed_market(_NOW, question_idx=42)
    scanner = _make_scanner(
        tmp_path,
        ms=ms,
        trace_question_idxs=frozenset({42}),
        decision_trace_path=None,  # no path
    )
    # Should not raise
    scanner.scan(now_ns=_NOW)
    scanner.scan(now_ns=_NOW + 1_000_000_000)
