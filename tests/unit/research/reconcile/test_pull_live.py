"""Offline tests for pull_live (SSM mocked) + settlement-winner parity.

These never touch real AWS/SSM: ``_ssm_python`` is monkeypatched to capture the
generated remote script and return canned JSON.
"""

from __future__ import annotations

import json
import re

import pandas as pd
import pytest

from hlanalysis.research.reconcile import pull_live
from hlanalysis.research.reconcile.reconcile import (
    _winner_from_settlement_fill,
    reconcile_pnl,
)

_EXPIRY_NS = 1_718_000_000_000_000_000


class _Recorder:
    """Capture the remote script and return a canned stdout payload."""

    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.scripts: list[str] = []

    def __call__(self, script: str, instance_id: str = "i-x", timeout_s: int = 60) -> str:
        self.scripts.append(script)
        return self.payload


# ── pull_live_fills ─────────────────────────────────────────────────────────


class TestPullLiveFillsSQL:
    def test_queries_singular_fill_table(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rec = _Recorder(json.dumps([]))
        monkeypatch.setattr(pull_live, "_ssm_python", rec)
        pull_live.pull_live_fills(4010, _EXPIRY_NS)
        script = rec.scripts[0]
        # Table is `fill` (singular), never `fills`.
        assert re.search(r"\bFROM fill\b", script)
        assert "FROM fills" not in script

    def test_targets_unified_db_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rec = _Recorder(json.dumps([]))
        monkeypatch.setattr(pull_live, "_ssm_python", rec)
        pull_live.pull_live_fills(4010, _EXPIRY_NS)
        script = rec.scripts[0]
        assert "/opt/hl-recorder/data/engine/state.db" in script
        # NOT the legacy per-slot path.
        assert "/engine/v31/state.db" not in script
        # Opened read-only.
        assert "mode=ro" in script

    def test_filters_by_strategy_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rec = _Recorder(json.dumps([]))
        monkeypatch.setattr(pull_live, "_ssm_python", rec)
        pull_live.pull_live_fills(4010, _EXPIRY_NS, strategy_id="v1")
        script = rec.scripts[0]
        assert "strategy_id = ?" in script
        assert "'v1'" in script  # bound param embedded via repr
        assert "source = 'venue'" in script

    def test_no_broad_except_swallow(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The wrong-table error must propagate, not be silently swallowed."""
        rec = _Recorder(json.dumps([]))
        monkeypatch.setattr(pull_live, "_ssm_python", rec)
        pull_live.pull_live_fills(4010, _EXPIRY_NS)
        script = rec.scripts[0]
        assert "except Exception" not in script

    def test_returns_parsed_rows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        payload = json.dumps(
            [
                {
                    "ts_ns": _EXPIRY_NS - 100,
                    "symbol": "BTC #410",
                    "side": "BUY",
                    "price": 0.85,
                    "size": 100.0,
                    "fee": 0.0,
                    "closed_pnl": 0.0,
                }
            ]
        )
        rec = _Recorder(payload)
        monkeypatch.setattr(pull_live, "_ssm_python", rec)
        df = pull_live.pull_live_fills(4010, _EXPIRY_NS)
        assert len(df) == 1
        assert df.iloc[0]["symbol"] == "BTC #410"


# ── pull_settlement ─────────────────────────────────────────────────────────


class TestPullSettlementSQL:
    def test_reads_settlement_table_with_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rec = _Recorder(json.dumps({}))
        monkeypatch.setattr(pull_live, "_ssm_python", rec)
        pull_live.pull_settlement(4010, _EXPIRY_NS)
        script = rec.scripts[0]
        # Primary: the settlement table (NOT the events table).
        assert "FROM settlement" in script
        assert "kind = 'settlement'" not in script
        assert "SUM(realized_pnl)" in script
        # Fallback: HL settlement-as-fill from the fill table.
        assert "FROM fill" in script
        assert "settlement_as_fill" in script
        assert "strategy_id = ?" in script

    def test_settlement_table_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        payload = json.dumps(
            {
                "question_idx": 4010,
                "realized_pnl": 12.5,
                "ts_ns": _EXPIRY_NS,
                "winner_side": None,
                "source": "settlement_table",
            }
        )
        rec = _Recorder(payload)
        monkeypatch.setattr(pull_live, "_ssm_python", rec)
        result = pull_live.pull_settlement(4010, _EXPIRY_NS)
        assert result["source"] == "settlement_table"
        assert result["realized_pnl"] == 12.5

    def test_settlement_as_fill_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        payload = json.dumps(
            {
                "question_idx": 4010,
                "realized_pnl": 15.0,
                "ts_ns": _EXPIRY_NS,
                "winner_side": "yes",
                "settlement_price": 1.0,
                "source": "settlement_as_fill",
            }
        )
        rec = _Recorder(payload)
        monkeypatch.setattr(pull_live, "_ssm_python", rec)
        result = pull_live.pull_settlement(4010, _EXPIRY_NS)
        assert result["source"] == "settlement_as_fill"
        assert result["winner_side"] == "yes"


# ── trace / config / halts paths ────────────────────────────────────────────


class TestSlotScopedPaths:
    def test_trace_path_built_from_strategy_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rec = _Recorder(json.dumps([]))
        monkeypatch.setattr(pull_live, "_ssm_python", rec)
        pull_live.pull_live_trace(4010, _EXPIRY_NS, strategy_id="v1")
        assert "/opt/hl-recorder/data/engine/v1/decision_trace.jsonl" in rec.scripts[0]

    def test_config_hash_path_built_from_strategy_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rec = _Recorder(json.dumps(None))
        monkeypatch.setattr(pull_live, "_ssm_python", rec)
        pull_live.pull_config_hash(strategy_id="v1")
        assert "/opt/hl-recorder/data/engine/v1/decision_trace.jsonl" in rec.scripts[0]

    def test_halts_filters_strategy_id_unified_db(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rec = _Recorder(json.dumps([]))
        monkeypatch.setattr(pull_live, "_ssm_python", rec)
        pull_live.pull_halts_rejects(4010, _EXPIRY_NS, strategy_id="v31")
        script = rec.scripts[0]
        assert "/opt/hl-recorder/data/engine/state.db" in script
        assert "/engine/v31/state.db" not in script
        assert "strategy_id = ?" in script


# ── settlement-as-fill winner derivation ────────────────────────────────────


class TestWinnerFromSettlementFill:
    def test_yes_leg_wins_at_one(self) -> None:
        # symbol #410 -> side_idx 0 = Yes; settled at ~1.0 => Yes won.
        fills = pd.DataFrame(
            [
                {"ts_ns": 1, "symbol": "BTC #410", "price": 0.85, "side": "BUY", "size": 100.0},
                {"ts_ns": 2, "symbol": "BTC #410", "price": 1.0, "side": "SELL", "size": 100.0},
            ]
        )
        assert _winner_from_settlement_fill(fills) == "yes"

    def test_yes_leg_loses_at_zero(self) -> None:
        # Yes leg settled at ~0.0 => No won.
        fills = pd.DataFrame(
            [
                {"ts_ns": 1, "symbol": "BTC #410", "price": 0.85, "side": "BUY", "size": 100.0},
                {"ts_ns": 2, "symbol": "BTC #410", "price": 0.0, "side": "SELL", "size": 100.0},
            ]
        )
        assert _winner_from_settlement_fill(fills) == "no"

    def test_no_leg_wins_at_one(self) -> None:
        # symbol #411 -> side_idx 1 = No; settled at ~1.0 => No won.
        fills = pd.DataFrame(
            [
                {"ts_ns": 2, "symbol": "BTC #411", "price": 0.99, "side": "SELL", "size": 100.0},
            ]
        )
        assert _winner_from_settlement_fill(fills) == "no"

    def test_no_settlement_priced_fill(self) -> None:
        fills = pd.DataFrame(
            [
                {"ts_ns": 1, "symbol": "BTC #410", "price": 0.5, "side": "BUY", "size": 100.0},
            ]
        )
        assert _winner_from_settlement_fill(fills) is None

    def test_empty_fills(self) -> None:
        assert _winner_from_settlement_fill(pd.DataFrame()) is None


class TestSettlementWinnerParity:
    def test_parity_pass_derived_from_fill(self) -> None:
        """Live winner derived from settlement-as-fill matches sim resolved outcome."""
        live_fills = pd.DataFrame(
            [
                {"ts_ns": 1, "symbol": "BTC #410", "side": "BUY", "price": 0.85, "size": 100.0, "closed_pnl": 0.0},
                {"ts_ns": 2, "symbol": "BTC #410", "side": "SELL", "price": 1.0, "size": 100.0, "closed_pnl": 15.0},
            ]
        )
        res = reconcile_pnl(
            live_fills=live_fills,
            sim_fills=pd.DataFrame(),
            live_settlement={},  # no explicit winner -> derive from fill
            sim_resolved={"resolved_outcome": "yes"},
        )
        assert res.settlement_winner_match == "PASS"

    def test_parity_fail_derived_from_fill(self) -> None:
        live_fills = pd.DataFrame(
            [
                {"ts_ns": 2, "symbol": "BTC #410", "side": "SELL", "price": 1.0, "size": 100.0, "closed_pnl": 15.0},
            ]
        )
        res = reconcile_pnl(
            live_fills=live_fills,
            sim_fills=pd.DataFrame(),
            live_settlement={},
            sim_resolved={"resolved_outcome": "no"},  # sim disagrees
        )
        assert res.settlement_winner_match.startswith("FAIL")

    def test_parity_skip_when_neither_available(self) -> None:
        live_fills = pd.DataFrame(
            [
                {"ts_ns": 1, "symbol": "BTC #410", "side": "BUY", "price": 0.5, "size": 100.0, "closed_pnl": 0.0},
            ]
        )
        res = reconcile_pnl(
            live_fills=live_fills,
            sim_fills=pd.DataFrame(),
            live_settlement={},
            sim_resolved={},
        )
        assert res.settlement_winner_match == "SKIP:no_settlement"
