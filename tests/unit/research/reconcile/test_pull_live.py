"""Offline tests for pull_live (SSM mocked) + settlement-winner parity.

These never touch real AWS/SSM: ``_ssm_python`` is monkeypatched to capture the
generated remote script and return canned JSON.
"""

from __future__ import annotations

import gzip
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


def _mock_s3_trace(monkeypatch: pytest.MonkeyPatch, rows: list[dict]) -> _Recorder:
    """Wire up the SSM + S3 transport so pull_live_trace returns ``rows`` offline.

    ``_ssm_python`` captures the remote (upload) script; ``_s3_download_bytes``
    returns the gzipped JSON the box would have written; ``_s3_delete`` is a
    no-op. This exercises the S3-routed path WITHOUT the 24KB inline cap.
    """
    rec = _Recorder("S3_UPLOAD_OK")
    monkeypatch.setattr(pull_live, "_ssm_python", rec)
    monkeypatch.setattr(
        pull_live,
        "_s3_download_bytes",
        lambda bucket, key: gzip.compress(json.dumps(rows).encode()),
    )
    monkeypatch.setattr(pull_live, "_s3_delete", lambda bucket, key: None)
    # No sealed segments by default — keeps these tests hermetic (no real `aws`).
    monkeypatch.setattr(pull_live, "_s3_list_keys", lambda bucket, prefix: [])
    return rec


class TestSlotScopedPaths:
    def test_trace_path_built_from_strategy_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rec = _mock_s3_trace(monkeypatch, [])
        pull_live.pull_live_trace(4010, _EXPIRY_NS, strategy_id="v1")
        assert "/opt/hl-recorder/data/engine/v1/decision_trace.jsonl" in rec.scripts[0]

    def test_config_hash_path_built_from_strategy_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rec = _Recorder(json.dumps(None))
        monkeypatch.setattr(pull_live, "_ssm_python", rec)
        # Empty tail triggers the S3 fallback; keep it hermetic (no real `aws`).
        monkeypatch.setattr(pull_live, "_s3_list_keys", lambda bucket, prefix: [])
        pull_live.pull_config_hash(strategy_id="v1")
        assert "/opt/hl-recorder/data/engine/v1/decision_trace.jsonl" in rec.scripts[0]

    def test_config_hash_uses_tail_not_full_read(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """config_hash must read only the file tail (a 157MB full read times out)."""
        rec = _Recorder(json.dumps("deadbeef"))
        monkeypatch.setattr(pull_live, "_ssm_python", rec)
        assert pull_live.pull_config_hash(strategy_id="v31") == "deadbeef"
        assert "tail" in rec.scripts[0]


class TestPullLiveTraceS3Transport:
    """The live trace can be huge (137k+ rows / ~11MB). The SSM inline-output
    channel is hard-capped at 24,000 bytes, so the trace MUST be routed through
    S3: filter+gzip on the box, upload, download+decompress locally."""

    def test_trace_routed_through_s3_upload(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rec = _mock_s3_trace(monkeypatch, [{"question_idx": 4010, "ts_ns": 1, "action": "hold"}])
        df = pull_live.pull_live_trace(4010, _EXPIRY_NS, strategy_id="v31")
        # The remote script gzips + uploads to S3 (no inline return of rows).
        assert "put_object" in rec.scripts[0]
        assert "gzip" in rec.scripts[0]
        assert len(df) == 1
        assert df.iloc[0]["action"] == "hold"

    def test_trace_large_payload_roundtrips(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A payload far larger than the 24KB inline cap must come back intact."""
        rows = [{"question_idx": 4010, "ts_ns": i, "action": "hold"} for i in range(5000)]
        _mock_s3_trace(monkeypatch, rows)
        df = pull_live.pull_live_trace(4010, _EXPIRY_NS, strategy_id="v31")
        assert len(df) == 5000  # > 24KB of JSON; would truncate on the inline path

    def test_trace_cache_read_skips_remote(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        cache_file = tmp_path / "q4010_v31_trace.json"
        cache_file.write_text(json.dumps([{"question_idx": 4010, "ts_ns": 9, "action": "hold"}]))

        def _boom(*a, **k):
            raise AssertionError("must not hit SSM/S3 when cache exists")

        monkeypatch.setattr(pull_live, "_ssm_python", _boom)
        monkeypatch.setattr(pull_live, "_s3_download_bytes", _boom)
        df = pull_live.pull_live_trace(4010, _EXPIRY_NS, strategy_id="v31", cache_dir=tmp_path)
        assert len(df) == 1 and df.iloc[0]["ts_ns"] == 9


class TestPullLiveTraceS3SealedSegments:
    """After the box rotates+archives the trace, old rows live ONLY in S3 sealed
    segments (``engine/date=<d>/<alias>/traces/*.jsonl.gz``). pull_live_trace must
    union those with the live on-box file, filtered to the question + window."""

    def _wire(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        live_rows: list[dict],
        seg_rows: list[dict],
    ) -> list[str]:
        """live file -> live_rows (via SSM/S3 transport); one sealed segment ->
        seg_rows. Returns the list of prefixes passed to _s3_list_keys."""
        prefixes: list[str] = []
        monkeypatch.setattr(pull_live, "_ssm_python", _Recorder("S3_UPLOAD_OK"))
        monkeypatch.setattr(pull_live, "_s3_delete", lambda bucket, key: None)

        def _list(bucket: str, prefix: str) -> list[str]:
            prefixes.append(prefix)
            if "traces" in prefix:
                return [f"{prefix}decision_trace.20240610T064500.jsonl.gz"]
            return []

        def _dl(bucket: str, key: str) -> bytes:
            if "traces" in key:
                # Sealed segments are JSONL (one object per line), gzipped.
                return gzip.compress(("\n".join(json.dumps(r) for r in seg_rows) + "\n").encode())
            # The live-file transport blob is a JSON array (json.dumps(rows)).
            return gzip.compress(json.dumps(live_rows).encode())

        monkeypatch.setattr(pull_live, "_s3_list_keys", _list)
        monkeypatch.setattr(pull_live, "_s3_download_bytes", _dl)
        return prefixes

    def test_unions_live_file_and_sealed_segment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        live = [{"question_idx": 4010, "ts_ns": _EXPIRY_NS - 10, "action": "enter"}]
        seg = [{"question_idx": 4010, "ts_ns": _EXPIRY_NS - 100_000, "action": "hold"}]
        self._wire(monkeypatch, live_rows=live, seg_rows=seg)
        df = pull_live.pull_live_trace(4010, _EXPIRY_NS, strategy_id="v31")
        assert len(df) == 2
        assert set(df["action"]) == {"enter", "hold"}
        # Output is time-ordered for downstream alignment.
        assert list(df["ts_ns"]) == sorted(df["ts_ns"])

    def test_dedupes_overlap_between_live_and_segment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        dup = {"question_idx": 4010, "ts_ns": _EXPIRY_NS - 50, "action": "hold"}
        self._wire(monkeypatch, live_rows=[dup], seg_rows=[dict(dup)])
        df = pull_live.pull_live_trace(4010, _EXPIRY_NS, strategy_id="v31")
        assert len(df) == 1

    def test_segment_rows_filtered_by_window_and_question(self, monkeypatch: pytest.MonkeyPatch) -> None:
        before = {"question_idx": 4010, "ts_ns": _EXPIRY_NS - 48 * 3600 * 1_000_000_000, "action": "hold"}
        other_q = {"question_idx": 999, "ts_ns": _EXPIRY_NS - 60, "action": "hold"}
        keep = {"question_idx": 4010, "ts_ns": _EXPIRY_NS - 60, "action": "enter"}
        self._wire(monkeypatch, live_rows=[], seg_rows=[before, other_q, keep])
        df = pull_live.pull_live_trace(4010, _EXPIRY_NS, strategy_id="v31")
        assert len(df) == 1
        assert df.iloc[0]["action"] == "enter"

    def test_lists_engine_date_traces_prefix_for_window(self, monkeypatch: pytest.MonkeyPatch) -> None:
        prefixes = self._wire(monkeypatch, live_rows=[], seg_rows=[])
        pull_live.pull_live_trace(4010, _EXPIRY_NS, strategy_id="v31")
        trace_prefixes = [p for p in prefixes if "traces" in p]
        assert trace_prefixes, "expected at least one traces/ prefix listed"
        for p in trace_prefixes:
            assert p.startswith("engine/date=")
            assert p.endswith("/v31/traces/")
        # Window spans up to 24h before expiry → must cover ≥2 date partitions.
        assert len({p.split("date=")[1].split("/")[0] for p in trace_prefixes}) >= 2


class TestPullConfigHashS3Fallback:
    def test_falls_back_to_sealed_segment_when_live_tail_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Post-rotation the live file is often freshly rotated/empty, so the tail
        yields no config_hash; fall back to the newest sealed S3 segment."""
        # Live-file tail returns null (no rows).
        monkeypatch.setattr(pull_live, "_ssm_python", _Recorder(json.dumps(None)))
        monkeypatch.setattr(
            pull_live,
            "_s3_list_keys",
            lambda bucket, prefix: [f"{prefix}decision_trace.20240610T064500.jsonl.gz"] if "traces" in prefix else [],
        )
        monkeypatch.setattr(
            pull_live,
            "_s3_download_bytes",
            lambda bucket, key: gzip.compress(
                (json.dumps({"question_idx": 4010, "ts_ns": 1, "config_hash": "cafef00d12345678"}) + "\n").encode()
            ),
        )
        assert pull_live.pull_config_hash(strategy_id="v31") == "cafef00d12345678"


class TestPullLiveTraceMisc:
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
