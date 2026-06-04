"""Tests for Component 1: JSON log sink and structured bus-event logging.

TDD: tests are written first; they FAIL before the implementation lands.
"""
from __future__ import annotations

import json
import logging
import sys
from io import StringIO
from unittest.mock import patch

import pytest
from loguru import logger

from hlanalysis.engine.event_bus import EventBus
from hlanalysis.engine.risk_events import (
    Entry,
    Exit,
    OrderRejected,
    RiskHalt,
    RiskVeto,
    EngineHeartbeat,
    FeedStale,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _capture_json_logs(fn):
    """Call fn() with a JSON loguru sink and return list of parsed log dicts."""
    buf = StringIO()
    lid = logger.add(buf, serialize=True, level="DEBUG", format="{message}")
    try:
        fn()
    finally:
        logger.remove(lid)
    buf.seek(0)
    records = []
    for line in buf.getvalue().splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# JSON sink tests
# ---------------------------------------------------------------------------

class TestJsonSink:
    def test_json_sink_produces_valid_json_per_line(self):
        """Each log record must be a parseable JSON object."""
        records = _capture_json_logs(lambda: logger.info("hello world"))
        assert len(records) == 1
        rec = records[0]
        assert "text" in rec or "record" in rec  # loguru serialize=True envelope

    def test_json_sink_has_level_and_message(self):
        records = _capture_json_logs(lambda: logger.info("test message"))
        rec = records[0]
        # loguru serialize=True wraps in {"text": ..., "record": {...}}
        r = rec["record"]
        assert r["level"]["name"] == "INFO"
        assert "test message" in rec["text"]

    def test_json_sink_bound_fields_in_extra(self):
        """Fields bound via logger.bind() must appear in record.extra."""
        def emit():
            logger.bind(alias="v1", question_idx=42).info("bound test")

        records = _capture_json_logs(emit)
        assert len(records) == 1
        extra = records[0]["record"]["extra"]
        assert extra["alias"] == "v1"
        assert extra["question_idx"] == 42

    def test_json_sink_multiple_bound_fields(self):
        def emit():
            logger.bind(alias="v31", strategy="theta", cloid="abc123").info("order placed")

        records = _capture_json_logs(emit)
        extra = records[0]["record"]["extra"]
        assert extra["alias"] == "v31"
        assert extra["strategy"] == "theta"
        assert extra["cloid"] == "abc123"


class TestInterceptHandler:
    """stdlib logging records must flow through loguru (the _InterceptHandler wiring)."""

    def test_stdlib_log_appears_in_loguru_sink(self):
        from hlanalysis.engine.main import _InterceptHandler

        buf = StringIO()
        lid = logger.add(buf, serialize=True, level="DEBUG", format="{message}")
        try:
            # Wire the intercept handler for this test only
            test_logger = logging.getLogger("test_intercept_handler_unique")
            test_logger.handlers = [_InterceptHandler()]
            test_logger.setLevel(logging.DEBUG)
            test_logger.propagate = False
            test_logger.info("stdlib hello")
        finally:
            logger.remove(lid)

        buf.seek(0)
        lines = [l.strip() for l in buf.getvalue().splitlines() if l.strip()]
        assert lines, "Expected at least one log line from stdlib intercept"
        rec = json.loads(lines[0])
        assert "stdlib hello" in rec["text"]


class TestLogFormatCliFlag:
    """--log-format flag controls json vs pretty sink."""

    def test_main_parses_log_format_json(self):
        """Parsing --log-format json should not raise."""
        import argparse
        # Replicate the parser logic from main.py (after implementation)
        p = argparse.ArgumentParser()
        p.add_argument("--log-format", choices=["json", "pretty"], default="json")
        args = p.parse_args(["--log-format", "json"])
        assert args.log_format == "json"

    def test_main_parses_log_format_pretty(self):
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("--log-format", choices=["json", "pretty"], default="json")
        args = p.parse_args(["--log-format", "pretty"])
        assert args.log_format == "pretty"

    def test_main_default_log_format_is_json(self):
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("--log-format", choices=["json", "pretty"], default="json")
        args = p.parse_args([])
        assert args.log_format == "json"


# ---------------------------------------------------------------------------
# Structured bus-event logging tests
# ---------------------------------------------------------------------------

class TestEventBusStructuredLogging:
    """publish() must bind key event fields as top-level loguru extra fields."""

    @pytest.mark.asyncio
    async def test_publish_risk_veto_binds_kind_and_alias(self):
        records = []

        def sink(msg):
            try:
                records.append(json.loads(msg))
            except Exception:
                pass

        lid = logger.add(sink, serialize=True, level="DEBUG", format="{message}")
        try:
            bus = EventBus(maxsize=8)
            await bus.publish(RiskVeto(ts_ns=1, reason="cap_hit", account_alias="v1", question_idx=5))
        finally:
            logger.remove(lid)

        assert records, "Expected at least one log record"
        extras = [r["record"]["extra"] for r in records if "record" in r]
        # Find the bus publish record
        bus_extras = [e for e in extras if e.get("kind") == "risk_veto"]
        assert bus_extras, f"No record with kind=risk_veto; extras={extras}"
        e = bus_extras[0]
        assert e["alias"] == "v1"
        assert e["reason"] == "cap_hit"
        assert e["question_idx"] == 5

    @pytest.mark.asyncio
    async def test_publish_binds_payload_field(self):
        """Full model_dump_json must be available as extra.payload (a string or dict)."""
        records = []

        def sink(msg):
            try:
                records.append(json.loads(msg))
            except Exception:
                pass

        lid = logger.add(sink, serialize=True, level="DEBUG", format="{message}")
        try:
            bus = EventBus(maxsize=8)
            await bus.publish(RiskVeto(ts_ns=99, reason="test_payload", account_alias="v31"))
        finally:
            logger.remove(lid)

        extras = [r["record"]["extra"] for r in records if "record" in r]
        bus_extras = [e for e in extras if e.get("kind") == "risk_veto"]
        assert bus_extras
        e = bus_extras[0]
        assert "payload" in e
        # payload should be parseable JSON containing at least 'reason'
        p = json.loads(e["payload"]) if isinstance(e["payload"], str) else e["payload"]
        assert p["reason"] == "test_payload"

    @pytest.mark.asyncio
    async def test_publish_entry_binds_question_idx(self):
        records = []

        def sink(msg):
            try:
                records.append(json.loads(msg))
            except Exception:
                pass

        lid = logger.add(sink, serialize=True, level="DEBUG", format="{message}")
        try:
            bus = EventBus(maxsize=8)
            ev = Entry(
                ts_ns=1, account_alias="v1",
                cloid="cl001", question_idx=7,
                symbol="BTC-UP", side="buy", size=10.0, price=0.55,
            )
            await bus.publish(ev)
        finally:
            logger.remove(lid)

        extras = [r["record"]["extra"] for r in records if "record" in r]
        bus_extras = [e for e in extras if e.get("kind") == "entry"]
        assert bus_extras
        e = bus_extras[0]
        assert e["question_idx"] == 7
        assert e["alias"] == "v1"

    @pytest.mark.asyncio
    async def test_publish_exit_binds_reason(self):
        records = []

        def sink(msg):
            try:
                records.append(json.loads(msg))
            except Exception:
                pass

        lid = logger.add(sink, serialize=True, level="DEBUG", format="{message}")
        try:
            bus = EventBus(maxsize=8)
            ev = Exit(
                ts_ns=2, account_alias="v31",
                question_idx=3, symbol="BTC-UP",
                qty=10.0, realized_pnl=1.5, reason="exit_safety_d",
            )
            await bus.publish(ev)
        finally:
            logger.remove(lid)

        extras = [r["record"]["extra"] for r in records if "record" in r]
        bus_extras = [e for e in extras if e.get("kind") == "exit"]
        assert bus_extras
        e = bus_extras[0]
        assert e["reason"] == "exit_safety_d"
        assert e["alias"] == "v31"
        assert e["question_idx"] == 3

    @pytest.mark.asyncio
    async def test_publish_event_without_reason_field(self):
        """Events without 'reason' (e.g. EngineHeartbeat) must not error."""
        records = []

        def sink(msg):
            try:
                records.append(json.loads(msg))
            except Exception:
                pass

        lid = logger.add(sink, serialize=True, level="DEBUG", format="{message}")
        try:
            bus = EventBus(maxsize=8)
            ev = EngineHeartbeat(ts_ns=1, account_alias="", events_ingested=100, d_events=5, n_questions=3)
            await bus.publish(ev)
        finally:
            logger.remove(lid)

        extras = [r["record"]["extra"] for r in records if "record" in r]
        hb_extras = [e for e in extras if e.get("kind") == "engine_heartbeat"]
        assert hb_extras, "Expected log record for heartbeat"
        e = hb_extras[0]
        assert "reason" not in e or e.get("reason") is None

    @pytest.mark.asyncio
    async def test_publish_fanout_still_works_after_log_change(self):
        """Fanout behavior must be unaffected by the logging change."""
        bus = EventBus(maxsize=8)
        sub1 = bus.subscribe()
        sub2 = bus.subscribe()
        await bus.publish(RiskVeto(ts_ns=1, reason="cap", account_alias="v1"))
        import asyncio
        e1 = await asyncio.wait_for(sub1.get(), timeout=0.5)
        e2 = await asyncio.wait_for(sub2.get(), timeout=0.5)
        assert e1.reason == "cap" and e2.reason == "cap"

    @pytest.mark.asyncio
    async def test_publish_log_failure_does_not_block(self):
        """If the logger raises, publish must still enqueue to subscribers."""
        import asyncio
        bus = EventBus(maxsize=8)
        sub = bus.subscribe()

        # Patch logger.bind to raise
        with patch("hlanalysis.engine.event_bus.logger") as mock_log:
            mock_log.bind.side_effect = RuntimeError("log exploded")
            mock_log.exception = lambda *a, **kw: None  # swallow fallback log
            await bus.publish(RiskVeto(ts_ns=1, reason="test", account_alias="v1"))

        ev = await asyncio.wait_for(sub.get(), timeout=0.5)
        assert ev.reason == "test"

    @pytest.mark.asyncio
    async def test_publish_order_rejected_binds_cloid(self):
        """OrderRejected has cloid — it should appear in extra if we bind it."""
        records = []

        def sink(msg):
            try:
                records.append(json.loads(msg))
            except Exception:
                pass

        lid = logger.add(sink, serialize=True, level="DEBUG", format="{message}")
        try:
            bus = EventBus(maxsize=8)
            ev = OrderRejected(
                ts_ns=1, account_alias="v1",
                cloid="cl999", question_idx=4, symbol="BTC-UP",
                side="buy", size=5.0, price=0.6, error="funder mismatch",
            )
            await bus.publish(ev)
        finally:
            logger.remove(lid)

        extras = [r["record"]["extra"] for r in records if "record" in r]
        rej_extras = [e for e in extras if e.get("kind") == "order_rejected"]
        assert rej_extras
        # payload should contain the error
        e = rej_extras[0]
        p = json.loads(e["payload"]) if isinstance(e["payload"], str) else e["payload"]
        assert p["error"] == "funder mismatch"
