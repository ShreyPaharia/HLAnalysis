from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from hlanalysis.alerts.telegram import TelegramClient


class _FakeResp:
    def __init__(self, status: int, body: dict | None = None) -> None:
        self.status = status
        self._body = body or {"ok": True}

    async def json(self) -> dict:
        return self._body

    async def __aenter__(self) -> "_FakeResp":
        return self

    async def __aexit__(self, *exc) -> None:
        return None


class _FakeSession:
    def __init__(self, responses: list[_FakeResp]) -> None:
        self._resps = responses
        self.calls: list[tuple[str, dict]] = []

    def post(self, url: str, json: dict | None = None, timeout: object | None = None):
        self.calls.append((url, json or {}))
        return self._resps.pop(0)

    async def close(self) -> None:
        pass


@pytest.mark.asyncio
async def test_send_uses_correct_url_and_payload():
    session = _FakeSession([_FakeResp(200, {"ok": True})])
    client = TelegramClient(bot_token="abc", chat_id="123", session=session)
    await client.send("hello")
    assert session.calls[0][0] == "https://api.telegram.org/botabc/sendMessage"
    payload = session.calls[0][1]
    assert payload["chat_id"] == "123"
    assert payload["text"] == "hello"
    assert payload["parse_mode"] == "Markdown"


@pytest.mark.asyncio
async def test_send_retries_on_5xx_and_succeeds():
    session = _FakeSession([_FakeResp(503, {}), _FakeResp(200, {"ok": True})])
    client = TelegramClient(bot_token="abc", chat_id="123", session=session,
                             max_retries=2, retry_delay_s=0.0)
    ok = await client.send("hi")
    assert ok is True
    assert len(session.calls) == 2


@pytest.mark.asyncio
async def test_send_gives_up_after_max_retries():
    session = _FakeSession([_FakeResp(500, {}), _FakeResp(500, {}), _FakeResp(500, {})])
    client = TelegramClient(bot_token="abc", chat_id="123", session=session,
                             max_retries=2, retry_delay_s=0.0)
    ok = await client.send("hi")
    assert ok is False
