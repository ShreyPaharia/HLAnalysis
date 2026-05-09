from __future__ import annotations

import asyncio
from typing import Protocol

from loguru import logger


class _SessionLike(Protocol):
    def post(self, url: str, json: dict | None = ..., timeout: object | None = ...): ...
    async def close(self) -> None: ...


class TelegramClient:
    """Outbound-only Telegram client. ~40 LOC.

    Constructed with an injected aiohttp.ClientSession-compatible object so
    tests can swap in a fake. The runtime owns the real ClientSession lifecycle.
    """

    def __init__(
        self,
        *,
        bot_token: str,
        chat_id: str,
        session: _SessionLike,
        max_retries: int = 2,
        retry_delay_s: float = 0.5,
        request_timeout_s: float = 5.0,
    ) -> None:
        self._url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        self._chat_id = chat_id
        self._session = session
        self._max_retries = max_retries
        self._retry_delay = retry_delay_s
        self._timeout_s = request_timeout_s

    async def send(self, text: str, *, markdown: bool = True) -> bool:
        # markdown=True now means HTML mode (stricter than legacy Markdown but
        # has unambiguous escape semantics: only <, >, & need escaping). Renamed
        # to keep the call sites compatible.
        payload = {
            "chat_id": self._chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        for attempt in range(self._max_retries + 1):
            try:
                async with self._session.post(self._url, json=payload, timeout=self._timeout_s) as r:
                    if 200 <= r.status < 300:
                        return True
                    if 400 <= r.status < 500:
                        body = await r.json()
                        logger.warning("telegram 4xx; not retrying status={} body={}", r.status, body)
                        return False
                    logger.warning("telegram {} on attempt {}", r.status, attempt + 1)
            except Exception as e:
                logger.warning("telegram error attempt={} err={}", attempt + 1, e)
            if attempt < self._max_retries:
                await asyncio.sleep(self._retry_delay)
        return False

    async def close(self) -> None:
        await self._session.close()
