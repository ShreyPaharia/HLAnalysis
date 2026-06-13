"""SHR-42 (recorder twin-bug): a dying adapter must trip a clean process exit.

`_run_adapter` swallows exceptions and returns, so a crashed venue task simply
completes while `run()` waits forever on the stop event — the process stays
alive with that venue silently not recording and systemd never restarts it.
`_supervise` must return (and latch stop) as soon as any adapter task ends.
"""

from __future__ import annotations

import asyncio

import pytest

from hlanalysis.recorder.runner import _supervise


@pytest.mark.asyncio
async def test_supervise_exits_when_an_adapter_task_dies():
    stop = asyncio.Event()
    dead = asyncio.create_task(asyncio.sleep(0))  # completes "immediately"
    alive = asyncio.create_task(asyncio.sleep(10))  # would run forever

    await asyncio.wait_for(_supervise(stop, [dead, alive]), timeout=1.0)

    assert stop.is_set(), "a dead adapter task must latch stop for a clean exit"
    alive.cancel()


@pytest.mark.asyncio
async def test_supervise_returns_on_operator_stop():
    stop = asyncio.Event()
    alive = asyncio.create_task(asyncio.sleep(10))

    async def _signal():
        await asyncio.sleep(0.02)
        stop.set()

    await asyncio.gather(
        asyncio.wait_for(_supervise(stop, [alive]), timeout=1.0),
        _signal(),
    )

    assert stop.is_set()
    alive.cancel()
