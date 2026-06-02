from __future__ import annotations

import asyncio
import logging
import signal
from collections import defaultdict
from pathlib import Path

from ..adapters.base import VenueAdapter
from ..adapters.binance import BinanceAdapter
from ..adapters.hyperliquid import HyperliquidAdapter
from ..adapters.polymarket import PolymarketAdapter
from ..config import Subscription, load_config
from .writer import ParquetWriter

log = logging.getLogger(__name__)

ADAPTERS: dict[str, type[VenueAdapter]] = {
    "hyperliquid": HyperliquidAdapter,
    "binance": BinanceAdapter,
    "polymarket": PolymarketAdapter,
}


async def _run_adapter(
    adapter: VenueAdapter,
    subs: list[Subscription],
    writer: ParquetWriter,
    stop: asyncio.Event,
) -> None:
    # Iterate directly: do NOT wrap in asyncio.wait_for. Cancelling __anext__ propagates
    # CancelledError out of the adapter generator (it's a BaseException), which terminates
    # the generator permanently and kills recording. Cancellation is handled in run() via
    # task.cancel() in the finally block.
    try:
        async for event in adapter.stream(subs):
            if stop.is_set():
                return
            writer.write(event.model_dump(mode="python"))
    except asyncio.CancelledError:
        raise
    except Exception:
        log.exception("adapter %s crashed", adapter.venue)


async def _flusher(writer: ParquetWriter, stop: asyncio.Event) -> None:
    while not stop.is_set():
        await asyncio.sleep(1.0)
        writer.maybe_flush()


async def _supervise(
    stop: asyncio.Event, adapter_tasks: list[asyncio.Task]
) -> None:
    """Return when the operator signals stop OR any adapter task ends (SHR-42).

    `_run_adapter` swallows exceptions and returns, so a crashed venue task just
    completes — adapters are meant to stream forever, so a completed adapter
    task means that venue is silently no longer recording. Waiting only on
    `stop` would leave the process alive with a dead feed and systemd would
    never restart it. We wait on stop OR the first adapter to finish, then latch
    stop so `run()` falls through to a clean shutdown + restart."""
    stop_waiter = asyncio.ensure_future(stop.wait())
    try:
        await asyncio.wait(
            {stop_waiter, *adapter_tasks},
            return_when=asyncio.FIRST_COMPLETED,
        )
    finally:
        stop_waiter.cancel()
    if not stop.is_set():
        n_dead = sum(1 for t in adapter_tasks if t.done())
        log.error(
            "recorder adapter task exited unexpectedly (%d done); shutting down "
            "for supervisor restart", n_dead,
        )
        stop.set()


async def run(config_path: Path, data_root: Path) -> None:
    config = load_config(config_path)
    # 60s time-trigger; per-key row cap stays at the writer default (5000).
    # HL/Binance saturate the row cap within seconds and never hit the timer
    # trigger; the 60s setting only matters for low-volume venues like
    # Polymarket where book/trade keys would otherwise sit in memory until
    # the row cap (essentially never). Live small files are merged hourly by
    # `scripts/compact-data.sh` (called from the S3 sync timer). Hard-crash
    # data-loss bound: ~60s.
    writer = ParquetWriter(data_root, flush_interval_s=60.0)

    by_venue: dict[str, list[Subscription]] = defaultdict(list)
    for sub in config.subscriptions:
        by_venue[sub.venue].append(sub)

    stop = asyncio.Event()

    def _signal_handler() -> None:
        log.info("stop signal received")
        stop.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    tasks: list[asyncio.Task] = []
    adapter_tasks: list[asyncio.Task] = []
    for venue, subs in by_venue.items():
        adapter_cls = ADAPTERS.get(venue)
        if adapter_cls is None:
            log.error("no adapter registered for venue=%s; skipping %d subs", venue, len(subs))
            continue
        adapter = adapter_cls()
        unsupported = [
            s for s in subs if not adapter.supports(s.product_type, s.mechanism)
        ]
        if unsupported:
            log.error(
                "%s does not support %d subs; skipping: %s",
                venue,
                len(unsupported),
                [(s.symbol, s.product_type, s.mechanism) for s in unsupported],
            )
        ok_subs = [s for s in subs if adapter.supports(s.product_type, s.mechanism)]
        if not ok_subs:
            continue
        log.info("starting %s with %d subscriptions", venue, len(ok_subs))
        atask = asyncio.create_task(_run_adapter(adapter, ok_subs, writer, stop))
        adapter_tasks.append(atask)
        tasks.append(atask)

    tasks.append(asyncio.create_task(_flusher(writer, stop)))

    try:
        await _supervise(stop, adapter_tasks)
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        writer.flush_all()
        log.info("recorder stopped, all buffers flushed")
