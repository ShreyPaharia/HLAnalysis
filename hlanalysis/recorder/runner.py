from __future__ import annotations

import asyncio
import logging
import signal
from collections import defaultdict
from pathlib import Path

from ..adapters.base import VenueAdapter
from ..adapters.binance import BinanceAdapter
from ..adapters.hyperliquid import HyperliquidAdapter
from ..config import Subscription, load_config
from .writer import ParquetWriter

log = logging.getLogger(__name__)

ADAPTERS: dict[str, type[VenueAdapter]] = {
    "hyperliquid": HyperliquidAdapter,
    "binance": BinanceAdapter,
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


async def run(config_path: Path, data_root: Path) -> None:
    config = load_config(config_path)
    writer = ParquetWriter(data_root)

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
        tasks.append(asyncio.create_task(_run_adapter(adapter, ok_subs, writer, stop)))

    tasks.append(asyncio.create_task(_flusher(writer, stop)))

    try:
        await stop.wait()
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        writer.flush_all()
        log.info("recorder stopped, all buffers flushed")
