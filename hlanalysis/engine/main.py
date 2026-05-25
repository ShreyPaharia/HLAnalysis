from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from loguru import logger

from ..adapters.composite import CompositeAdapter
from ..adapters.hyperliquid import HyperliquidAdapter
from ..adapters.polymarket import PolymarketAdapter
from ..config import load_config
from .config import load_deploy_config, load_strategies_config
from .runtime import EngineRuntime


class _InterceptHandler(logging.Handler):
    """Route stdlib logging records into loguru so adapter / SDK logs land
    in the same stream as engine logs."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def main() -> None:
    p = argparse.ArgumentParser(description="HLAnalysis MM engine (Phase 1)")
    p.add_argument("--strategy-config", type=Path, default=Path("config/strategy.yaml"))
    p.add_argument("--deploy-config", type=Path, default=Path("config/deploy.yaml"))
    p.add_argument(
        "--symbols-config", type=Path, default=Path("config/symbols.yaml"),
        help="Subscriptions feeding MarketState (HIP-4 binaries + BTC perp).",
    )
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logger.remove()
    logger.add(sys.stderr, level=args.log_level)
    # Route stdlib logging (used by adapters/hyperliquid.py + SDK + asyncio)
    # into loguru. Without this, adapter INFO logs are silently dropped and
    # only WARNING+ leaks to stderr via Python's lastResort handler.
    logging.basicConfig(handlers=[_InterceptHandler()], level=args.log_level.upper(), force=True)

    strategies_cfg = load_strategies_config(args.strategy_config)
    deploy_cfg = load_deploy_config(args.deploy_config)
    sym_cfg = load_config(args.symbols_config)

    # Engine consumes HL + PM subscriptions; Binance stays recorder-only.
    engine_subs = [
        s for s in sym_cfg.subscriptions
        if s.venue in ("hyperliquid", "polymarket")
    ]

    def _composite_factory() -> CompositeAdapter:
        # One merged stream over HL + PM. Each child adapter only receives
        # its own venue's subs (CompositeAdapter filters by `venue`).
        return CompositeAdapter([HyperliquidAdapter(), PolymarketAdapter()])

    runtime = EngineRuntime(
        strategies=strategies_cfg.strategies,
        deploy_cfg=deploy_cfg,
        adapter_factory=_composite_factory,
        subscriptions=engine_subs,
    )
    for s in strategies_cfg.strategies:
        if s.paper_mode:
            logger.warning(
                "PAPER MODE alias={} ({}) — no real orders will be placed",
                s.account_alias, s.strategy_type,
            )
        else:
            # warning, not error: this is a startup banner, not a failure
            # condition. Logging at ERROR pollutes journalctl ERROR filters
            # and any alerting hooked off the loguru level.
            logger.warning(
                "LIVE MODE alias={} ({}) — real money at stake; "
                "ensure caps in strategy.yaml are correct",
                s.account_alias, s.strategy_type,
            )

    asyncio.run(runtime.run())


if __name__ == "__main__":
    main()
